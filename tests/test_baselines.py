"""Unit tests for src/hgc/baselines.py.

All tests use fake embedders and a fake ReAct factory — no real API calls.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from hgc.baselines import (
    AnswerCacheAgent,
    LongContextStuffAgent,
    NaiveRAGAgent,
    TrajectoryCacheAgent,
    _build_demo_prefix,
    _trajectory_to_summary,
)
from hgc.emb_cache import DocEmbeddingCache

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


def _unit_emb(n: int, dims: int = 8) -> np.ndarray:
    """Return a deterministic unit-norm float32 vector of length *dims*."""
    v = np.zeros(dims, dtype=np.float32)
    v[n % dims] = 1.0
    return v


def _make_embedder(*vecs: np.ndarray):
    """Return a fake embedder that cycles through *vecs* in call order."""
    calls = iter(vecs)

    def embedder(text: str) -> np.ndarray:  # noqa: ARG001
        return next(calls)

    return embedder


def _always_correct(question: str, answer: str) -> bool:  # noqa: ARG001
    return True


def _always_wrong(question: str, answer: str) -> bool:  # noqa: ARG001
    return False


class _FakeReActResult:
    """Mimics the object returned by dspy.ReAct.__call__."""

    def __init__(self, answer: str = "fake_answer", trajectory: dict | None = None) -> None:
        self.answer = answer
        self.trajectory = trajectory or {
            "thought_0": "I need to find the answer.",
            "tool_name_0": "search",
            "tool_args_0": {"query": "test"},
            "observation_0": "Some result",
        }
        self.tokens = 42


def _make_react_factory(answer: str = "fake_answer", trajectory: dict | None = None):
    """Return a factory that always produces a FakeReAct returning *answer*."""
    result = _FakeReActResult(answer=answer, trajectory=trajectory)

    class _FakeReAct:
        def __call__(self, question: str) -> _FakeReActResult:  # noqa: ARG002
            return result

    def factory(tools, max_iters, prefix=""):  # noqa: ARG001
        return _FakeReAct()

    return factory


# ---------------------------------------------------------------------------
# AnswerCacheAgent tests
# ---------------------------------------------------------------------------


class TestAnswerCacheAgentCacheHit:
    """Seeding with an identical embedding then querying yields a cache hit."""

    def setup_method(self):
        emb_a = _unit_emb(0)
        emb_b = _unit_emb(0)  # identical direction → cosine = 1.0

        # embedder returns emb_a for the first call (seed run) and emb_b for hit
        embedder = _make_embedder(emb_a, emb_b)

        self.agent = AnswerCacheAgent(
            tools=[],
            embedder=embedder,
            judge=_always_correct,
            sim_threshold=0.95,
            react_factory=_make_react_factory(answer="seeded_answer"),
        )
        # First run → cache miss, populates cache
        self.first_result = self.agent.run("question one")

    def test_first_run_is_cache_miss(self):
        assert self.first_result["cache_hit"] is False

    def test_first_run_has_answer(self):
        assert self.first_result["answer"] == "seeded_answer"

    def test_first_run_has_nonzero_tokens(self):
        assert self.first_result["tokens"] == 42

    def test_cache_size_after_first_run(self):
        assert self.agent.cache_size == 1

    def test_second_run_is_cache_hit(self):
        result = self.agent.run("question one paraphrased")
        assert result["cache_hit"] is True

    def test_cache_hit_returns_zero_tokens(self):
        result = self.agent.run("question one paraphrased")
        assert result["tokens"] == 0

    def test_cache_hit_returns_zero_iters(self):
        result = self.agent.run("question one paraphrased")
        assert result["n_iters"] == 0

    def test_cache_hit_returns_empty_trajectory(self):
        result = self.agent.run("question one paraphrased")
        assert result["trajectory"] == {}

    def test_cache_hit_returns_same_answer(self):
        result = self.agent.run("question one paraphrased")
        assert result["answer"] == "seeded_answer"


class TestAnswerCacheAgentCacheMiss:
    """A different embedding below threshold yields a cache miss and full run."""

    def test_different_emb_is_miss(self):
        emb_a = _unit_emb(0)  # first call: seed run
        emb_b = _unit_emb(1)  # orthogonal → cosine = 0.0

        embedder = _make_embedder(emb_a, emb_b)
        agent = AnswerCacheAgent(
            tools=[],
            embedder=embedder,
            judge=_always_correct,
            sim_threshold=0.95,
            react_factory=_make_react_factory(answer="first_answer"),
        )

        agent.run("question one")  # seed
        result = agent.run("unrelated question")  # should miss

        assert result["cache_hit"] is False

    def test_miss_runs_full_react(self):
        emb_a = _unit_emb(0)
        emb_b = _unit_emb(1)  # orthogonal

        embedder = _make_embedder(emb_a, emb_b)
        agent = AnswerCacheAgent(
            tools=[],
            embedder=embedder,
            judge=_always_correct,
            sim_threshold=0.95,
            react_factory=_make_react_factory(answer="second_answer"),
        )

        agent.run("question one")
        result = agent.run("totally different question")

        assert result["tokens"] == 42

    def test_wrong_judgment_does_not_cache(self):
        """On failed judgment, the answer is NOT added to the cache."""
        emb_a = _unit_emb(0)
        embedder = _make_embedder(emb_a)

        agent = AnswerCacheAgent(
            tools=[],
            embedder=embedder,
            judge=_always_wrong,  # judge says wrong
            sim_threshold=0.95,
            react_factory=_make_react_factory(),
        )
        agent.run("question")
        assert agent.cache_size == 0

    def test_empty_store_always_misses(self):
        emb_a = _unit_emb(0)
        embedder = _make_embedder(emb_a)

        agent = AnswerCacheAgent(
            tools=[],
            embedder=embedder,
            judge=_always_correct,
            sim_threshold=0.95,
            react_factory=_make_react_factory(),
        )
        result = agent.run("question")
        assert result["cache_hit"] is False


class TestAnswerCacheAgentRecordSchema:
    """run() returns a dict with all required fields."""

    def _run_once(self, judge=_always_correct):
        emb = _unit_emb(0)
        agent = AnswerCacheAgent(
            tools=[],
            embedder=_make_embedder(emb),
            judge=judge,
            sim_threshold=0.95,
            react_factory=_make_react_factory(),
        )
        return agent.run("q")

    def test_all_fields_present(self):
        result = self._run_once()
        assert set(result.keys()) >= {
            "answer",
            "trajectory",
            "judgment",
            "tokens",
            "wall_time",
            "n_iters",
            "cache_hit",
        }

    def test_wall_time_is_positive(self):
        result = self._run_once()
        assert result["wall_time"] >= 0.0

    def test_judgment_bool(self):
        result = self._run_once(judge=_always_correct)
        assert result["judgment"] is True


# ---------------------------------------------------------------------------
# TrajectoryCacheAgent tests
# ---------------------------------------------------------------------------


class TestTrajectoryCacheEmptyStore:
    """With no stored trajectories, no demo prefix is injected."""

    def test_empty_store_no_prefix(self):
        emb = _unit_emb(0)
        agent = TrajectoryCacheAgent(
            tools=[],
            embedder=_make_embedder(emb),
            judge=_always_correct,
            sim_threshold=0.7,
            top_k=3,
            react_factory=_make_react_factory(),
        )
        result = agent.run("question")
        assert result["demo_prefix"] == ""

    def test_empty_store_cache_hit_false(self):
        emb = _unit_emb(0)
        agent = TrajectoryCacheAgent(
            tools=[],
            embedder=_make_embedder(emb),
            judge=_always_correct,
            sim_threshold=0.7,
            top_k=3,
            react_factory=_make_react_factory(),
        )
        result = agent.run("question")
        assert result["cache_hit"] is False


class TestTrajectoryCacheDemoInjection:
    """Seeding with 2 similar trajectories injects them as a prefix."""

    def _make_agent_with_2_demos(self):
        """Seed agent with 2 trajectories then return (agent, embedder_state)."""
        # Embeddings for 3 calls:
        #   call 0: seed run 1 → emb[0]
        #   call 1: seed run 2 → emb[0]  (same direction)
        #   call 2: real query → emb[0]  (same direction → above threshold)
        emb_same = _unit_emb(0)
        embedder = _make_embedder(emb_same, emb_same, emb_same)

        factory = _make_react_factory(answer="correct_answer")

        agent = TrajectoryCacheAgent(
            tools=[],
            embedder=embedder,
            judge=_always_correct,
            sim_threshold=0.7,
            top_k=3,
            react_factory=factory,
        )
        agent.run("seed question 1")
        agent.run("seed question 2")
        return agent

    def test_store_has_2_entries_after_seeding(self):
        agent = self._make_agent_with_2_demos()
        assert agent.store_size == 2

    def test_prefix_contains_both_demos(self):
        agent = self._make_agent_with_2_demos()

        # One more embedding for the real query (same direction)
        emb_same = _unit_emb(0)

        # Replace embedder for final call
        agent._embedder = _make_embedder(emb_same)
        result = agent.run("real question")

        prefix = result["demo_prefix"]
        assert "seed question 1" in prefix
        assert "seed question 2" in prefix

    def test_store_grows_to_3_after_third_run(self):
        agent = self._make_agent_with_2_demos()
        emb_same = _unit_emb(0)
        agent._embedder = _make_embedder(emb_same)
        agent.run("real question")
        assert agent.store_size == 3


class TestTrajectoryCacheLowSimilarity:
    """Trajectories below threshold are not injected."""

    def test_orthogonal_emb_no_demos(self):
        emb_a = _unit_emb(0)  # seed run
        emb_b = _unit_emb(1)  # orthogonal → cosine = 0.0 < 0.7

        embedder = _make_embedder(emb_a, emb_b)
        agent = TrajectoryCacheAgent(
            tools=[],
            embedder=embedder,
            judge=_always_correct,
            sim_threshold=0.7,
            top_k=3,
            react_factory=_make_react_factory(),
        )
        agent.run("seed question")
        result = agent.run("very different question")
        assert result["demo_prefix"] == ""


class TestTrajectoryCacheRecordSchema:
    """run() returns a dict with all required fields."""

    def test_all_fields_present(self):
        emb = _unit_emb(0)
        agent = TrajectoryCacheAgent(
            tools=[],
            embedder=_make_embedder(emb),
            judge=_always_correct,
            sim_threshold=0.7,
            react_factory=_make_react_factory(),
        )
        result = agent.run("q")
        required = {
            "answer",
            "trajectory",
            "judgment",
            "tokens",
            "wall_time",
            "n_iters",
            "cache_hit",
            "demo_prefix",
        }
        assert set(result.keys()) >= required

    def test_cache_hit_always_false(self):
        """TrajectoryCacheAgent never sets cache_hit=True (it always runs ReAct)."""
        emb = _unit_emb(0)
        agent = TrajectoryCacheAgent(
            tools=[],
            embedder=_make_embedder(emb, emb),
            judge=_always_correct,
            sim_threshold=0.7,
            react_factory=_make_react_factory(),
        )
        agent.run("q1")
        result = agent.run("q2")
        assert result["cache_hit"] is False


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestTrajectoryToSummary:
    def test_empty_trajectory(self):
        summary = _trajectory_to_summary({})
        assert summary == ""

    def test_captures_thought_and_tool(self):
        traj = {
            "thought_0": "think",
            "tool_name_0": "search",
            "tool_args_0": {"q": "test"},
            "observation_0": "result",
        }
        summary = _trajectory_to_summary(traj)
        assert "think" in summary
        assert "search" in summary

    def test_respects_max_chars(self):
        long_thought = "x" * 2000
        traj = {"thought_0": long_thought}
        summary = _trajectory_to_summary(traj, max_chars=100)
        assert len(summary) <= 100


class TestBuildDemoPrefix:
    def test_empty_demos_returns_empty_string(self):
        assert _build_demo_prefix([]) == ""

    def test_single_demo_contains_question(self):
        prefix = _build_demo_prefix([("What is X?", "T: looked up X\nA: lookup(X)")])
        assert "What is X?" in prefix

    def test_multiple_demos_all_present(self):
        demos = [("Q1", "traj1"), ("Q2", "traj2")]
        prefix = _build_demo_prefix(demos)
        assert "Q1" in prefix
        assert "Q2" in prefix
        assert "Demo 1" in prefix
        assert "Demo 2" in prefix


# ---------------------------------------------------------------------------
# Mem0ReActAgent tests
# ---------------------------------------------------------------------------


class _FakeMem0Memory:
    """In-memory stub matching the mem0 v2 Memory API used by Mem0ReActAgent.

    Mirrors the real signatures:
        add(messages, *, user_id=None, ...)
        search(query, *, top_k=20, filters=None, ...)  -> {"results": [...]}
        get_all(*, filters=None, top_k=20, ...)        -> {"results": [...]}
    """

    def __init__(self, config=None) -> None:  # noqa: ARG002
        self._store: list[dict] = []

    def add(self, messages: str, *, user_id: str | None = None, **kwargs) -> None:  # noqa: ARG002
        self._store.append({"memory": messages, "user_id": user_id})

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict | None = None,
        **kwargs,  # noqa: ARG002
    ) -> dict:
        return {"results": self._store[:top_k]}

    def get_all(
        self,
        *,
        filters: dict | None = None,
        top_k: int = 20,
        **kwargs,  # noqa: ARG002
    ) -> dict:
        return {"results": list(self._store[:top_k])}


def _make_mem0_react_factory(answer: str = "fake_answer", trajectory: dict | None = None):
    """Return a factory that accepts (tools, max_iters) and produces a FakeReAct."""
    result = _FakeReActResult(answer=answer, trajectory=trajectory)

    class _FakeReAct:
        def __init__(self, captured_tools):
            self._tools = captured_tools

        def __call__(self, question: str) -> _FakeReActResult:  # noqa: ARG002
            return result

    def factory(tools, max_iters):  # noqa: ARG001
        return _FakeReAct(tools)

    return factory, _FakeReAct


class TestMem0ReActAgent:
    """Unit tests for Mem0ReActAgent using a fake Mem0 Memory stub."""

    def test_run_returns_expected_schema(self, monkeypatch):
        monkeypatch.setattr("hgc.baselines.Memory", _FakeMem0Memory)
        from hgc.baselines import Mem0ReActAgent

        factory, _ = _make_mem0_react_factory(answer="the_answer")
        agent = Mem0ReActAgent(
            tools=[],
            judge=_always_correct,
            user_id="u1",
            react_factory=factory,
        )
        result = agent.run("What is the capital of France?")

        required_keys = {
            "answer",
            "trajectory",
            "judgment",
            "tokens",
            "wall_time",
            "n_iters",
            "cache_hit",
        }
        assert required_keys <= set(
            result.keys()
        ), f"Missing keys: {required_keys - set(result.keys())}"

    def test_run_answer_value(self, monkeypatch):
        monkeypatch.setattr("hgc.baselines.Memory", _FakeMem0Memory)
        from hgc.baselines import Mem0ReActAgent

        factory, _ = _make_mem0_react_factory(answer="Paris")
        agent = Mem0ReActAgent(
            tools=[],
            judge=_always_correct,
            user_id="u1",
            react_factory=factory,
        )
        result = agent.run("What is the capital of France?")
        assert result["answer"] == "Paris"

    def test_cache_hit_always_false(self, monkeypatch):
        monkeypatch.setattr("hgc.baselines.Memory", _FakeMem0Memory)
        from hgc.baselines import Mem0ReActAgent

        factory, _ = _make_mem0_react_factory()
        agent = Mem0ReActAgent(
            tools=[],
            judge=_always_correct,
            user_id="u1",
            react_factory=factory,
        )
        result = agent.run("some question")
        assert result["cache_hit"] is False

    def test_judgment_passed_through(self, monkeypatch):
        monkeypatch.setattr("hgc.baselines.Memory", _FakeMem0Memory)
        from hgc.baselines import Mem0ReActAgent

        factory, _ = _make_mem0_react_factory()
        agent = Mem0ReActAgent(
            tools=[],
            judge=_always_wrong,
            user_id="u1",
            react_factory=factory,
        )
        result = agent.run("some question")
        assert result["judgment"] is False

    def test_wall_time_is_non_negative(self, monkeypatch):
        monkeypatch.setattr("hgc.baselines.Memory", _FakeMem0Memory)
        from hgc.baselines import Mem0ReActAgent

        factory, _ = _make_mem0_react_factory()
        agent = Mem0ReActAgent(
            tools=[],
            judge=_always_correct,
            user_id="u1",
            react_factory=factory,
        )
        result = agent.run("q")
        assert result["wall_time"] >= 0.0

    def test_exposes_exactly_three_mem0_tools(self, monkeypatch):
        monkeypatch.setattr("hgc.baselines.Memory", _FakeMem0Memory)
        from hgc.baselines import Mem0ReActAgent

        factory, _ = _make_mem0_react_factory()
        agent = Mem0ReActAgent(
            tools=[],
            judge=_always_correct,
            user_id="u1",
            react_factory=factory,
        )
        assert len(agent._mem0_tools) == 3

    def test_mem0_tool_names(self, monkeypatch):
        monkeypatch.setattr("hgc.baselines.Memory", _FakeMem0Memory)
        from hgc.baselines import Mem0ReActAgent

        factory, _ = _make_mem0_react_factory()
        agent = Mem0ReActAgent(
            tools=[],
            judge=_always_correct,
            user_id="u1",
            react_factory=factory,
        )
        tool_names = {fn.__name__ for fn in agent._mem0_tools}
        assert tool_names == {"store_memory", "search_memories", "get_all_memories"}

    def test_mem0_tools_passed_to_react(self, monkeypatch):
        monkeypatch.setattr("hgc.baselines.Memory", _FakeMem0Memory)
        from hgc.baselines import Mem0ReActAgent

        captured_tools = []

        def capturing_factory(tools, max_iters):  # noqa: ARG001
            captured_tools.extend(tools)

            class _FakeReAct:
                def __call__(self, question):  # noqa: ARG002
                    return _FakeReActResult()

            return _FakeReAct()

        agent = Mem0ReActAgent(
            tools=[],  # no user tools
            judge=_always_correct,
            user_id="u1",
            react_factory=capturing_factory,
        )
        agent.run("q")
        assert len(captured_tools) == 3

    def test_import_error_when_mem0ai_missing(self, monkeypatch):
        monkeypatch.setattr("hgc.baselines.Memory", None)
        from hgc.baselines import Mem0ReActAgent

        with pytest.raises(ImportError, match="mem0ai not installed"):
            Mem0ReActAgent(tools=[], judge=_always_correct)


# ---------------------------------------------------------------------------
# NaiveRAGAgent tests
# ---------------------------------------------------------------------------


def _text_hash_vec(text: str, dims: int = 8) -> np.ndarray:
    """Return a deterministic unit-norm float32 vector keyed by text hash."""
    digest = int(hashlib.md5(text.encode()).hexdigest(), 16)
    v = np.zeros(dims, dtype=np.float32)
    v[digest % dims] = 1.0
    return v


class _FakeEmbedder:
    """Fake embedder: embed() and embed_batch() return deterministic vectors."""

    DIMS = 8

    def __init__(self) -> None:
        self.embed_batch_call_count = 0

    def embed(self, text: str) -> np.ndarray:
        return _text_hash_vec(text, self.DIMS)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.embed_batch_call_count += 1
        if not texts:
            return np.empty((0, self.DIMS), dtype=np.float32)
        return np.stack([_text_hash_vec(t, self.DIMS) for t in texts])


class _FakeLM:
    """Fake LM that returns a canned answer string and records call count."""

    def __init__(self, answer: str = "canned_answer") -> None:
        self._answer = answer
        self.call_count = 0

    def __call__(self, prompt: str) -> list[str]:  # noqa: ARG002
        self.call_count += 1
        return [self._answer]


def _make_rag_docs(n: int) -> list[dict]:
    """Return *n* synthetic doc dicts with unique text."""
    return [
        {"docid": f"doc_{i}", "text": f"Document {i} content about topic {i}."} for i in range(n)
    ]


class TestNaiveRAGAgentRetrieval:
    """Given 5 docs and top_k=3, run() retrieves exactly the 3 most similar."""

    def test_retrieves_top_k_docs(self):
        docs = _make_rag_docs(5)
        fake_lm = _FakeLM()
        agent = NaiveRAGAgent(
            embedder=_FakeEmbedder(),
            judge=_always_correct,
            lm=fake_lm,
            top_k=3,
        )
        result = agent.run("some question", docs)
        assert len(result["trajectory"]["retrieved_docids"]) == 3

    def test_lm_called_exactly_once(self):
        docs = _make_rag_docs(5)
        fake_lm = _FakeLM()
        agent = NaiveRAGAgent(
            embedder=_FakeEmbedder(),
            judge=_always_correct,
            lm=fake_lm,
            top_k=3,
        )
        agent.run("some question", docs)
        assert fake_lm.call_count == 1


class TestNaiveRAGAgentSchema:
    """run() returns a record with all required schema keys."""

    def _run_once(self, judge=_always_correct) -> dict:
        docs = _make_rag_docs(5)
        fake_lm = _FakeLM()
        agent = NaiveRAGAgent(
            embedder=_FakeEmbedder(),
            judge=judge,
            lm=fake_lm,
            top_k=3,
        )
        return agent.run("test question", docs)

    def test_all_schema_keys_present(self):
        result = self._run_once()
        required = {
            "answer",
            "trajectory",
            "judgment",
            "tokens",
            "wall_time",
            "n_iters",
            "cache_hit",
        }
        assert required <= set(result.keys())

    def test_n_iters_is_one(self):
        result = self._run_once()
        assert result["n_iters"] == 1

    def test_cache_hit_is_false(self):
        result = self._run_once()
        assert result["cache_hit"] is False

    def test_wall_time_non_negative(self):
        result = self._run_once()
        assert result["wall_time"] >= 0.0


class TestNaiveRAGAgentJudgment:
    """judgment reflects the judge's verdict."""

    def test_judgment_correct(self):
        docs = _make_rag_docs(3)
        agent = NaiveRAGAgent(
            embedder=_FakeEmbedder(),
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=2,
        )
        result = agent.run("q", docs)
        assert result["judgment"] is True

    def test_judgment_wrong(self):
        docs = _make_rag_docs(3)
        agent = NaiveRAGAgent(
            embedder=_FakeEmbedder(),
            judge=_always_wrong,
            lm=_FakeLM(),
            top_k=2,
        )
        result = agent.run("q", docs)
        assert result["judgment"] is False


class TestNaiveRAGAgentTrajectory:
    """trajectory["retrieved_docids"] has exactly top_k entries."""

    def test_retrieved_docids_count(self):
        docs = _make_rag_docs(5)
        agent = NaiveRAGAgent(
            embedder=_FakeEmbedder(),
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=3,
        )
        result = agent.run("q", docs)
        assert len(result["trajectory"]["retrieved_docids"]) == 3

    def test_retrieved_docids_are_subset_of_input(self):
        docs = _make_rag_docs(5)
        input_docids = {d["docid"] for d in docs}
        agent = NaiveRAGAgent(
            embedder=_FakeEmbedder(),
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=3,
        )
        result = agent.run("q", docs)
        assert set(result["trajectory"]["retrieved_docids"]) <= input_docids

    def test_prompt_preview_in_trajectory(self):
        docs = _make_rag_docs(3)
        agent = NaiveRAGAgent(
            embedder=_FakeEmbedder(),
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=2,
        )
        result = agent.run("my question", docs)
        assert "my question" in result["trajectory"]["prompt_preview"]


# ---------------------------------------------------------------------------
# NaiveRAGAgent + DocEmbeddingCache integration tests
# ---------------------------------------------------------------------------


class TestNaiveRAGAgentWithCache:
    """With doc_emb_cache enabled, a second run on the same docs skips embed_batch."""

    def test_second_run_skips_embed_batch(self):
        """After caching embeddings on the first run, a second run with the
        same docs should not call embed_batch at all."""
        docs = _make_rag_docs(5)
        embedder = _FakeEmbedder()
        cache = DocEmbeddingCache()  # in-memory only

        agent = NaiveRAGAgent(
            embedder=embedder,
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=3,
            doc_emb_cache=cache,
        )

        # First run — embeds all 5 docs (1 embed_batch call)
        agent.run("first question", docs)
        assert embedder.embed_batch_call_count == 1

        # Second run with identical docs — should use cache, no new embed_batch
        agent.run("second question", docs)
        assert (
            embedder.embed_batch_call_count == 1
        ), "embed_batch should not be called again when all docids are cached"

    def test_partial_cache_embeds_only_missing(self):
        """If some docs are already cached and some are new, only the new
        subset is sent to embed_batch."""
        docs_first = _make_rag_docs(3)  # doc_0, doc_1, doc_2
        docs_second = _make_rag_docs(5)  # doc_0..doc_4 (2 new: doc_3, doc_4)

        embedder = _FakeEmbedder()
        cache = DocEmbeddingCache()

        agent = NaiveRAGAgent(
            embedder=embedder,
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=2,
            doc_emb_cache=cache,
        )

        agent.run("q1", docs_first)
        assert embedder.embed_batch_call_count == 1
        assert cache.size() == 3

        # Second run has 5 docs; 3 are cached, 2 are new
        agent.run("q2", docs_second)
        assert (
            embedder.embed_batch_call_count == 2
        ), "embed_batch should be called once more for the 2 new docs"
        assert cache.size() == 5

    def test_result_schema_unchanged_with_cache(self):
        """Enabling the cache must not change the returned result schema."""
        docs = _make_rag_docs(5)
        cache = DocEmbeddingCache()
        agent = NaiveRAGAgent(
            embedder=_FakeEmbedder(),
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=3,
            doc_emb_cache=cache,
        )
        result = agent.run("test question", docs)
        required = {
            "answer",
            "trajectory",
            "judgment",
            "tokens",
            "wall_time",
            "n_iters",
            "cache_hit",
        }
        assert required <= set(result.keys())

    def test_cache_none_falls_back_to_normal(self):
        """doc_emb_cache=None must use the original full embed_batch path."""
        docs = _make_rag_docs(5)
        embedder = _FakeEmbedder()
        agent = NaiveRAGAgent(
            embedder=embedder,
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=3,
            doc_emb_cache=None,
        )
        agent.run("q1", docs)
        agent.run("q2", docs)
        # Both runs must call embed_batch (no caching)
        assert embedder.embed_batch_call_count == 2

    def test_embedding_reuse_produces_same_scores(self):
        """Cached embeddings must yield the same retrieval result as fresh ones."""
        docs = _make_rag_docs(5)
        embedder1 = _FakeEmbedder()
        embedder2 = _FakeEmbedder()

        # Run without cache
        agent_no_cache = NaiveRAGAgent(
            embedder=embedder1,
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=3,
        )
        result_no_cache = agent_no_cache.run("same question", docs)

        # Run with cache (populate on first run, reuse on second)
        cache = DocEmbeddingCache()
        agent_with_cache = NaiveRAGAgent(
            embedder=embedder2,
            judge=_always_correct,
            lm=_FakeLM(),
            top_k=3,
            doc_emb_cache=cache,
        )
        agent_with_cache.run("same question", docs)  # populates cache
        result_cached = agent_with_cache.run("same question", docs)  # uses cache

        assert (
            result_no_cache["trajectory"]["retrieved_docids"]
            == result_cached["trajectory"]["retrieved_docids"]
        )


# ---------------------------------------------------------------------------
# LongContextStuffAgent tests
# ---------------------------------------------------------------------------


def _make_stuff_docs(n: int, chars_per_doc: int = 50) -> list[dict]:
    """Return *n* synthetic doc dicts with text of approximately *chars_per_doc* chars."""
    return [{"docid": f"sdoc_{i}", "text": f"Doc{i}: " + ("x" * chars_per_doc)} for i in range(n)]


class TestLongContextStuffAgent:
    """Unit tests for LongContextStuffAgent."""

    # ------------------------------------------------------------------
    # Test 1: LM called exactly once
    # ------------------------------------------------------------------

    def test_lm_called_exactly_once(self):
        docs = _make_stuff_docs(5)
        fake_lm = _FakeLM(answer="the_answer")
        agent = LongContextStuffAgent(judge=_always_correct, lm=fake_lm)
        agent.run("What is X?", docs)
        assert fake_lm.call_count == 1

    # ------------------------------------------------------------------
    # Test 2: Schema completeness + n_iters=1, cache_hit=False
    # ------------------------------------------------------------------

    def test_schema_keys_present_and_fixed_fields(self):
        docs = _make_stuff_docs(3)
        fake_lm = _FakeLM()
        agent = LongContextStuffAgent(judge=_always_correct, lm=fake_lm)
        result = agent.run("question?", docs)

        required = {
            "answer",
            "trajectory",
            "judgment",
            "tokens",
            "wall_time",
            "n_iters",
            "cache_hit",
        }
        assert required <= set(result.keys()), f"Missing: {required - set(result.keys())}"
        assert result["n_iters"] == 1
        assert result["cache_hit"] is False

    # ------------------------------------------------------------------
    # Test 3: Truncation when docs exceed budget
    # ------------------------------------------------------------------

    def test_truncation_when_over_budget(self):
        # Each doc has ~8000 chars; with max_input_tokens=500 the first doc
        # alone will exceed the budget, so truncated=True and stuffed list
        # is a strict prefix (possibly empty) of the input.
        docs = _make_stuff_docs(5, chars_per_doc=8000)
        fake_lm = _FakeLM()
        agent = LongContextStuffAgent(
            judge=_always_correct,
            lm=fake_lm,
            max_input_tokens=500,  # tiny budget → forces truncation
        )
        result = agent.run("question?", docs)

        traj = result["trajectory"]
        assert traj["truncated"] is True
        # stuffed_docids must be a prefix of the input doc list
        all_docids = [d["docid"] for d in docs]
        assert traj["stuffed_docids"] == all_docids[: len(traj["stuffed_docids"])]

    # ------------------------------------------------------------------
    # Test 4: No truncation when docs fit in budget
    # ------------------------------------------------------------------

    def test_no_truncation_when_within_budget(self):
        # 3 tiny docs easily fit in a 900 000-token budget.
        docs = _make_stuff_docs(3, chars_per_doc=10)
        fake_lm = _FakeLM()
        agent = LongContextStuffAgent(
            judge=_always_correct,
            lm=fake_lm,
            max_input_tokens=900_000,
        )
        result = agent.run("question?", docs)

        traj = result["trajectory"]
        assert traj["truncated"] is False
        assert traj["stuffed_docids"] == [d["docid"] for d in docs]

    # ------------------------------------------------------------------
    # Test 5: stuffed_docids preserves input order
    # ------------------------------------------------------------------

    def test_stuffed_docids_preserve_order(self):
        docs = [
            {"docid": "alpha", "text": "short"},
            {"docid": "beta", "text": "short"},
            {"docid": "gamma", "text": "short"},
        ]
        fake_lm = _FakeLM()
        agent = LongContextStuffAgent(judge=_always_correct, lm=fake_lm, max_input_tokens=900_000)
        result = agent.run("q", docs)
        assert result["trajectory"]["stuffed_docids"] == ["alpha", "beta", "gamma"]

    # ------------------------------------------------------------------
    # Test 6: Judge called once with (question, answer)
    # ------------------------------------------------------------------

    def test_judge_called_once_with_correct_args(self):
        calls: list[tuple[str, str]] = []

        def recording_judge(question: str, answer: str) -> bool:
            calls.append((question, answer))
            return True

        docs = _make_stuff_docs(2)
        fake_lm = _FakeLM(answer="my_answer")
        agent = LongContextStuffAgent(judge=recording_judge, lm=fake_lm)
        agent.run("my question", docs)

        assert len(calls) == 1
        assert calls[0] == ("my question", "my_answer")


# ---------------------------------------------------------------------------
# LongContextStuffAgent doc_order tests
# ---------------------------------------------------------------------------


def _make_ordered_docs(n: int) -> list[dict]:
    """Return *n* docs with docids 'doc_0' .. 'doc_{n-1}' and short text."""
    return [{"docid": f"doc_{i}", "text": f"text {i}"} for i in range(n)]


class TestLongContextStuffDocOrder:
    """Verify that doc_order correctly reorders documents before stuffing."""

    def _stuffed_ids(self, docs: list[dict], doc_order: str) -> list[str]:
        """Run LongContextStuffAgent and return stuffed_docids from trajectory."""
        fake_lm = _FakeLM()
        agent = LongContextStuffAgent(
            judge=_always_correct,
            lm=fake_lm,
            max_input_tokens=900_000,
            doc_order=doc_order,
        )
        result = agent.run("question?", docs)
        return result["trajectory"]["stuffed_docids"]

    def test_primacy_preserves_input_order(self):
        """doc_order='primacy' must not change doc order."""
        docs = _make_ordered_docs(6)
        expected = [d["docid"] for d in docs]
        assert self._stuffed_ids(docs, "primacy") == expected

    def test_recency_reverses_order(self):
        """doc_order='recency' must reverse the doc list."""
        docs = _make_ordered_docs(6)
        expected = [d["docid"] for d in reversed(docs)]
        assert self._stuffed_ids(docs, "recency") == expected

    def test_middle_places_first_10_in_centre(self):
        """doc_order='middle' must place docs[:10] in the middle of the list.

        Negatives (docs[10:]) are split in half; gold block goes between them.
        With 30 docs: negatives = docs[10:] (20 docs), neg_half = 10.
        Expected: docs[10:20] + docs[:10] + docs[20:30].
        """
        docs = _make_ordered_docs(30)
        result_ids = self._stuffed_ids(docs, "middle")
        all_ids = [d["docid"] for d in docs]
        # negatives = all_ids[10:], neg_half = 10
        # expected: all_ids[10:20] + all_ids[:10] + all_ids[20:]
        expected = all_ids[10:20] + all_ids[:10] + all_ids[20:]
        assert result_ids == expected

    def test_random_is_permutation_of_input(self):
        """doc_order='random' must be a permutation of the input docs."""
        docs = _make_ordered_docs(10)
        result_ids = self._stuffed_ids(docs, "random")
        assert sorted(result_ids) == sorted(d["docid"] for d in docs)

    def test_random_is_deterministic_with_same_seed(self):
        """Two agents with the same seed and doc_order='random' must produce
        identical orderings."""
        docs = _make_ordered_docs(10)
        ids_a = self._stuffed_ids(docs, "random")
        ids_b = self._stuffed_ids(docs, "random")
        assert ids_a == ids_b

    def test_doc_order_recorded_in_trajectory(self):
        """trajectory['doc_order'] must equal the configured doc_order string."""
        docs = _make_ordered_docs(5)
        for order in ("primacy", "recency", "middle", "random"):
            fake_lm = _FakeLM()
            agent = LongContextStuffAgent(
                judge=_always_correct,
                lm=fake_lm,
                max_input_tokens=900_000,
                doc_order=order,
            )
            result = agent.run("q", docs)
            assert result["trajectory"]["doc_order"] == order, f"failed for doc_order={order!r}"

    def test_middle_and_primacy_differ_for_large_doc_list(self):
        """For >=11 docs, middle order must differ from primacy order."""
        docs = _make_ordered_docs(20)
        primacy_ids = self._stuffed_ids(docs, "primacy")
        middle_ids = self._stuffed_ids(docs, "middle")
        assert primacy_ids != middle_ids
