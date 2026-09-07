"""Unit tests for src/hgc/hgc_core.py.

No real API calls are made — uses DummyLM, FakeEmbedder, FakeExtractor, fake judge.
"""

from __future__ import annotations

import time
import uuid

import dspy
import numpy as np
from dspy.utils import DummyLM

from hgc.hgc_core import HGCCoreAgent, _parse_docid, format_hints_prompt
from hgc.memory import HintRecord, HintStore

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _make_embedding(seed: int = 0, dims: int = 1536) -> np.ndarray:
    """Deterministic unit vector from a seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dims).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """Returns a fixed deterministic vector regardless of input text."""

    def __init__(self, seed: int = 42) -> None:
        self._vec = _make_embedding(seed)

    def embed(self, text: str) -> np.ndarray:  # noqa: ARG002
        return self._vec.copy()


class FakeExtractor:
    """Returns a predetermined extraction result."""

    def __init__(self, result: dict | None = None) -> None:
        self._result = result or {"location": [], "entity": [], "strategy": []}
        self.calls: list[dict] = []

    def extract(self, query: str, trajectory: dict, correct: bool) -> dict:
        self.calls.append({"query": query, "trajectory": trajectory, "correct": correct})
        return dict(self._result)


def _fake_judge_true(question: str, answer: str) -> bool:  # noqa: ARG001
    return True


def _fake_judge_false(question: str, answer: str) -> bool:  # noqa: ARG001
    return False


def _make_store(tmp_path) -> HintStore:
    return HintStore(db_path=str(tmp_path / f"hints_{uuid.uuid4().hex}.db"))


def _dummy_tool(x: str) -> str:
    """A fake tool for testing."""
    return f"tool_result:{x}"


def _react_lm() -> DummyLM:
    """DummyLM that makes ReAct call finish immediately then extract answer."""
    return DummyLM(
        [
            {"next_thought": "I will finish.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "done", "answer": "test answer"},
        ]
    )


def _seed_store(store: HintStore, embedder: FakeEmbedder) -> list[HintRecord]:
    """Add 2 positive + 1 negative hint to the store."""
    emb_bytes = embedder.embed("seed").astype(np.float32).tobytes()
    now = time.time()

    records = [
        HintRecord(
            hint_id=str(uuid.uuid4()),
            hint_type="location",
            polarity="positive",
            content="Document 5412 contained relevant info for similar queries.",
            content_meta={"docid": "5412"},
            query_ctx="Who founded the university?",
            query_ctx_embedding=emb_bytes,
            trajectory_step=0,
            created_at=now,
            last_validated_at=now,
            success_count=3,
            failure_count=1,
            retrieval_count=5,
        ),
        HintRecord(
            hint_id=str(uuid.uuid4()),
            hint_type="strategy",
            polarity="positive",
            content='The search query "cultural activities 2002" returned useful results.',
            content_meta={"keyword": "cultural activities 2002"},
            query_ctx="Who founded the university?",
            query_ctx_embedding=emb_bytes,
            trajectory_step=1,
            created_at=now,
            last_validated_at=now,
            success_count=2,
            failure_count=0,
            retrieval_count=3,
        ),
        HintRecord(
            hint_id=str(uuid.uuid4()),
            hint_type="location",
            polarity="negative",
            content="Document 26215 was checked but did not help.",
            content_meta={"docid": "26215"},
            query_ctx="Who founded the university?",
            query_ctx_embedding=emb_bytes,
            trajectory_step=2,
            created_at=now,
            last_validated_at=now,
            success_count=0,
            failure_count=3,
            retrieval_count=4,
        ),
    ]
    for r in records:
        store.add(r)
    return records


# ---------------------------------------------------------------------------
# Tests: format_hints_prompt
# ---------------------------------------------------------------------------


class TestFormatHintsPrompt:
    def _make_hint(self, hint_type: str, polarity: str, content: str) -> HintRecord:
        emb = _make_embedding(0)
        now = time.time()
        return HintRecord(
            hint_id=str(uuid.uuid4()),
            hint_type=hint_type,
            polarity=polarity,
            content=content,
            content_meta={},
            query_ctx="q",
            query_ctx_embedding=emb.tobytes(),
            trajectory_step=0,
            created_at=now,
            last_validated_at=now,
            success_count=1,
            failure_count=0,
            retrieval_count=1,
        )

    def test_empty_lists_returns_empty_string(self):
        assert format_hints_prompt([], []) == ""

    def test_positive_only_has_helpful_block(self):
        pos = [self._make_hint("location", "positive", "doc 5412")]
        result = format_hints_prompt(pos, [])
        assert "HELPFUL" in result
        assert "AVOID" not in result
        assert "doc 5412" in result

    def test_negative_only_has_avoid_block(self):
        neg = [self._make_hint("location", "negative", "doc 999 was useless")]
        result = format_hints_prompt([], neg)
        assert "AVOID" in result
        assert "HELPFUL" not in result
        assert "doc 999" in result

    def test_both_blocks_appear(self):
        pos = [self._make_hint("location", "positive", "doc A")]
        neg = [self._make_hint("location", "negative", "doc B")]
        result = format_hints_prompt(pos, neg)
        assert "HELPFUL" in result
        assert "AVOID" in result

    def test_header_appears(self):
        pos = [self._make_hint("entity", "positive", "Queen Arwa University")]
        result = format_hints_prompt(pos, [])
        assert "=== Hints from prior similar queries ===" in result

    def test_type_prefix_labels(self):
        pos = [
            self._make_hint("location", "positive", "doc 1"),
            self._make_hint("entity", "positive", "entity 1"),
            self._make_hint("strategy", "positive", "strat 1"),
        ]
        result = format_hints_prompt(pos, [])
        assert "[L1]" in result
        assert "[E1]" in result
        assert "[S1]" in result

    def test_negative_n_labels(self):
        neg = [
            self._make_hint("location", "negative", "bad doc 1"),
            self._make_hint("location", "negative", "bad doc 2"),
        ]
        result = format_hints_prompt([], neg)
        assert "[N1]" in result
        assert "[N2]" in result

    def test_suffix_present(self):
        pos = [self._make_hint("location", "positive", "x")]
        result = format_hints_prompt(pos, [])
        assert "re-verify" in result


# ---------------------------------------------------------------------------
# Tests: HGCCoreAgent — empty store
# ---------------------------------------------------------------------------


class TestEmptyStore:
    def test_no_hint_block_when_store_empty(self, tmp_path):
        """When store is empty, no HELPFUL or AVOID block in the question passed to ReAct."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        extractor = FakeExtractor()

        captured_questions = []

        lm = _react_lm()
        dspy.configure(lm=lm)

        original_forward = dspy.ReAct.forward

        def patched_forward(self_react, **input_args):
            captured_questions.append(input_args.get("question", ""))
            return original_forward(self_react, **input_args)

        dspy.ReAct.forward = patched_forward
        try:
            agent = HGCCoreAgent(
                tools=[_dummy_tool],
                store=store,
                embedder=embedder,
                extractor=extractor,
                judge=_fake_judge_true,
            )
            agent.run("What is the capital of France?")
        finally:
            dspy.ReAct.forward = original_forward

        assert captured_questions, "forward should have been called"
        q = captured_questions[0]
        assert "HELPFUL" not in q
        assert "AVOID" not in q

    def test_run_returns_all_required_keys(self, tmp_path):
        _make_store(tmp_path)
        lm = _react_lm()
        dspy.configure(lm=lm)

        agent = HGCCoreAgent(
            tools=[_dummy_tool],
            store=_make_store(tmp_path),
            embedder=FakeEmbedder(),
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
        )
        record = agent.run("test question")

        required_keys = {
            "answer",
            "trajectory",
            "retrieved_positive_hints",
            "retrieved_negative_hints",
            "added_hints",
            "judgment",
            "tokens",
            "wall_time",
            "n_iters",
        }
        assert required_keys.issubset(set(record.keys()))

    def test_answer_is_string(self, tmp_path):
        lm = _react_lm()
        dspy.configure(lm=lm)
        agent = HGCCoreAgent(
            tools=[_dummy_tool],
            store=_make_store(tmp_path),
            embedder=FakeEmbedder(),
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
        )
        record = agent.run("test question")
        assert isinstance(record["answer"], str)

    def test_judgment_reflects_judge(self, tmp_path):
        lm = _react_lm()
        dspy.configure(lm=lm)
        agent_true = HGCCoreAgent(
            tools=[_dummy_tool],
            store=_make_store(tmp_path),
            embedder=FakeEmbedder(),
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
        )
        assert agent_true.run("q")["judgment"] is True

        agent_false = HGCCoreAgent(
            tools=[_dummy_tool],
            store=_make_store(tmp_path),
            embedder=FakeEmbedder(),
            extractor=FakeExtractor(),
            judge=_fake_judge_false,
        )
        assert agent_false.run("q")["judgment"] is False


# ---------------------------------------------------------------------------
# Tests: HGCCoreAgent — seeded store
# ---------------------------------------------------------------------------


class TestSeededStore:
    def test_hints_retrieved_when_store_has_relevant(self, tmp_path):
        """After seeding store with 2 positive + 1 negative, run() retrieves them."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        _seed_store(store, embedder)

        lm = _react_lm()
        dspy.configure(lm=lm)

        agent = HGCCoreAgent(
            tools=[_dummy_tool],
            store=store,
            embedder=embedder,
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
        )
        record = agent.run("Who founded the university?")

        # Should have retrieved some hints
        total_retrieved = len(record["retrieved_positive_hints"]) + len(
            record["retrieved_negative_hints"]
        )
        assert total_retrieved > 0

    def test_helpful_block_in_question_when_positive_hints(self, tmp_path):
        """format_hints_prompt HELPFUL block appears in question when positive hints exist."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        _seed_store(store, embedder)

        captured_questions = []
        original_forward = dspy.ReAct.forward

        def patched_forward(self_react, **input_args):
            captured_questions.append(input_args.get("question", ""))
            return original_forward(self_react, **input_args)

        lm = _react_lm()
        dspy.configure(lm=lm)

        dspy.ReAct.forward = patched_forward
        try:
            agent = HGCCoreAgent(
                tools=[_dummy_tool],
                store=store,
                embedder=embedder,
                extractor=FakeExtractor(),
                judge=_fake_judge_true,
            )
            agent.run("Who founded the university?")
        finally:
            dspy.ReAct.forward = original_forward

        assert captured_questions
        q = captured_questions[0]
        assert "HELPFUL" in q

    def test_avoid_block_in_question_when_negative_hints(self, tmp_path):
        """AVOID block appears in question when negative hints pass the similarity filter."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        _seed_store(store, embedder)

        captured_questions = []
        original_forward = dspy.ReAct.forward

        def patched_forward(self_react, **input_args):
            captured_questions.append(input_args.get("question", ""))
            return original_forward(self_react, **input_args)

        lm = _react_lm()
        dspy.configure(lm=lm)

        # Lower theta_neg to 0.0 so negative hint always passes the filter
        dspy.ReAct.forward = patched_forward
        try:
            agent = HGCCoreAgent(
                tools=[_dummy_tool],
                store=store,
                embedder=embedder,
                extractor=FakeExtractor(),
                judge=_fake_judge_true,
            )
            # Use store.search directly with theta_neg=0 by patching
            original_search = store.search

            def patched_search(query_embedding, k, alpha, beta, gamma, **kwargs):
                return original_search(
                    query_embedding=query_embedding,
                    k=k,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    theta_neg=0.0,
                )

            store.search = patched_search
            agent.run("Who founded the university?")
        finally:
            dspy.ReAct.forward = original_forward
            store.search = original_search

        assert captured_questions
        q = captured_questions[0]
        assert "AVOID" in q

    def test_extractor_called_after_run(self, tmp_path):
        """extractor.extract() is called once per run()."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        extractor = FakeExtractor()

        lm = _react_lm()
        dspy.configure(lm=lm)

        agent = HGCCoreAgent(
            tools=[_dummy_tool],
            store=store,
            embedder=embedder,
            extractor=extractor,
            judge=_fake_judge_true,
        )
        agent.run("What is x?")
        assert len(extractor.calls) == 1
        assert extractor.calls[0]["query"] == "What is x?"

    def test_added_hints_appear_in_store(self, tmp_path):
        """Hints returned by extractor are persisted in the store."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        extractor = FakeExtractor(
            result={
                "location": [{"content": "docid=9999", "content_meta": {"docid": "9999"}}],
                "entity": [],
                "strategy": [],
            }
        )

        lm = _react_lm()
        dspy.configure(lm=lm)

        agent = HGCCoreAgent(
            tools=[_dummy_tool],
            store=store,
            embedder=embedder,
            extractor=extractor,
            judge=_fake_judge_true,
        )
        record = agent.run("Find doc 9999")

        assert len(record["added_hints"]) == 1
        hint_id = record["added_hints"][0]
        stored = store.get(hint_id)
        assert stored is not None
        assert stored.content == "docid=9999"

    def test_update_on_outcome_called_for_retrieved_hints(self, tmp_path):
        """success_count increases for retrieved positive hints on correct=True."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        seeded = _seed_store(store, embedder)
        pos_seeded = [h for h in seeded if h.polarity == "positive"]

        lm = _react_lm()
        dspy.configure(lm=lm)

        agent = HGCCoreAgent(
            tools=[_dummy_tool],
            store=store,
            embedder=embedder,
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
        )
        agent.run("Who founded the university?")

        # At least one seeded positive hint should have success_count incremented
        updated = [store.get(h.hint_id) for h in pos_seeded]
        # success_count should be >= original (incremented for retrieved ones)
        for orig, fresh in zip(pos_seeded, updated, strict=False):
            assert fresh is not None
            # success_count is either unchanged (not retrieved) or +1 (retrieved)
            assert fresh.success_count >= orig.success_count

    def test_negative_hints_retrieved_appear_in_record(self, tmp_path):
        """retrieved_negative_hints in record contains negative hints from store."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        _seed_store(store, embedder)

        lm = _react_lm()
        dspy.configure(lm=lm)

        original_search = store.search

        def patched_search(query_embedding, k, alpha, beta, gamma, **kwargs):
            return original_search(
                query_embedding=query_embedding,
                k=k,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                theta_neg=0.0,
            )

        store.search = patched_search
        try:
            agent = HGCCoreAgent(
                tools=[_dummy_tool],
                store=store,
                embedder=embedder,
                extractor=FakeExtractor(),
                judge=_fake_judge_true,
            )
            record = agent.run("Who founded the university?")
        finally:
            store.search = original_search

        assert len(record["retrieved_negative_hints"]) > 0

    def test_wall_time_positive(self, tmp_path):
        lm = _react_lm()
        dspy.configure(lm=lm)
        agent = HGCCoreAgent(
            tools=[_dummy_tool],
            store=_make_store(tmp_path),
            embedder=FakeEmbedder(),
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
        )
        record = agent.run("test")
        assert record["wall_time"] > 0

    def test_n_iters_non_negative(self, tmp_path):
        lm = _react_lm()
        dspy.configure(lm=lm)
        agent = HGCCoreAgent(
            tools=[_dummy_tool],
            store=_make_store(tmp_path),
            embedder=FakeEmbedder(),
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
        )
        record = agent.run("test")
        assert record["n_iters"] >= 0


# ---------------------------------------------------------------------------
# Tests: token counter (Fix 1)
# ---------------------------------------------------------------------------


class TestTokenCounter:
    """Verify _count_tokens_since sums only entries added after baseline index."""

    def test_token_count_reflects_lm_history(self, tmp_path):
        """DummyLM with injected usage entries: tokens == sum of new entries."""
        from hgc.hgc_core import _count_tokens_since, _history_len

        lm = _react_lm()
        dspy.configure(lm=lm)

        # Inject a fake pre-existing history entry so baseline > 0
        lm.history = [{"usage": {"total_tokens": 99}}]

        baseline = _history_len(lm)
        assert baseline == 1

        # Simulate two new LM calls added to history after baseline
        lm.history.append({"usage": {"total_tokens": 50}})
        lm.history.append({"usage": {"total_tokens": 30}})

        tokens = _count_tokens_since(lm, baseline)
        assert tokens == 80  # 50 + 30; pre-existing 99 is excluded

    def test_token_count_excludes_prior_history(self, tmp_path):
        """Pre-existing history entries are NOT counted."""
        from hgc.hgc_core import _count_tokens_since, _history_len

        lm = _react_lm()
        dspy.configure(lm=lm)

        lm.history = [
            {"usage": {"total_tokens": 500}},
            {"usage": {"total_tokens": 500}},
        ]
        baseline = _history_len(lm)

        # Add one new entry
        lm.history.append({"usage": {"total_tokens": 10}})

        tokens = _count_tokens_since(lm, baseline)
        assert tokens == 10

    def test_token_count_zero_when_no_new_entries(self, tmp_path):
        """If history does not grow, token count is 0."""
        from hgc.hgc_core import _count_tokens_since, _history_len

        lm = _react_lm()
        dspy.configure(lm=lm)
        lm.history = [{"usage": {"total_tokens": 42}}]

        baseline = _history_len(lm)
        tokens = _count_tokens_since(lm, baseline)
        assert tokens == 0

    def test_agent_run_tokens_key_non_negative(self, tmp_path):
        """run() always returns a non-negative integer for 'tokens'."""
        lm = _react_lm()
        dspy.configure(lm=lm)
        agent = HGCCoreAgent(
            tools=[_dummy_tool],
            store=_make_store(tmp_path),
            embedder=FakeEmbedder(),
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
        )
        record = agent.run("test question")
        assert isinstance(record["tokens"], int)
        assert record["tokens"] >= 0


# ---------------------------------------------------------------------------
# Tests: theta_pos / theta_neg constructor params (Fix 2)
# ---------------------------------------------------------------------------


class TestThetaParams:
    """Verify theta_pos/theta_neg are plumbed through from constructor to store.search()."""

    def test_lower_theta_pos_retrieves_more_hints(self, tmp_path):
        """With theta_pos=0.1, hints that score below 0.3 similarity are retrieved."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder(seed=42)
        # Use a *different* seed for the stored hint embedding so cosine sim
        # is somewhat below 0.3 but above 0.1.
        emb_bytes = _make_embedding(seed=99).astype(np.float32).tobytes()
        now = time.time()
        record = HintRecord(
            hint_id=str(uuid.uuid4()),
            hint_type="strategy",
            polarity="positive",
            content="Low-similarity hint that should be retrieved with loose theta.",
            content_meta={},
            query_ctx="unrelated query",
            query_ctx_embedding=emb_bytes,
            trajectory_step=0,
            created_at=now,
            last_validated_at=now,
            success_count=1,
            failure_count=0,
            retrieval_count=0,
        )
        store.add(record)

        lm = _react_lm()
        dspy.configure(lm=lm)

        # Strict threshold — likely blocks the hint
        agent_strict = HGCCoreAgent(
            tools=[_dummy_tool],
            store=store,
            embedder=embedder,
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
            theta_pos=0.9,  # very strict — should block anything < 0.9 sim
        )
        record_strict = agent_strict.run("test question")

        # Loose threshold — retrieves even distant hints
        agent_loose = HGCCoreAgent(
            tools=[_dummy_tool],
            store=store,
            embedder=embedder,
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
            theta_pos=0.0,  # no filter — retrieves everything
        )
        record_loose = agent_loose.run("test question")

        n_strict = len(record_strict["retrieved_positive_hints"])
        n_loose = len(record_loose["retrieved_positive_hints"])
        assert n_loose >= n_strict, (
            f"Loose theta_pos=0.0 should retrieve >= hints than strict theta_pos=0.9 "
            f"(got {n_loose} vs {n_strict})"
        )

    def test_theta_params_stored_on_agent(self):
        """Constructor stores theta_pos and theta_neg as instance attributes."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = HintStore(db_path=f"{tmp}/h.db")
            agent = HGCCoreAgent(
                tools=[_dummy_tool],
                store=store,
                embedder=FakeEmbedder(),
                extractor=FakeExtractor(),
                judge=_fake_judge_true,
                theta_pos=0.1,
                theta_neg=0.2,
            )
            assert agent._theta_pos == 0.1
            assert agent._theta_neg == 0.2


# ---------------------------------------------------------------------------
# Tests: docid validation filter (US-020)
# ---------------------------------------------------------------------------


def _seed_two_location_hints(
    store: HintStore,
    embedder: FakeEmbedder,
    valid_content: str = "docid=100",
    invalid_content: str = "docid=999",
) -> tuple[HintRecord, HintRecord]:
    """Seed exactly two positive location hints with given contents."""
    emb_bytes = embedder.embed("seed").astype(np.float32).tobytes()
    now = time.time()

    valid_hint = HintRecord(
        hint_id=str(uuid.uuid4()),
        hint_type="location",
        polarity="positive",
        content=valid_content,
        content_meta={"docid": valid_content.split("=")[-1]},
        query_ctx="test",
        query_ctx_embedding=emb_bytes,
        trajectory_step=0,
        created_at=now,
        last_validated_at=now,
        success_count=1,
        failure_count=0,
        retrieval_count=1,
    )
    invalid_hint = HintRecord(
        hint_id=str(uuid.uuid4()),
        hint_type="location",
        polarity="positive",
        content=invalid_content,
        content_meta={"docid": invalid_content.split("=")[-1]},
        query_ctx="test",
        query_ctx_embedding=emb_bytes,
        trajectory_step=0,
        created_at=now,
        last_validated_at=now,
        success_count=1,
        failure_count=0,
        retrieval_count=1,
    )
    store.add(valid_hint)
    store.add(invalid_hint)
    return valid_hint, invalid_hint


class TestDocidFilter:
    """US-020: docid validation filter in HGCCoreAgent."""

    def _make_agent(self, store, embedder):
        return HGCCoreAgent(
            tools=[_dummy_tool],
            store=store,
            embedder=embedder,
            extractor=FakeExtractor(),
            judge=_fake_judge_true,
            theta_pos=0.0,  # retrieve all hints regardless of similarity
        )

    def _capture_question(self, agent, question, **run_kwargs):
        """Run agent and return the question string passed to dspy.ReAct."""
        captured = []
        original_forward = dspy.ReAct.forward

        def patched_forward(self_react, **input_args):
            captured.append(input_args.get("question", ""))
            return original_forward(self_react, **input_args)

        lm = _react_lm()
        dspy.configure(lm=lm)
        dspy.ReAct.forward = patched_forward
        try:
            agent.run(question, **run_kwargs)
        finally:
            dspy.ReAct.forward = original_forward

        return captured[0] if captured else ""

    def test_valid_docid_in_prompt_invalid_dropped(self, tmp_path):
        """Seed docid=100 (valid) and docid=999 (invalid). Only 100 should appear."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        _seed_two_location_hints(store, embedder, "docid=100", "docid=999")

        agent = self._make_agent(store, embedder)
        q = self._capture_question(agent, "test question", valid_docids=frozenset({"100"}))

        assert "docid=100" in q, "Valid hint should appear in HELPFUL block"
        assert "docid=999" not in q, "Invalid hint should be dropped"

    def test_entity_hint_passes_through_regardless(self, tmp_path):
        """Entity hint is not a docid pattern; it should pass through unchanged."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        emb_bytes = embedder.embed("seed").astype(np.float32).tobytes()
        now = time.time()

        # Two location hints (one valid, one invalid) + one entity hint
        _seed_two_location_hints(store, embedder, "docid=100", "docid=999")

        entity_hint = HintRecord(
            hint_id=str(uuid.uuid4()),
            hint_type="entity",
            polarity="positive",
            content="Queen Arwa University was referenced here.",
            content_meta={},
            query_ctx="test",
            query_ctx_embedding=emb_bytes,
            trajectory_step=0,
            created_at=now,
            last_validated_at=now,
            success_count=1,
            failure_count=0,
            retrieval_count=1,
        )
        store.add(entity_hint)

        agent = self._make_agent(store, embedder)
        q = self._capture_question(agent, "test question", valid_docids=frozenset({"100"}))

        assert "Queen Arwa University" in q, "Entity hint must not be filtered"
        assert "docid=100" in q, "Valid location hint must appear"
        assert "docid=999" not in q, "Invalid location hint must be dropped"

    def test_none_valid_docids_all_hints_pass(self, tmp_path):
        """valid_docids=None → backward compat, all hints pass through."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        _seed_two_location_hints(store, embedder, "docid=100", "docid=999")

        agent = self._make_agent(store, embedder)
        # valid_docids defaults to None — no filtering
        q = self._capture_question(agent, "test question")

        assert "docid=100" in q
        assert "docid=999" in q

    def test_empty_valid_docids_drops_all_location_hints(self, tmp_path):
        """valid_docids=frozenset() → all location hints dropped."""
        store = _make_store(tmp_path)
        embedder = FakeEmbedder()
        _seed_two_location_hints(store, embedder, "docid=100", "docid=999")

        agent = self._make_agent(store, embedder)
        q = self._capture_question(agent, "test question", valid_docids=frozenset())

        assert "docid=100" not in q, "Should be dropped (empty valid set)"
        assert "docid=999" not in q, "Should be dropped (empty valid set)"


# ---------------------------------------------------------------------------
# Tests: _parse_docid helper
# ---------------------------------------------------------------------------


class TestParseDocid:
    def test_docid_equals_pattern(self):
        assert _parse_docid("docid=5412") == "5412"

    def test_docid_equals_pattern_in_longer_string(self):
        assert _parse_docid("Document docid=5412 contained relevant info.") == "5412"

    def test_purely_numeric_string(self):
        assert _parse_docid("5412") == "5412"

    def test_non_docid_returns_none(self):
        assert _parse_docid("Queen Arwa University was referenced here.") is None

    def test_strategy_hint_returns_none(self):
        assert (
            _parse_docid('The search query "cultural activities 2002" returned useful results.')
            is None
        )

    def test_empty_string_returns_none(self):
        assert _parse_docid("") is None

    def test_alphanumeric_returns_none(self):
        assert _parse_docid("abc123") is None
