"""Baseline agents for HGC comparison experiments.

Implements two GPTCache / RAP-lite style baselines that share the same
``run(question) -> dict`` interface as HGCCoreAgent so experiment harness
can swap them in place.

Baseline 1 — AnswerCacheAgent (GPTCache-style)
    Stores (query_embedding, answer_text).  On a new query, if the best
    cosine similarity against stored embeddings meets sim_threshold,
    returns the cached answer immediately (no ReAct loop, zero tokens).

Baseline 2 — TrajectoryCacheAgent (RAP-lite)
    Stores (query_embedding, question, trajectory_summary).  On a new query,
    retrieves top-K summaries above sim_threshold and injects them as
    in-context demonstrations in the agent prompt prefix before running
    the ReAct loop.

Baseline 3 — Mem0ReActAgent (DSPy/Mem0 tutorial-style)
    Agent-discretionary memory via three tools:
      store_memory(content)          — agent-initiated write
      search_memories(query, limit)  — semantic lookup
      get_all_memories()             — dump all memories
    Memory is NOT auto-injected; the agent must pull it via tool calls.

All agents use the same ``run()`` return schema:
    {
        "answer":     str,
        "trajectory": dict,          # dspy trajectory dict; empty on cache hit
        "judgment":   bool,          # judge(question, answer)
        "tokens":     int,           # 0 on cache hit
        "wall_time":  float,         # seconds
        "n_iters":    int,           # 0 on cache hit
        "cache_hit":  bool,
    }
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import tiktoken

from hgc.emb_cache import DocEmbeddingCache
from hgc.memory import _cosine_sim

# Optional mem0ai dependency — imported lazily so the module loads without it.
# Tests can monkeypatch this name to inject a fake Memory class.
try:
    from mem0 import Memory
except ImportError:
    Memory = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_similarity(
    query_emb: np.ndarray,
    store: list[tuple[np.ndarray, Any]],
) -> tuple[float, int]:
    """Return (best_cosine_sim, index) over *store*.

    *store* items must have a numpy array as their first element.
    Returns (-1.0, -1) if *store* is empty.
    """
    if not store:
        return -1.0, -1
    best_sim = -1.0
    best_idx = -1
    for i, (emb, *_) in enumerate(store):
        sim = _cosine_sim(query_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return best_sim, best_idx


def _top_k_above_threshold(
    query_emb: np.ndarray,
    store: list[tuple],
    threshold: float,
    k: int,
) -> list[tuple[float, int]]:
    """Return up to *k* (sim, index) pairs with sim >= threshold, sorted desc."""
    scored: list[tuple[float, int]] = []
    for i, (emb, *_) in enumerate(store):
        sim = _cosine_sim(query_emb, emb)
        if sim >= threshold:
            scored.append((sim, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def _trajectory_to_summary(trajectory: dict, max_chars: int = 800) -> str:
    """Compact textual form of a dspy ReAct trajectory dict."""
    lines: list[str] = []
    i = 0
    while True:
        thought = trajectory.get(f"thought_{i}")
        tool_name = trajectory.get(f"tool_name_{i}")
        tool_args = trajectory.get(f"tool_args_{i}")
        observation = trajectory.get(f"observation_{i}")
        if thought is None and tool_name is None:
            break
        if thought:
            lines.append(f"T: {str(thought)[:200]}")
        if tool_name:
            args_str = str(tool_args)[:150] if tool_args is not None else ""
            lines.append(f"A: {tool_name}({args_str})")
        if observation is not None:
            lines.append(f"O: {str(observation)[:200]}")
        i += 1
    summary = "\n".join(lines)
    return summary[:max_chars]


def _build_demo_prefix(demos: list[tuple[str, str]]) -> str:
    """Build a demonstration prefix string from (question, trajectory_summary) pairs."""
    if not demos:
        return ""
    parts = ["=== Prior similar trajectories (for reference) ===\n"]
    for idx, (q, traj_summary) in enumerate(demos, start=1):
        parts.append(f"[Demo {idx}] Question: {q}\nTrajectory:\n{traj_summary}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# AnswerCacheAgent
# ---------------------------------------------------------------------------


class AnswerCacheAgent:
    """GPTCache-style answer caching baseline.

    On cache hit (similarity >= sim_threshold) the stored answer is returned
    immediately with zero tokens and zero iterations.  On miss, the ReAct
    agent is run and the result is stored if the judge deems it correct.

    Parameters
    ----------
    tools:
        List of dspy tool functions passed to dspy.ReAct.
    embedder:
        Callable[str] -> np.ndarray that returns a float32 embedding.
    judge:
        Callable[[str, str], bool] that returns True when the answer is correct.
    sim_threshold:
        Cosine similarity threshold above which a cached answer is reused.
    max_iters:
        Maximum ReAct iterations.
    react_factory:
        Optional callable(tools, max_iters) -> react_agent.  Defaults to
        constructing dspy.ReAct.  Provided for test injection.
    """

    def __init__(
        self,
        tools: list,
        embedder: Callable[[str], np.ndarray],
        judge: Callable[[str, str], bool],
        sim_threshold: float = 0.95,
        max_iters: int = 15,
        react_factory: Callable | None = None,
    ) -> None:
        self._tools = tools
        self._embedder = embedder
        self._judge = judge
        self._sim_threshold = sim_threshold
        self._max_iters = max_iters
        self._react_factory = react_factory

        self._cache: list[tuple[np.ndarray, str]] = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_react(self):
        """Construct (or delegate to factory for) the ReAct agent."""
        if self._react_factory is not None:
            return self._react_factory(self._tools, self._max_iters)
        import dspy  # deferred so tests can run without real API config
        from dspy.predict import ReAct

        class _QA(dspy.Signature):
            """Answer the question using the provided tools."""

            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        return ReAct(_QA, tools=self._tools, max_iters=self._max_iters)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict:
        """Run the agent or return a cached answer.

        Returns
        -------
        dict with keys:
            answer, trajectory, judgment, tokens, wall_time, n_iters, cache_hit
        """
        t0 = time.monotonic()
        q_emb = self._embedder(question)

        best_sim, best_idx = _best_similarity(q_emb, self._cache)

        if best_sim >= self._sim_threshold:
            cached_answer = self._cache[best_idx][1]
            judgment = self._judge(question, cached_answer)
            return {
                "answer": cached_answer,
                "trajectory": {},
                "judgment": judgment,
                "tokens": 0,
                "wall_time": time.monotonic() - t0,
                "n_iters": 0,
                "cache_hit": True,
            }

        import dspy  # deferred — align with the other ReAct paths

        react = self._build_react()
        lm = dspy.settings.lm
        history_before = _history_len(lm)
        result = react(question=question)

        answer: str = getattr(result, "answer", "") or ""
        trajectory: dict = getattr(result, "trajectory", {}) or {}
        tokens = _resolve_tokens(result, lm, history_before)
        n_iters: int = len([k for k in trajectory if k.startswith("thought_")])

        judgment = self._judge(question, answer)

        if judgment:
            self._cache.append((q_emb, answer))

        return {
            "answer": answer,
            "trajectory": trajectory,
            "judgment": judgment,
            "tokens": tokens,
            "wall_time": time.monotonic() - t0,
            "n_iters": n_iters,
            "cache_hit": False,
        }

    @property
    def cache_size(self) -> int:
        """Number of cached (embedding, answer) pairs."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# TrajectoryCacheAgent
# ---------------------------------------------------------------------------


class TrajectoryCacheAgent:
    """RAP-lite full-trajectory retrieval baseline.

    Stores (query_embedding, question, trajectory_summary).  For a new query,
    retrieves top-K stored trajectories whose similarity >= sim_threshold and
    injects them as in-context demonstrations in the agent prompt prefix before
    running the ReAct loop.

    Parameters
    ----------
    tools:
        List of dspy tool functions passed to dspy.ReAct.
    embedder:
        Callable[str] -> np.ndarray.
    judge:
        Callable[[str, str], bool].
    sim_threshold:
        Minimum cosine similarity for a stored trajectory to be retrieved.
    top_k:
        Maximum number of trajectories to inject.
    max_iters:
        Maximum ReAct iterations.
    react_factory:
        Optional callable(tools, max_iters, prefix) -> react_agent for test
        injection.  The *prefix* argument is the demo prefix string (may be
        empty).
    """

    def __init__(
        self,
        tools: list,
        embedder: Callable[[str], np.ndarray],
        judge: Callable[[str, str], bool],
        sim_threshold: float = 0.7,
        top_k: int = 3,
        max_iters: int = 15,
        react_factory: Callable | None = None,
    ) -> None:
        self._tools = tools
        self._embedder = embedder
        self._judge = judge
        self._sim_threshold = sim_threshold
        self._top_k = top_k
        self._max_iters = max_iters
        self._react_factory = react_factory

        self._store: list[tuple[np.ndarray, str, str]] = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _retrieve_demos(self, q_emb: np.ndarray) -> list[tuple[str, str]]:
        """Return list of (question, trajectory_summary) for top-K similar entries."""
        scored = _top_k_above_threshold(q_emb, self._store, self._sim_threshold, self._top_k)
        return [(self._store[idx][1], self._store[idx][2]) for _, idx in scored]

    def _build_react(self, prefix: str):
        """Construct the ReAct agent, injecting prefix into the signature docstring."""
        if self._react_factory is not None:
            return self._react_factory(self._tools, self._max_iters, prefix)

        import dspy
        from dspy.predict import ReAct

        doc = "Answer the question using the provided tools."
        if prefix:
            doc = prefix + "\n\n" + doc

        # Build signature dynamically so the prefix is baked into the docstring
        sig = dspy.Signature(
            {"question": (str, dspy.InputField()), "answer": (str, dspy.OutputField())},
            instructions=doc,
        )
        return ReAct(sig, tools=self._tools, max_iters=self._max_iters)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict:
        """Retrieve similar trajectories, inject as demos, then run ReAct.

        Returns
        -------
        dict with keys:
            answer, trajectory, judgment, tokens, wall_time, n_iters,
            cache_hit, demo_prefix
        """
        t0 = time.monotonic()
        q_emb = self._embedder(question)

        demos = self._retrieve_demos(q_emb)
        prefix = _build_demo_prefix(demos)

        import dspy  # deferred — align with the other ReAct paths

        react = self._build_react(prefix)
        lm = dspy.settings.lm
        history_before = _history_len(lm)
        result = react(question=question)

        answer: str = getattr(result, "answer", "") or ""
        trajectory: dict = getattr(result, "trajectory", {}) or {}
        tokens = _resolve_tokens(result, lm, history_before)
        n_iters: int = len([k for k in trajectory if k.startswith("thought_")])

        judgment = self._judge(question, answer)

        if judgment:
            traj_summary = _trajectory_to_summary(trajectory)
            self._store.append((q_emb, question, traj_summary))

        return {
            "answer": answer,
            "trajectory": trajectory,
            "judgment": judgment,
            "tokens": tokens,
            "wall_time": time.monotonic() - t0,
            "n_iters": n_iters,
            "cache_hit": False,
            "demo_prefix": prefix,
        }

    @property
    def store_size(self) -> int:
        """Number of stored (embedding, question, trajectory_summary) entries."""
        return len(self._store)


# ---------------------------------------------------------------------------
# Mem0ReActAgent
# ---------------------------------------------------------------------------


class Mem0ReActAgent:
    """DSPy/Mem0 tutorial-style baseline with agent-discretionary memory.

    Memory is exposed to the agent as three callable tools:
      - ``store_memory(content)``        — agent-initiated write
      - ``search_memories(query, limit)`` — semantic lookup
      - ``get_all_memories()``            — dump all memories

    The agent decides *if* and *when* to call these tools.  No automatic
    injection occurs.  This mirrors https://dspy.ai/tutorials/mem0_react_agent/.

    Parameters
    ----------
    tools:
        List of domain-specific dspy tool functions passed to dspy.ReAct.
    judge:
        Callable[[str, str], bool] that returns True when the answer is correct.
    user_id:
        Identifier used to namespace memories inside Mem0.
    mem0_config:
        Optional dict passed as ``config=`` to ``mem0.Memory()``.
        If None, a default in-memory Mem0 instance is created.
    mem0_memory:
        Optional pre-built ``mem0.Memory`` instance.  When provided this instance
        is used directly and ``mem0_config`` is ignored entirely — no new Qdrant
        client is opened.  Use this to share ONE Memory instance across many agent
        invocations and avoid the "Storage folder already accessed" lock error.
    max_iters:
        Maximum ReAct iterations.
    signature:
        DSPy signature string, e.g. ``"question -> answer"``.
    react_factory:
        Optional callable(tools, max_iters) -> react_agent.  Defaults to
        constructing dspy.ReAct.  Provided for test injection.
    """

    def __init__(
        self,
        tools: list,
        judge: Callable[[str, str], bool],
        user_id: str = "user",
        mem0_config: dict | None = None,
        mem0_memory: Any | None = None,
        max_iters: int = 15,
        signature: str = "question -> answer",
        react_factory: Callable | None = None,
    ) -> None:
        import hgc.baselines as _self_module

        _Memory = getattr(_self_module, "Memory", None)
        if _Memory is None:
            raise ImportError("mem0ai not installed. Install with: uv add mem0ai --optional mem0")

        self._user_tools = tools
        self._judge = judge
        self._user_id = user_id
        self._max_iters = max_iters
        self._signature = signature
        self._react_factory = react_factory

        if mem0_memory is not None:
            # Use the caller-supplied shared Memory instance directly.
            # This avoids opening a second Qdrant client at the same local path
            # (which would throw "Storage folder already accessed").
            self._mem0 = mem0_memory
        elif mem0_config is not None:
            # mem0ai v2.0: Memory() requires a MemoryConfig object, not a raw dict.
            # Convert here so callers can pass plain dicts for convenience.
            if isinstance(mem0_config, dict):
                from mem0.configs.base import MemoryConfig as _MemoryConfig

                mem0_config = _MemoryConfig(**mem0_config)
            self._mem0 = _Memory(config=mem0_config)
        else:
            self._mem0 = _Memory()

        # Adapts the mem0 v2 API (filters={"user_id": uid}, top_k=) to the
        # tutorial-compatible tool signatures (content, query/limit).
        mem0_instance = self._mem0
        uid = self._user_id

        def store_memory(content: str) -> str:
            """Store a memory string for future retrieval."""
            mem0_instance.add(content, user_id=uid)
            return f"Stored: {content[:80]}"

        def search_memories(query: str, limit: int = 3) -> list[dict]:
            """Search memories semantically for the given query."""
            response = mem0_instance.search(
                query,
                top_k=limit,
                filters={"user_id": uid},
            )
            # mem0 v2 returns {"results": [...]}; unwrap for the agent
            if isinstance(response, dict):
                return response.get("results", [])
            return response if isinstance(response, list) else []

        def get_all_memories() -> list[dict]:
            """Return all stored memories for the current user."""
            response = mem0_instance.get_all(filters={"user_id": uid})
            if isinstance(response, dict):
                return response.get("results", [])
            return response if isinstance(response, list) else []

        self._mem0_tools = [store_memory, search_memories, get_all_memories]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_react(self):
        """Construct (or delegate to factory for) the ReAct agent."""
        all_tools = self._user_tools + self._mem0_tools
        if self._react_factory is not None:
            return self._react_factory(all_tools, self._max_iters)
        from dspy.predict import ReAct

        return ReAct(self._signature, tools=all_tools, max_iters=self._max_iters)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict:
        """Run the ReAct agent with Mem0 tools available.

        Returns
        -------
        dict with keys:
            answer, trajectory, judgment, tokens, wall_time, n_iters, cache_hit
        """
        import dspy

        t0 = time.monotonic()

        lm = dspy.settings.lm
        history_before = _history_len(lm)

        react = self._build_react()
        try:
            result = react(question=question)
            answer: str = getattr(result, "answer", "") or ""
            trajectory: dict = dict(getattr(result, "trajectory", {}) or {})
        except Exception:
            answer = ""
            trajectory = {}

        tokens: int = _count_tokens_since(lm, history_before)
        n_iters: int = len([k for k in trajectory if k.startswith("thought_")])
        judgment: bool = bool(self._judge(question, answer))

        return {
            "answer": answer,
            "trajectory": trajectory,
            "judgment": judgment,
            "tokens": tokens,
            "wall_time": time.monotonic() - t0,
            "n_iters": n_iters,
            "cache_hit": False,
        }


# ---------------------------------------------------------------------------
# NaiveRAGAgent
# ---------------------------------------------------------------------------


class NaiveRAGAgent:
    """Single-shot Retrieve-then-Generate baseline (no agent loop, no memory).

    For each call to ``run(question, docs)``:
      1. Embed all docs via ``embedder.embed_batch``.
      2. Embed the question.
      3. Score by cosine similarity, select top-K.
      4. Concatenate selected doc texts as context (truncated to
         ``max_context_chars``).
      5. Compose a single prompt and call the LM once.
      6. Return the standard result record.

    Parameters
    ----------
    embedder:
        ``Embedder`` instance (must expose ``embed`` and ``embed_batch``).
    judge:
        Callable ``(question, answer) -> bool``.
    lm:
        Optional ``dspy.LM``; if None, ``dspy.settings.lm`` is used at
        call time.
    top_k:
        Number of top-scoring documents to include as context.
    max_chars_per_doc:
        Each doc's text is truncated to this length before embedding and
        before inclusion in the context.
    max_context_chars:
        Total context string length cap (applied after concatenation).
    """

    def __init__(
        self,
        embedder,
        judge: Callable[[str, str], bool],
        lm=None,
        top_k: int = 10,
        max_chars_per_doc: int = 2000,
        max_context_chars: int = 60_000,
        doc_emb_cache: DocEmbeddingCache | None = None,
    ) -> None:
        self._embedder = embedder
        self._judge = judge
        self._lm = lm
        self._top_k = top_k
        self._max_chars_per_doc = max_chars_per_doc
        self._max_context_chars = max_context_chars
        self._doc_emb_cache = doc_emb_cache

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_lm(self, prompt: str) -> str:
        """Return the first completion string from the configured LM."""
        lm = self._lm
        if lm is None:
            import dspy

            lm = dspy.settings.lm
        result = lm(prompt)
        if isinstance(result, list):
            return str(result[0]) if result else ""
        return str(result)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, question: str, docs: list[dict]) -> dict:
        """Retrieve top-K docs, generate one answer, return result record.

        Parameters
        ----------
        question:
            The user question.
        docs:
            List of dicts each with at least ``{"docid": str, "text": str}``.

        Returns
        -------
        dict with keys:
            answer, trajectory, judgment, tokens, wall_time, n_iters,
            cache_hit
        """
        t0 = time.monotonic()

        texts = [d.get("text", "")[: self._max_chars_per_doc] for d in docs]
        docids = [d.get("docid", str(i)) for i, d in enumerate(docs)]

        if texts:
            if self._doc_emb_cache is not None:
                # --- cache-assisted embedding ---
                hits = self._doc_emb_cache.get_many(docids)
                miss_indices = [i for i, did in enumerate(docids) if did not in hits]

                if miss_indices:
                    miss_texts = [texts[i] for i in miss_indices]
                    miss_vecs = self._embedder.embed_batch(miss_texts)
                    for idx, vec in zip(miss_indices, miss_vecs, strict=False):
                        self._doc_emb_cache.put(docids[idx], vec)
                        hits[docids[idx]] = vec

                # Assemble doc_embs in original doc order
                doc_embs = np.stack([hits[did] for did in docids])
                self._doc_emb_cache.save()
            else:
                doc_embs = self._embedder.embed_batch(texts)
        else:
            doc_embs = np.empty((0, 1), dtype=np.float32)

        q_emb = self._embedder.embed(question)

        scores: list[float] = [float(_cosine_sim(q_emb, doc_embs[i])) for i in range(len(docs))]
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked[: self._top_k]
        retrieved_docids = [docids[i] for i in top_indices]

        context_parts = [f"[{docids[i]}] {texts[i]}" for i in top_indices]
        context = "\n\n".join(context_parts)[: self._max_context_chars]

        prompt = (
            f"Answer the following question using only the provided documents.\n\n"
            f"Question: {question}\n\n"
            f"Documents:\n{context}\n\n"
            f"Answer:"
        )

        lm = self._lm
        if lm is None:
            import dspy

            lm = dspy.settings.lm

        tokens_before = _history_len(lm)
        answer = self._call_lm(prompt)
        tokens_used = _count_tokens_since(lm, tokens_before)

        wall_time = time.monotonic() - t0
        judgment = bool(self._judge(question, answer))

        return {
            "answer": answer,
            "trajectory": {
                "retrieved_docids": retrieved_docids,
                "prompt_preview": prompt[:500],
            },
            "judgment": judgment,
            "tokens": tokens_used,
            "wall_time": wall_time,
            "n_iters": 1,
            "cache_hit": False,
        }


# ---------------------------------------------------------------------------
# LongContextStuffAgent
# ---------------------------------------------------------------------------

# Reuse a single tiktoken encoder across all instances.
_TOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4")

# Rough token budget reserved for prompt scaffolding, the question itself, and
# the model's completion.  Subtract this from max_input_tokens before packing docs.
_STUFFING_SCAFFOLD_TOKENS = 1000


class LongContextStuffAgent:
    """Long-context stuffing baseline — single LLM call with all docs concatenated.

    No retrieval, no agent loop, no memory.  Tests whether long-context LLMs
    obviate the need for agentic iteration.

    Parameters
    ----------
    judge:
        Callable ``(question, answer) -> bool``.
    lm:
        Optional ``dspy.LM``; if None, ``dspy.settings.lm`` is used at call time.
    max_input_tokens:
        Hard token budget for the full prompt (docs + scaffolding).  Docs are
        greedily included in list order until this budget is exhausted.
    max_chars_per_doc:
        Each doc's text is truncated to this many characters before token
        counting and inclusion in the prompt.
    doc_order:
        Controls document placement order before stuffing.  Four modes:

        ``"primacy"`` (default)
            Docs are left in the caller-supplied order.  Gold/evidence docs
            typically arrive first, so they benefit from primacy attention.

        ``"recency"``
            Docs are reversed.  Gold docs end up at the tail of the prompt,
            benefiting from recency attention.

        ``"middle"``
            The first 10 docs (gold + evidence) are placed in the centre of
            the doc list, surrounded by negatives on both sides.  This is the
            Lost-in-the-Middle test: if the model struggles to attend to the
            middle, accuracy should drop relative to ``"primacy"``.
            Concretely: ``docs[10:mid] + docs[:10] + docs[mid:]`` where
            ``mid = len(docs) // 2``.

        ``"random"``
            Docs are shuffled with ``random.Random(doc_order_seed).sample``.

    doc_order_seed:
        RNG seed used only for ``doc_order="random"``.  Default 42.
    """

    def __init__(
        self,
        judge: Callable[[str, str], bool],
        lm: Any = None,
        max_input_tokens: int = 900_000,
        max_chars_per_doc: int = 8_000,
        doc_order: Literal["primacy", "recency", "middle", "random"] = "primacy",
        doc_order_seed: int = 42,
    ) -> None:
        self._judge = judge
        self._lm = lm
        self._max_input_tokens = max_input_tokens
        self._max_chars_per_doc = max_chars_per_doc
        self._doc_order = doc_order
        self._doc_order_seed = doc_order_seed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reorder_docs(self, docs: list[dict]) -> list[dict]:
        """Return docs reordered according to ``self._doc_order``.

        ``"primacy"``  — no change (gold at head, as caller supplied).
        ``"recency"``  — reversed (gold at tail).
        ``"middle"``   — first 10 docs placed in centre, negatives on both
                         sides.  Simulates Lost-in-the-Middle degradation.
        ``"random"``   — seeded shuffle.
        """
        if self._doc_order == "primacy":
            return list(docs)
        if self._doc_order == "recency":
            return list(reversed(docs))
        if self._doc_order == "middle":
            # Place the first 10 (gold + evidence) in the centre of the list.
            # The negatives (docs[10:]) are split in half; gold goes between them.
            # docs[:10]                → gold/evidence block
            # docs[10 : 10 + neg_half] → first half of negatives (before gold)
            # docs[10 + neg_half :]    → second half of negatives (after gold)
            gold_block = docs[:10]
            negatives = docs[10:]
            neg_half = len(negatives) // 2
            first_half_neg = negatives[:neg_half]
            second_half_neg = negatives[neg_half:]
            return list(first_half_neg) + list(gold_block) + list(second_half_neg)
        if self._doc_order == "random":
            return random.Random(self._doc_order_seed).sample(docs, len(docs))
        # Fallback — treat unknown as primacy
        return list(docs)

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Return the approximate token count for *text* via tiktoken."""
        return len(_TOKEN_ENCODER.encode(text))

    def _build_prompt(self, question: str, docs: list[dict]) -> tuple[str, list[str], bool]:
        """Pack docs greedily into a prompt within the token budget.

        Returns
        -------
        (prompt_str, stuffed_docids, truncated)
        """
        budget = self._max_input_tokens - _STUFFING_SCAFFOLD_TOKENS
        # Reserve tokens for question line and the fixed scaffold text.
        question_tokens = self._count_tokens(question)
        budget -= question_tokens

        doc_lines: list[str] = []
        stuffed_docids: list[str] = []
        truncated = False

        for doc in docs:
            docid = doc.get("docid", "")
            text = doc.get("text", "")[: self._max_chars_per_doc]
            line = f"[docid={docid}] {text}"
            line_tokens = self._count_tokens(line)
            if line_tokens > budget:
                truncated = True
                break
            doc_lines.append(line)
            stuffed_docids.append(docid)
            budget -= line_tokens

        docs_block = "\n".join(doc_lines)
        prompt = (
            "You are answering a question using the provided documents.\n\n"
            f"Documents:\n{docs_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return prompt, stuffed_docids, truncated

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, question: str, docs: list[dict]) -> dict:
        """Stuff all docs into a single prompt and call the LM once.

        Parameters
        ----------
        question:
            The user question.
        docs:
            List of dicts each with at least ``{"docid": str, "text": str}``.
            Docs are included in list order; gold/evidence docs should appear
            first in the upstream combining step.

        Returns
        -------
        dict with keys:
            answer, trajectory, judgment, tokens, wall_time, n_iters, cache_hit
        """
        t0 = time.monotonic()

        # Apply doc ordering strategy before stuffing.
        docs = self._reorder_docs(docs)

        prompt, stuffed_docids, truncated = self._build_prompt(question, docs)
        total_input_chars = len(prompt)

        lm = self._lm
        if lm is None:
            import dspy

            lm = dspy.settings.lm

        tokens_before = _history_len(lm)
        raw = lm(prompt)
        if isinstance(raw, list):
            answer = str(raw[0]) if raw else ""
        else:
            answer = str(raw)
        tokens_used = _count_tokens_since(lm, tokens_before)

        judgment = bool(self._judge(question, answer))

        return {
            "answer": answer,
            "trajectory": {
                "stuffed_docids": stuffed_docids,
                "truncated": truncated,
                "total_input_chars": total_input_chars,
                "doc_order": self._doc_order,
            },
            "judgment": judgment,
            "tokens": tokens_used,
            "wall_time": time.monotonic() - t0,
            "n_iters": 1,
            "cache_hit": False,
        }


# ---------------------------------------------------------------------------
# Internal helpers (shared with agent.py pattern)
# ---------------------------------------------------------------------------


def _history_len(lm: Any) -> int:
    """Return the current length of the LM history list (0 if unavailable)."""
    if lm is None:
        return 0
    history = getattr(lm, "history", None)
    if not isinstance(history, list):
        return 0
    return len(history)


def _count_tokens_since(lm: Any, baseline: int) -> int:
    """Sum total_tokens from LM history entries added after *baseline* index."""
    if lm is None:
        return 0
    history = getattr(lm, "history", None)
    if not isinstance(history, list):
        return 0
    total = 0
    for h in history[baseline:]:
        usage = h.get("usage", {}) if isinstance(h, dict) else {}
        if isinstance(usage, dict):
            total += usage.get("total_tokens", 0)
    return total


def _resolve_tokens(result: Any, lm: Any, history_before: int) -> int:
    """Return the token count for a ReAct fallback run.

    Production DSPy ReAct does not attach a ``.tokens`` attribute to its
    result; it writes usage into ``lm.history[-1]["usage"]["total_tokens"]``.
    Some unit-test mocks, however, expose ``result.tokens`` directly. Prefer
    the mock attribute when present (preserves legacy test contracts), fall
    back to the canonical history-delta accounting otherwise.
    """
    direct = getattr(result, "tokens", None)
    if isinstance(direct, int) and direct > 0:
        return direct
    return _count_tokens_since(lm, history_before)
