"""HGC core agent — dspy.ReAct with hint retrieval/injection and post-run memory update.

Implements DESIGN.md §3 (Retrieval & Injection) and §4 (Update Rules).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

import dspy
import numpy as np

from hgc.embeddings import Embedder
from hgc.extraction import HintExtractor
from hgc.memory import HintRecord, HintStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

_HINT_TYPE_PREFIX: dict[str, str] = {
    "location": "L",
    "entity": "E",
    "strategy": "S",
}

_HINT_SUFFIX = "\nYou may re-verify these hints; they are suggestions, not answers."

_AGENT_SUFFIX = (
    "\n\nYou have access to hints from past similar queries. Use them as starting points"
    " for exploration, but always verify by reading the actual documents. Do NOT"
    " assume the hints contain the final answer."
)


def format_hints_prompt(
    positive_hints: list[HintRecord],
    negative_hints: list[HintRecord],
) -> str:
    """Build the hint prefix block as specified in DESIGN.md §3.

    Example output::

        === Hints from prior similar queries ===

        HELPFUL (positive hints from past successes):
          [L1] Document 5412 contained relevant info for similar queries.
          [E1] Entity "Queen Arwa University" was referenced in this context.
          [S1] The search query "cultural activities 2002" returned useful results.

        AVOID (negative hints from past failures):
          [N1] Document 26215 was checked but did not help.

        You may re-verify these hints; they are suggestions, not answers.
    """
    if not positive_hints and not negative_hints:
        return ""

    lines: list[str] = ["=== Hints from prior similar queries ===", ""]

    if positive_hints:
        lines.append("HELPFUL (positive hints from past successes):")
        counters: dict[str, int] = {}
        for h in positive_hints:
            prefix = _HINT_TYPE_PREFIX.get(h.hint_type, "H")
            counters[prefix] = counters.get(prefix, 0) + 1
            tag = f"[{prefix}{counters[prefix]}]"
            lines.append(f"  {tag} {h.content}")
        lines.append("")

    if negative_hints:
        lines.append("AVOID (negative hints from past failures):")
        for i, h in enumerate(negative_hints, 1):
            lines.append(f"  [N{i}] {h.content}")
        lines.append("")

    lines.append(_HINT_SUFFIX.strip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HGCCoreAgent
# ---------------------------------------------------------------------------


class HGCCoreAgent:
    """ReAct agent augmented with hint-based persistent memory.

    Parameters
    ----------
    tools:
        List of callable tools passed to dspy.ReAct.
    store:
        HintStore instance (SQLite-backed or in-memory).
    embedder:
        Embedder used to embed the question for retrieval.
    extractor:
        HintExtractor used to extract new hints post-run.
    judge:
        Callable ``judge(question: str, answer: str) -> bool`` scoring correctness.
    max_iters:
        Maximum ReAct iterations.
    signature:
        DSPy signature string, e.g. ``"question -> answer"``.
    k_positive:
        Top-K positive hints to retrieve.
    k_negative:
        Top-K negative hints to retrieve.
    alpha, beta, gamma:
        Scoring weights passed to HintStore.search().
    theta_pos:
        Minimum cosine similarity for positive hints to be retrieved (default 0.3).
    theta_neg:
        Minimum cosine similarity for negative hints to be retrieved (default 0.6).
    """

    def __init__(
        self,
        tools: list[Callable],
        store: HintStore,
        embedder: Embedder,
        extractor: HintExtractor,
        judge: Callable[[str, str], bool],
        max_iters: int = 15,
        signature: str = "question -> answer",
        k_positive: int = 5,
        k_negative: int = 3,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        theta_pos: float = 0.3,
        theta_neg: float = 0.6,
        scope_id: str = "default",
    ) -> None:
        self._tools = tools
        self._store = store
        self._embedder = embedder
        self._extractor = extractor
        self._judge = judge
        self._max_iters = max_iters
        self._signature = signature
        self._k_positive = k_positive
        self._k_negative = k_negative
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._theta_pos = theta_pos
        self._theta_neg = theta_neg
        self._scope_id = scope_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        docs: list[dict] | None = None,
        valid_docids: frozenset[str] | None = None,
    ) -> dict:
        """Run the agent on *question* and return a record dict.

        Parameters
        ----------
        question:
            The question to answer.
        docs:
            Optional list of doc dicts (each with a ``"docid"`` key).  When
            provided, ``valid_docids`` is derived from this list automatically
            (any explicitly passed ``valid_docids`` takes precedence).
        valid_docids:
            Optional frozenset of docid strings that are valid for this query.
            When provided, location hints whose parsed docid is not in this set
            are dropped before prompt injection (defense-in-depth filter).
            When None, no filtering is applied (backward compat).

        Returns
        -------
        dict with keys:
            answer, trajectory, retrieved_positive_hints, retrieved_negative_hints,
            added_hints, judgment, tokens, wall_time, n_iters
        """
        # Derive valid_docids from docs if not explicitly provided
        if valid_docids is None and docs is not None:
            valid_docids = frozenset(str(d["docid"]) for d in docs if "docid" in d)

        t0 = time.time()

        q_emb = self._embedder.embed(question)

        all_retrieved = self._store.search(
            query_embedding=q_emb,
            k=self._k_positive + self._k_negative,
            alpha=self._alpha,
            beta=self._beta,
            gamma=self._gamma,
            theta_pos=self._theta_pos,
            theta_neg=self._theta_neg,
            scope_id=self._scope_id,
        )
        positive_hints = [h for h in all_retrieved if h.polarity == "positive"][: self._k_positive]
        negative_hints = [h for h in all_retrieved if h.polarity == "negative"][: self._k_negative]

        # Apply docid validation filter to location hints
        if valid_docids is not None:
            positive_hints = _filter_location_hints(positive_hints, valid_docids)
            negative_hints = _filter_location_hints(negative_hints, valid_docids)

        hint_prefix = format_hints_prompt(positive_hints, negative_hints)
        if hint_prefix:
            augmented_question = hint_prefix + _AGENT_SUFFIX + "\n\n" + question
        else:
            augmented_question = question

        lm = dspy.settings.lm
        history_before = _history_len(lm)

        react_agent = dspy.ReAct(self._signature, tools=self._tools, max_iters=self._max_iters)
        try:
            out = react_agent(question=augmented_question)
            answer = out.answer
            trajectory: dict = dict(getattr(out, "trajectory", {}))
        except Exception:
            answer = ""
            trajectory = {}

        tokens = _count_tokens_since(lm, history_before)

        judgment: bool = bool(self._judge(question, answer))
        extracted = self._extractor.extract(question, trajectory, correct=judgment)

        now = time.time()
        q_emb_bytes: bytes = q_emb.astype(np.float32).tobytes()
        added_hints: list[str] = []

        polarity: Literal["positive", "negative"] = "positive" if judgment else "negative"

        for hint_type in ("location", "entity", "strategy"):
            for item in extracted.get(hint_type, []):
                content = item.get("content", "").strip()
                if not content:
                    continue
                record = HintRecord(
                    hint_id=str(uuid.uuid4()),
                    hint_type=hint_type,  # type: ignore[arg-type]
                    polarity=polarity,
                    content=content,
                    content_meta=item.get("content_meta", {}),
                    query_ctx=question,
                    query_ctx_embedding=q_emb_bytes,
                    trajectory_step=0,
                    created_at=now,
                    last_validated_at=now,
                    success_count=0,
                    failure_count=0,
                    retrieval_count=0,
                    scope_id=self._scope_id,
                )
                hint_id = self._store.add(record)
                added_hints.append(hint_id)

        retrieved_ids = [h.hint_id for h in positive_hints + negative_hints]
        if retrieved_ids:
            self._store.update_on_outcome(retrieved_ids, correct=judgment)

        wall_time = time.time() - t0
        n_iters = sum(1 for k in trajectory if k.startswith("thought_"))

        return {
            "answer": answer,
            "trajectory": trajectory,
            "retrieved_positive_hints": positive_hints,
            "retrieved_negative_hints": negative_hints,
            "added_hints": added_hints,
            "judgment": judgment,
            "tokens": tokens,
            "wall_time": wall_time,
            "n_iters": n_iters,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_docid(content: str) -> str | None:
    """Extract a docid string from a hint content string.

    Handles three forms:
    - ``"docid=5412"``   → ``"5412"``
    - ``"5412"``         → ``"5412"`` (if the whole string is purely numeric)
    - anything else      → ``None``   (not a docid-style hint, skip validation)

    Parameters
    ----------
    content:
        The hint content string to parse.

    Returns
    -------
    str | None
        The extracted docid, or None if the content is not a docid hint.
    """
    content = content.strip()
    # Pattern: "docid=NNN" (anywhere in the string)
    m = re.search(r"docid=(\d+)", content)
    if m:
        return m.group(1)
    # Pattern: purely numeric string
    if re.fullmatch(r"\d+", content):
        return content
    return None


def _filter_location_hints(
    hints: list[HintRecord],
    valid_docids: frozenset[str],
) -> list[HintRecord]:
    """Drop location hints whose docid is not in *valid_docids*.

    Non-location hints and location hints whose content does not parse as a
    docid are passed through unchanged.

    Logs at DEBUG level when any hints are dropped.
    """
    kept: list[HintRecord] = []
    dropped = 0
    for h in hints:
        if h.hint_type != "location":
            kept.append(h)
            continue
        docid = _parse_docid(h.content)
        if docid is None:
            # Content is not a docid pattern — let it through (no validation)
            kept.append(h)
            continue
        if docid in valid_docids:
            kept.append(h)
        else:
            dropped += 1

    if dropped:
        logger.debug("Docid filter: dropped %d/%d location hints", dropped, dropped + len(kept))

    return kept


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
