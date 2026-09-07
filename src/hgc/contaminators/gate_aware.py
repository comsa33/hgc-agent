"""Gate-aware contamination: poison that is built to survive the gate.

The three random modes (cross-swap, entity-swap, typo-mutation) model an
upstream system producing wrong answers, which is what the paper's threat
model claims. They say nothing about an adversary who knows the gate is there.
This module supplies that adversary, for the one opening the gate actually
has.

``SupportVerifier`` short-circuits before the LLM ever runs: if the candidate
answer appears verbatim in the source document, support is granted (see
``gate/support_verifier.py``, ``_containment_pass``). The check asks whether
the string is *in* the document, never whether it *answers the question*. So
a poisoned entry whose answer is a sentence lifted out of the victim's own
source document passes every component:

  G1  the entry keeps its own scope, so the scope filter is satisfied;
  G2  it keeps its own location hint, so the docid resolves in the pool;
  G3  containment matches on the first try, and no LLM call is made.

Contrast with cross-swap, where the answer comes from a *different* document
and G3 has to reason about it. The point of the experiment is not that HGC
fails — it is to state precisely where the boundary of the threat model sits,
which ``limitations.tex`` already predicts in prose. The appendix's QASPER-RAG
false accepts are a different thing: those rows reached the verifier LM and it
approved them, and most of them served answers the clean run served too.

Victim selection mirrors :mod:`hgc.contaminators.cross_swap` exactly (same
seed, same ``rng.sample``), so the two modes corrupt the same entries and the
comparison isolates the poison's construction rather than its placement.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable as _Callable
from math import floor as _floor
from pathlib import Path as _Path
from typing import TYPE_CHECKING
from typing import Any as _Any

if TYPE_CHECKING:
    from hgc.memory import HintRecord, HintStore

# A source of poison documents for one seeded P1 record, in the order the
# gate will read them. The default reads the trajectory's own ``context``.
ContextsOf = _Callable[[dict], list[str]]

# Header the long-context and RAG backbones prepend to each document they read.
_DOCID_HEADER = re.compile(r"\[docid=[^\]]*\]\s*")
# Sentence-ish split. Deliberately crude: the span only has to be a verbatim
# substring of the document, so imperfect boundaries cost coverage, not
# correctness — and every candidate is verified against the source before use.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalise(text: str) -> str:
    """Match SupportVerifier._normalise so containment is judged identically."""
    return " ".join(text.lower().split())


def usable_spans(context: str, gold: str, min_chars: int = 5) -> list[str]:
    """Sentences of *context* that are safe to use as gate-passing poison.

    A span qualifies when it is long enough to clear the containment floor,
    is verbatim present in the document, and does **not** contain the gold
    answer — a span carrying the gold answer would be scored correct and the
    attack would poison nothing.
    """
    body = _DOCID_HEADER.sub("", context)
    document = _normalise(body)
    gold_norm = _normalise(gold)

    spans: list[str] = []
    for raw in _SENTENCE_SPLIT.split(body):
        span = raw.strip()
        if len(span) < min_chars:
            continue
        span_norm = _normalise(span)
        if gold_norm and gold_norm in span_norm:
            continue
        # Must survive the very check it is meant to defeat.
        if span_norm not in document:
            continue
        spans.append(span)
    return spans


def default_contexts(record: dict) -> list[str]:
    """The long-context and RAG backbones record what they read as ``trajectory.context``."""
    context = (record.get("trajectory") or {}).get("context", "")
    return [context] if context else []


def rank_location_hints(
    store: HintStore,
    query_embedding: _Any,
    scope_id: str | None,
    *,
    k: int = 3,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    t_half_days: float = 30.0,
    theta_pos: float = 0.3,
    theta_neg: float = 0.6,
) -> list[HintRecord]:
    """The location hints the gate will try for *query_embedding*, in gate order.

    A read-only mirror of ``HintStore.search`` followed by the agents'
    ``_retrieve_location_hints`` filter: the store's own ``search`` bumps
    ``retrieval_count`` on every call, and this module must not leave a mark
    on the hint store it is only looking at. Kept in lockstep with
    ``memory._SQLiteBackend.search`` by a test that compares the two.
    """
    import numpy as np

    from hgc.memory import _confidence, _cosine_sim, _recency

    q_emb = np.asarray(query_embedding, dtype=np.float32)
    candidates: list[tuple[float, HintRecord]] = []
    for h in store.all():
        if scope_id is not None and h.hint_type == "location" and h.scope_id != scope_id:
            continue
        sim = _cosine_sim(q_emb, np.frombuffer(h.query_ctx_embedding, dtype=np.float32))
        if h.polarity == "positive" and sim < theta_pos:
            continue
        if h.polarity == "negative" and sim < theta_neg:
            continue
        score = (
            alpha * sim
            + beta * _confidence(h.success_count, h.failure_count)
            + gamma * _recency(h.created_at, t_half_days)
        )
        candidates.append((score, h))
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = [h for _, h in candidates[: k * 3]]
    return [h for h in top if h.hint_type == "location"][:k]


def hint_document_contexts(
    store: HintStore,
    embedder: _Callable[[str], _Any],
    queries: list[dict],
    scope_of: _Callable[[dict], str],
    *,
    top_k: int = 3,
    **rank_params: float,
) -> ContextsOf:
    """Poison source for backbones whose trajectory does not carry the document.

    The ReAct backbone on BCP records tool calls, not the page it read, and
    the RAG backbone's ``context`` concatenates five sections of which the
    gate will read one. Both cases need the same thing: the text of the
    document that the entry's own location hint resolves to, because that is
    the document ``CompositeGate`` hands to $G_3$. This hook looks the record's
    query up in the dataset, ranks the scope's location hints the way the gate
    will at serve time (with the original question standing in for the
    paraphrase), and returns those hints' documents in that order.
    """
    from hgc.gate.docid_check import resolve_docid
    from hgc.runner import _collect_docs

    by_qid = {str(qr.get("query_id")): qr for qr in queries}

    def contexts(record: dict) -> list[str]:
        qr = by_qid.get(str(record.get("query_id")))
        if qr is None:
            return []
        doc_map = {str(d["docid"]): d.get("text", "") for d in _collect_docs(qr)}
        if not doc_map:
            return []
        hints = rank_location_hints(
            store, embedder(record.get("question", "")), scope_of(qr), k=top_k, **rank_params
        )
        out: list[str] = []
        for hint in hints:
            text = doc_map.get(resolve_docid(hint.content, doc_map) or "", "")
            if text and text not in out:
                out.append(text)
        return out

    return contexts


def _seeded_records(p1_trajectories: list[_Path]) -> list[dict]:
    """P1 records in the order ``seed_answer_cache_from_p1`` appends them.

    Kept in lockstep with that function: judge-correct, non-empty question and
    answer. If the two drift apart the cache index no longer identifies the
    trajectory it came from, and the poison lands on the wrong document.
    """
    import json

    records: list[dict] = []
    for path in sorted(p1_trajectories):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        question = data.get("question", "")
        answer = data.get("pred", data.get("answer", ""))
        judgment = data.get("judgment_correct", data.get("judgment", False))
        if not question or not answer or not judgment:
            continue
        records.append(data)
    return records


def contaminate_answer_cache_from_p1(
    ac_agent: _Any,
    p1_trajectories: list[_Path],
    embedder: _Callable[[str], _Any],
    seed: int = 42,
    fraction: float = 0.20,
    min_chars: int = 5,
    contexts_of: ContextsOf | None = None,
) -> list[int]:
    """Seed the cache from P1, then replace victim answers with gate-passing spans.

    Returns the sorted indices actually corrupted. An entry whose document
    yields no usable span is left clean and excluded from the return value, so
    the caller can tell attempted coverage from achieved coverage rather than
    assuming every victim was poisoned.

    *contexts_of* names where the poison comes from. The default reads the
    trajectory's own ``context`` (long-context cells); backbones whose
    trajectory does not carry the document pass
    :func:`hint_document_contexts`. Documents are tried in the order given
    and the first that yields a usable span is used.
    """
    from hgc.runner import seed_answer_cache_from_p1

    seed_answer_cache_from_p1(ac_agent, p1_trajectories, embedder)

    records = _seeded_records(p1_trajectories)
    n = len(ac_agent._cache)
    if n != len(records):
        raise ValueError(
            f"cache/trajectory misalignment: {n} cache entries vs {len(records)} "
            "seeded records — the seeding filter changed underneath this module"
        )
    if n < 1:
        raise ValueError("cannot contaminate an empty cache")

    k = _floor(fraction * n)
    if k == 0:
        return []

    rng = random.Random(seed)
    victim_indices: list[int] = rng.sample(range(n), k)

    source = contexts_of or default_contexts
    corrupted: list[int] = []
    for i in victim_indices:
        record = records[i]
        spans: list[str] = []
        for context in source(record):
            spans = usable_spans(context, record.get("gold", ""), min_chars=min_chars)
            if spans:
                break
        if not spans:
            continue
        # Deterministic per victim, mirroring cross_swap's sub-rng convention.
        span = random.Random(seed + i).choice(spans)
        ac_agent._cache[i] = (ac_agent._cache[i][0], span)
        corrupted.append(i)

    return sorted(corrupted)
