"""HGC and AC patterns on a RAG backbone.

Mirrors :class:`hgc.hgc_agent.HGCAgent`, but the fallback is a single-shot
RAG pipeline instead of an iterative DSPy ReAct loop.

Same four paths (cache_miss / cache_hit_no_hint / cache_verified /
cache_fallback) so downstream stats scripts and paper tables work
identically across backbones. The opt-in ``strict_no_hint`` variant emits a
fifth label (``cache_hit_no_hint_strict``) in place of ``cache_hit_no_hint``;
it appears only when strict mode is enabled, so the default path taxonomy that
downstream consumers rely on is unchanged.

:class:`ACRAGAgent` is the gate-less counterpart used as the AC baseline on
RAG benchmarks. It mirrors HGCRAGAgent's fallback architecture so that AC
and HGC cost comparisons are apples-to-apples on RAG benchmarks.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from hgc.baselines import _best_similarity
from hgc.gate import CompositeGate, GateDecision
from hgc.hgc_core import _count_tokens_since, _history_len
from hgc.memory import HintRecord


class HGCRAGAgent:
    """HGC agent whose fallback is a RAG pipeline.

    Parameters
    ----------
    ac_agent:
        AnswerCacheAgent instance.
    rag_backbone:
        :class:`hgc.rag_backbone.RAGBackbone`-compatible object exposing
        ``.run(question, docs)``.
    store:
        HintStore from which the gate's candidate location hints are pulled.
    embedder:
        Callable ``str -> np.ndarray`` used to embed the question for hint retrieval.
    gate:
        CompositeGate (default full gate).
    top_k_hints:
        Number of candidate location hints retrieved per query. Default 3.
    hint_retrieval_params:
        Dict of (alpha, beta, gamma, theta_pos, theta_neg, scope_id). RAG has
        no HGCCoreAgent to pull these from, so the RAG-side caller must pass
        them explicitly.
    """

    def __init__(
        self,
        ac_agent: Any,
        rag_backbone: Any,
        store: Any,
        embedder: Callable[[str], np.ndarray],
        hint_retrieval_params: dict,
        gate: CompositeGate | None = None,
        top_k_hints: int = 3,
        strict_no_hint: bool = False,
    ) -> None:
        self._ac = ac_agent
        self._rag = rag_backbone
        self._store = store
        self._embedder = embedder
        self._params = hint_retrieval_params
        self._gate = gate if gate is not None else CompositeGate()
        self._top_k = top_k_hints
        self._strict_no_hint = strict_no_hint

    def _retrieve_location_hints(self, query_emb: np.ndarray) -> list[HintRecord]:
        hints = self._store.search(
            query_embedding=query_emb,
            k=self._top_k * 3,
            alpha=self._params.get("alpha", 1.0),
            beta=self._params.get("beta", 0.5),
            gamma=self._params.get("gamma", 0.3),
            theta_pos=self._params.get("theta_pos", 0.3),
            theta_neg=self._params.get("theta_neg", 0.6),
            scope_id=self._params.get("scope_id"),
        )
        return [h for h in hints if h.hint_type == "location"][: self._top_k]

    def run(self, question: str, docs: list[dict] | None = None) -> dict:
        import dspy

        t0 = time.monotonic()
        lm = dspy.settings.lm
        history_before = _history_len(lm)

        # Cache lookup (no LM cost): bypass AnswerCacheAgent.run()'s ReAct
        # fallback path; we control the fallback ourselves below.
        q_emb = self._embedder(question)
        best_sim, best_idx = _best_similarity(q_emb, self._ac._cache)
        cache_hit = best_sim >= self._ac._sim_threshold
        ac_answer = self._ac._cache[best_idx][1] if cache_hit else ""

        if not cache_hit:
            rag_result = self._rag.run(question, docs=docs)
            rag_tokens = int(rag_result.get("tokens") or 0)
            return {
                **rag_result,
                "tokens": _count_tokens_since(lm, history_before) + rag_tokens,
                "gate_tokens": 0,
                "wall_time": time.monotonic() - t0,
                "cache_hit": False,
                "path": "cache_miss",
            }

        query_emb = self._embedder(question)
        location_hints = self._retrieve_location_hints(query_emb)

        if not location_hints:
            if self._strict_no_hint:
                # Strict-fallback variant (previous-round ablation): reroute
                # an uncovered cache hit through the RAG backbone instead of
                # trusting the cached answer. Default policy is unchanged.
                rag_result = self._rag.run(question, docs=docs)
                rag_tokens = int(rag_result.get("tokens") or 0)
                return {
                    **rag_result,
                    "tokens": _count_tokens_since(lm, history_before) + rag_tokens,
                    "gate_tokens": 0,
                    "wall_time": time.monotonic() - t0,
                    "cache_hit": False,
                    "path": "cache_hit_no_hint_strict",
                }
            return {
                "answer": ac_answer,
                "trajectory": {},
                "wall_time": time.monotonic() - t0,
                "cache_hit": True,
                "tokens": _count_tokens_since(lm, history_before),
                "gate_tokens": 0,
                "n_iters": 0,
                "path": "cache_hit_no_hint",
            }

        pre_gate_history = _history_len(lm)
        decision: GateDecision = self._gate(
            question=question,
            answer=ac_answer,
            hints=location_hints,
            docs=docs,
            scope_id=self._params.get("scope_id"),
        )
        gate_tokens = _count_tokens_since(lm, pre_gate_history)

        if decision.passed:
            return {
                "answer": ac_answer,
                "trajectory": {},
                "wall_time": time.monotonic() - t0,
                "cache_hit": True,
                "tokens": _count_tokens_since(lm, history_before),
                "gate_tokens": gate_tokens,
                "n_iters": 0,
                "path": "cache_verified",
                "gate_evidence": getattr(decision.evidence_hint, "hint_id", None),
                "gate_stage_counts": decision.stage_counts,
            }

        rag_result = self._rag.run(question, docs=docs)
        rag_tokens = int(rag_result.get("tokens", 0))
        return {
            **rag_result,
            "tokens": _count_tokens_since(lm, history_before) + rag_tokens,
            "gate_tokens": gate_tokens,
            "wall_time": time.monotonic() - t0,
            "cache_hit": False,
            "path": "cache_fallback",
            "gate_reject_reason": decision.reason,
            "gate_stage_counts": decision.stage_counts,
        }


class ACRAGAgent:
    """AnswerCache pattern with single-shot RAG fallback (no gate).

    The AC counterpart of :class:`HGCRAGAgent`: cache hit returns the cached
    answer (no LM cost); cache miss runs the RAG backbone. Mirrors
    HGCRAGAgent's fallback architecture so AC-RAG and HGC-RAG cost
    comparisons reflect the gate's contribution rather than a heavier
    fallback inherited from the standalone AnswerCacheAgent.

    Parameters
    ----------
    ac_agent:
        AnswerCacheAgent instance used only as a cache holder; its ReAct
        fallback is bypassed.
    rag_backbone:
        Object exposing ``.run(question, docs)`` (see
        :class:`hgc.rag_backbone.RAGBackbone`).
    """

    def __init__(self, ac_agent: Any, rag_backbone: Any) -> None:
        self._ac = ac_agent
        self._rag = rag_backbone

    def run(self, question: str, docs: list[dict] | None = None) -> dict:
        import dspy

        t0 = time.monotonic()
        lm = dspy.settings.lm
        history_before = _history_len(lm)

        q_emb = self._ac._embedder(question)
        best_sim, best_idx = _best_similarity(q_emb, self._ac._cache)

        if best_sim >= self._ac._sim_threshold:
            cached_answer = self._ac._cache[best_idx][1]
            return {
                "answer": cached_answer,
                "trajectory": {},
                "tokens": _count_tokens_since(lm, history_before),
                "gate_tokens": 0,
                "wall_time": time.monotonic() - t0,
                "n_iters": 0,
                "cache_hit": True,
                "path": "cache_hit",
            }

        rag_result = self._rag.run(question, docs=docs)
        rag_tokens = int(rag_result.get("tokens", 0))
        return {
            **rag_result,
            "tokens": _count_tokens_since(lm, history_before) + rag_tokens,
            "gate_tokens": 0,
            "wall_time": time.monotonic() - t0,
            "cache_hit": False,
            "path": "cache_miss",
        }
