"""HGCAgent — full Hint-Gated Cache pattern on the DSPy ReAct backbone.

Composes an AnswerCacheAgent (memory), a CompositeGate (verification), and a
HGCCoreAgent (fallback). The constructor accepts any callable shaped like
:class:`hgc.gate.CompositeGate`, so A1 ablation variants (G1-only, G1+G2,
etc.) plug in without modifying this class.

Flow on ``.run(question, docs=None)``:
  1. Delegate AC lookup.
  2. If AC miss → run HGCCoreAgent. ``path='cache_miss'``.
  3. If AC hit → retrieve top-k hints from the hint store (scope-agnostic
     here — the ScopeFilter inside the gate handles scope semantics).
     - If no location hints are retrievable → trust AC, ``path='cache_hit_no_hint'``
       (or, under the opt-in ``strict_no_hint`` variant, reroute
       through the core agent instead, ``path='cache_hit_no_hint_strict'``).
     - Otherwise run the gate. If the gate passes → return AC,
       ``path='cache_verified'``. If the gate rejects → run the core agent,
       ``path='cache_fallback'``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from hgc.gate import CompositeGate, GateDecision
from hgc.hgc_core import _count_tokens_since, _history_len
from hgc.memory import HintRecord


class HGCAgent:
    """Hint-Gated Cache agent wrapping AnswerCache + Gate + HGCCoreAgent.

    Parameters
    ----------
    ac_agent:
        AnswerCacheAgent instance (memory layer).
    core_agent:
        HGCCoreAgent instance (fallback). Attributes ``_alpha``, ``_beta``,
        ``_gamma``, ``_theta_pos``, ``_theta_neg``, ``_scope_id`` are read
        to parameterise hint retrieval so the hybrid uses the same axis
        tuning as the standalone core agent.
    store:
        HintStore from which the gate's candidate location hints are pulled.
    embedder:
        Callable ``str -> np.ndarray`` used to embed the question.
    gate:
        CompositeGate (or compatible callable). Default: ``CompositeGate()``.
    top_k_hints:
        Number of candidate location hints retrieved per query (before the
        gate filters them). Default 3.
    """

    def __init__(
        self,
        ac_agent: Any,
        core_agent: Any,
        store: Any,
        embedder: Callable[[str], np.ndarray],
        gate: CompositeGate | None = None,
        top_k_hints: int = 3,
        strict_no_hint: bool = False,
    ) -> None:
        self._ac = ac_agent
        self._core = core_agent
        self._store = store
        self._embedder = embedder
        self._gate = gate if gate is not None else CompositeGate()
        self._top_k = top_k_hints
        self._strict_no_hint = strict_no_hint

    def _retrieve_location_hints(self, query_emb: np.ndarray) -> list[HintRecord]:
        hints = self._store.search(
            query_embedding=query_emb,
            k=self._top_k * 3,
            alpha=self._core._alpha,
            beta=self._core._beta,
            gamma=self._core._gamma,
            theta_pos=self._core._theta_pos,
            theta_neg=self._core._theta_neg,
            scope_id=self._core._scope_id,
        )
        return [h for h in hints if h.hint_type == "location"][: self._top_k]

    def run(self, question: str, docs: list[dict] | None = None) -> dict:
        import dspy

        t0 = time.monotonic()
        lm = dspy.settings.lm
        history_before = _history_len(lm)

        ac_result = self._ac.run(question)
        pre_gate_history = _history_len(lm)

        if not bool(ac_result.get("cache_hit", False)):
            core_result = self._core.run(question, docs=docs)
            return {
                **core_result,
                "tokens": _count_tokens_since(lm, history_before),
                "gate_tokens": 0,
                "wall_time": time.monotonic() - t0,
                "cache_hit": False,
                "path": "cache_miss",
            }

        ac_answer: str = ac_result.get("answer", "")
        query_emb = self._embedder(question)
        location_hints = self._retrieve_location_hints(query_emb)

        if not location_hints:
            if self._strict_no_hint:
                # Strict-fallback variant (previous-round ablation): an
                # uncovered cache hit is not trusted; reroute through the core
                # agent instead of returning the cached answer. Default policy
                # (strict_no_hint=False) is unchanged.
                core_result = self._core.run(question, docs=docs)
                return {
                    **core_result,
                    "tokens": _count_tokens_since(lm, history_before),
                    "gate_tokens": 0,
                    "wall_time": time.monotonic() - t0,
                    "cache_hit": False,
                    "path": "cache_hit_no_hint_strict",
                }
            return {
                **ac_result,
                "wall_time": time.monotonic() - t0,
                "cache_hit": True,
                "tokens": _count_tokens_since(lm, history_before),
                "gate_tokens": 0,
                "n_iters": 0,
                "trajectory": {},
                "path": "cache_hit_no_hint",
            }

        decision: GateDecision = self._gate(
            question=question,
            answer=ac_answer,
            hints=location_hints,
            docs=docs,
            scope_id=self._core._scope_id,
        )
        gate_tokens = _count_tokens_since(lm, pre_gate_history)

        if decision.passed:
            return {
                **ac_result,
                "wall_time": time.monotonic() - t0,
                "cache_hit": True,
                "tokens": _count_tokens_since(lm, history_before),
                "gate_tokens": gate_tokens,
                "n_iters": 0,
                "trajectory": {},
                "path": "cache_verified",
                "gate_evidence": getattr(decision.evidence_hint, "hint_id", None),
                "gate_stage_counts": decision.stage_counts,
            }

        core_result = self._core.run(question, docs=docs)
        return {
            **core_result,
            "tokens": _count_tokens_since(lm, history_before),
            "gate_tokens": gate_tokens,
            "wall_time": time.monotonic() - t0,
            "cache_hit": False,
            "path": "cache_fallback",
            "gate_reject_reason": decision.reason,
            "gate_stage_counts": decision.stage_counts,
        }
