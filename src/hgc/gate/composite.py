"""Composite gate: chain G1 → G2 → G3 and report which stage a cache hit
accepted or failed at.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from hgc.gate.docid_check import DocidCheck, resolve_docid
from hgc.gate.scope_filter import ScopeFilter
from hgc.gate.support_verifier import SupportVerifier
from hgc.memory import HintRecord


class _VerifierCallable(Protocol):
    def __call__(self, question: str, answer: str, doc_text: str) -> bool: ...


@dataclass
class GateDecision:
    """Structured result of one CompositeGate invocation."""

    passed: bool
    reason: str
    evidence_hint: HintRecord | None = None
    # Per-stage drop census for the candidate hints this invocation examined:
    # g1_scope / g2_docid / g3_doc_unresolved / g3_llm. On an accept it counts
    # the hints dropped before the accepting one. Without it every rejection
    # collapses into "no_hint_passed_all_components" and the ablation tables
    # cannot say which component actually did the work.
    stage_counts: dict | None = None


class CompositeGate:
    """Chain ScopeFilter → DocidCheck → SupportVerifier.

    The gate returns :class:`GateDecision` with ``passed=True`` as soon as any
    candidate location hint survives all three components. If no candidate
    survives, ``passed=False`` and ``reason`` identifies the stage that
    dropped the last surviving hint.

    Parameters
    ----------
    scope_filter:
        G1 instance. If None a default ``ScopeFilter()`` is used.
    docid_check:
        G2 instance. If None a default ``DocidCheck()`` is used.
    support_verifier:
        G3 instance (must be callable ``(question, answer, doc_text) -> bool``).
        If None a new ``SupportVerifier()`` is built; pass a stub in tests.
    enabled:
        Subset of ``{"g1","g2","g3"}`` indicating which components to run.
        Components not in this set always pass. Used for ablation experiments.
    """

    def __init__(
        self,
        scope_filter: ScopeFilter | None = None,
        docid_check: DocidCheck | None = None,
        support_verifier: _VerifierCallable | Callable | None = None,
        enabled: set[str] | None = None,
    ) -> None:
        self._scope = scope_filter or ScopeFilter()
        self._docid = docid_check or DocidCheck()
        self._support = support_verifier or SupportVerifier()
        self._enabled = enabled if enabled is not None else {"g1", "g2", "g3"}

    def __call__(
        self,
        question: str,
        answer: str,
        hints: list[HintRecord],
        docs: list[dict] | None,
        scope_id: str | None,
    ) -> GateDecision:
        doc_map: dict[str, str] = {}
        if docs:
            for d in docs:
                doc_map[str(d["docid"])] = d.get("text", "")

        counts = {"g1_scope": 0, "g2_docid": 0, "g3_doc_unresolved": 0, "g3_llm": 0}

        # G1 scope filter
        candidates = self._scope(hints, scope_id) if "g1" in self._enabled else list(hints)
        counts["g1_scope"] = len(hints) - len(candidates)
        location_hints = [h for h in candidates if h.hint_type == "location"]
        if not location_hints:
            return GateDecision(
                passed=False, reason="no_location_hint_in_scope", stage_counts=counts
            )

        # G2 docid check + G3 support verifier, iterated per hint
        for hint in location_hints:
            if "g2" in self._enabled and not self._docid(hint, doc_map):
                counts["g2_docid"] += 1
                continue
            # G2 passed (or disabled). Need doc text to hand to G3.
            docid = resolve_docid(hint.content, doc_map)
            doc_text = doc_map.get(docid or "", "")
            if "g3" in self._enabled:
                if not doc_text:
                    counts["g3_doc_unresolved"] += 1
                    continue
                if not self._support(question, answer, doc_text):
                    counts["g3_llm"] += 1
                    continue
            return GateDecision(
                passed=True, reason="verified", evidence_hint=hint, stage_counts=counts
            )

        return GateDecision(
            passed=False, reason="no_hint_passed_all_components", stage_counts=counts
        )
