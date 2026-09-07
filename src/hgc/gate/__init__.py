"""HGC verification gate — three independent components that together decide
whether a cached answer is supported by the current document substrate.

Components
----------
G1 :class:`ScopeFilter`       — drops location hints whose corpus scope does not match.
G2 :class:`DocidCheck`        — drops location hints whose docid is absent from the current doc pool
                                (string docids as well as numeric ones; see :func:`resolve_docid`).
G3 :class:`SupportVerifier`   — LLM yes/no: does this document support this proposed answer.

A :class:`CompositeGate` chains them: the cached answer is accepted only when at
least one surviving location hint passes all three.
"""

from hgc.gate.composite import CompositeGate, GateDecision
from hgc.gate.docid_check import DocidCheck, parse_docid, resolve_docid
from hgc.gate.scope_filter import ScopeFilter
from hgc.gate.support_verifier import SupportVerifier

__all__ = [
    "CompositeGate",
    "GateDecision",
    "ScopeFilter",
    "DocidCheck",
    "SupportVerifier",
    "parse_docid",
    "resolve_docid",
]
