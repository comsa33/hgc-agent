"""G1: typed-scope filter for hint retrieval.

Location hints are corpus-local (docids are only meaningful inside their
originating document pool), so they must match the current query's scope exactly.
Entity and strategy hints are portable across scopes and pass through unchanged.
"""

from __future__ import annotations

from hgc.memory import HintRecord


class ScopeFilter:
    """Apply strict scope matching to location hints.

    Parameters
    ----------
    strict_for : set of hint_type strings whose membership requires exact
        scope match (default: ``{"location"}``). Other hint types are returned
        as-is regardless of scope.
    """

    def __init__(self, strict_for: set[str] | None = None) -> None:
        self._strict_for = strict_for if strict_for is not None else {"location"}

    def __call__(self, hints: list[HintRecord], scope_id: str | None) -> list[HintRecord]:
        if scope_id is None:
            return list(hints)
        kept: list[HintRecord] = []
        for h in hints:
            if h.hint_type in self._strict_for:
                hint_scope = getattr(h, "scope_id", None)
                if hint_scope is None or hint_scope == scope_id:
                    kept.append(h)
            else:
                kept.append(h)
        return kept
