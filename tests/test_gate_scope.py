"""Unit tests for hgc.gate.scope_filter.ScopeFilter (G1)."""

from __future__ import annotations

import time
import uuid

import numpy as np

from hgc.gate import ScopeFilter
from hgc.memory import HintRecord


def _make_hint(
    *,
    hint_type: str,
    content: str = "x",
    scope_id: str | None = None,
) -> HintRecord:
    now = time.time()
    h = HintRecord(
        hint_id=str(uuid.uuid4()),
        hint_type=hint_type,
        polarity="positive",
        content=content,
        content_meta={},
        query_ctx="q",
        query_ctx_embedding=np.array([1.0, 0.0], dtype=np.float32).tobytes(),
        trajectory_step=0,
        created_at=now,
        last_validated_at=now,
        success_count=0,
        failure_count=0,
        retrieval_count=0,
    )
    if scope_id is not None:
        h.scope_id = scope_id  # type: ignore[attr-defined]
    return h


def test_passes_through_when_scope_is_none():
    hints = [
        _make_hint(hint_type="location", scope_id="bcp_query_1"),
        _make_hint(hint_type="entity"),
    ]
    kept = ScopeFilter()(hints, scope_id=None)
    assert len(kept) == 2


def test_drops_location_hint_with_mismatched_scope():
    h1 = _make_hint(hint_type="location", scope_id="bcp_query_1")
    h2 = _make_hint(hint_type="location", scope_id="bcp_query_2")
    kept = ScopeFilter()([h1, h2], scope_id="bcp_query_1")
    assert kept == [h1]


def test_passes_entity_and_strategy_regardless_of_scope():
    h_entity = _make_hint(hint_type="entity")
    h_strategy = _make_hint(hint_type="strategy")
    h_loc_wrong = _make_hint(hint_type="location", scope_id="other")
    kept = ScopeFilter()([h_entity, h_strategy, h_loc_wrong], scope_id="bcp_query_1")
    assert h_entity in kept and h_strategy in kept
    assert h_loc_wrong not in kept


def test_location_hint_without_scope_field_dropped():
    # HintRecord defaults scope_id to "default"; it does not match any real query scope.
    h = _make_hint(hint_type="location", scope_id=None)  # keeps dataclass default "default"
    kept = ScopeFilter()([h], scope_id="bcp_query_1")
    assert kept == []


def test_strict_for_configurable():
    # Turn OFF strictness for location; now it passes through regardless of scope.
    h_loc_wrong = _make_hint(hint_type="location", scope_id="other")
    kept = ScopeFilter(strict_for=set())([h_loc_wrong], scope_id="bcp_query_1")
    assert kept == [h_loc_wrong]
