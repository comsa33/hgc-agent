"""Unit tests for hgc.contaminators.entity_swap.

Covers HintStore entity-swap and AnswerCache entity-swap. No network / LM calls.
"""

from __future__ import annotations

import time
import uuid

import numpy as np

from hgc.contaminators.entity_swap import (
    _CAPWORD_RE,
    _FALLBACK_ENTITIES,
    _swap_first_entity,
    corrupt_hint_store_entity_swap,
)
from hgc.memory import HintRecord, HintStore


def _make_hint(hint_type: str, content: str, scope_id: str = "default") -> HintRecord:
    now = time.time()
    return HintRecord(
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
        scope_id=scope_id,
    )


def test_swap_first_entity_replaces_capitalised_run():
    import random

    rng = random.Random(0)
    out = _swap_first_entity("Queen Arwa University is located", rng, _FALLBACK_ENTITIES)
    assert out is not None
    assert "Queen Arwa University" not in out
    assert any(e in out for e in _FALLBACK_ENTITIES)


def test_swap_first_entity_returns_none_without_capitalised_run():
    import random

    rng = random.Random(0)
    assert _swap_first_entity("lowercase text with no caps", rng, _FALLBACK_ENTITIES) is None


def test_corrupt_hint_store_swaps_capitalised_runs(tmp_path):
    store = HintStore(db_path=str(tmp_path / "entity.db"))
    hints = [
        _make_hint("entity", "Queen Arwa University"),
        _make_hint("entity", "University of Bologna"),
        _make_hint("entity", "Harvard University"),
        _make_hint("entity", "Stanford University"),
        _make_hint("entity", "MIT"),  # single word but still matches _CAPWORD_RE
    ]
    for h in hints:
        store.add(h)

    before = {h.hint_id: h.content for h in hints}
    corrupted_ids = corrupt_hint_store_entity_swap(store, seed=42, fraction=0.40)

    assert len(corrupted_ids) > 0
    for hid in corrupted_ids:
        after_content = next(h.content for h in store.all() if h.hint_id == hid)
        assert after_content != before[hid]
        # Corrupted content must still contain some capitalised run (a swappable entity)
        assert _CAPWORD_RE.search(after_content) is not None


def test_corrupt_hint_store_determinism(tmp_path):
    import shutil

    src = tmp_path / "src.db"
    store_a = HintStore(db_path=str(src))
    for i in range(6):
        store_a.add(_make_hint("entity", f"Test Entity {chr(ord('A') + i)}"))

    # Clone DB to a sibling path so the two runs operate on identical state.
    dst = tmp_path / "dst.db"
    shutil.copy2(src, dst)
    store_b = HintStore(db_path=str(dst))

    ids_a = corrupt_hint_store_entity_swap(store_a, seed=42, fraction=0.5)
    ids_b = corrupt_hint_store_entity_swap(store_b, seed=42, fraction=0.5)
    assert sorted(ids_a) == sorted(ids_b)


def test_corrupt_hint_store_skips_unswappable_content(tmp_path):
    """Bare digit content has no capitalised run -> not corrupted."""
    store = HintStore(db_path=str(tmp_path / "entity2.db"))
    bare = _make_hint("location", "63970")
    store.add(bare)
    for i in range(4):
        store.add(_make_hint("entity", f"Swappable Entity {chr(ord('A') + i)}"))

    corrupted_ids = corrupt_hint_store_entity_swap(store, seed=42, fraction=0.40)
    # Bare-digit hint must not be among the corrupted.
    assert bare.hint_id not in corrupted_ids
