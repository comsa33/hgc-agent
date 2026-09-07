"""Unit tests for hgc.contaminators.typo_mutation."""

from __future__ import annotations

import random
import time
import uuid

import numpy as np

from hgc.contaminators.typo_mutation import (
    _ADJ,
    _mutate,
    corrupt_hint_store_typo,
)
from hgc.memory import HintRecord, HintStore


def _make_hint(hint_type: str, content: str) -> HintRecord:
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
    )


def test_mutate_changes_text_but_keeps_nonempty():
    rng = random.Random(42)
    text = "Queen Arwa University is located in Yemen"
    out = _mutate(text, rng, rate=0.3)
    assert out != text
    assert len(out) >= len(text) - int(len(text) * 0.3) - 1
    assert out.strip() != ""


def test_mutate_empty_string_is_noop():
    assert _mutate("", random.Random(0), rate=0.5) == ""


def test_mutate_deterministic_given_seed():
    rng1 = random.Random(7)
    rng2 = random.Random(7)
    text = "The Roman Empire fell in 476 AD."
    assert _mutate(text, rng1, rate=0.2) == _mutate(text, rng2, rate=0.2)


def test_adj_map_has_all_letters():
    """Every lowercase letter a-z should appear as a dict key so _mutate never
    skips a substitutable character because of a missing adjacency entry.
    """
    missing = {chr(c) for c in range(ord("a"), ord("z") + 1) if chr(c) not in _ADJ}
    # Only the letters we actually defined are expected — report any gaps.
    assert missing == set(), f"adjacency map missing keys: {missing}"


def test_corrupt_hint_store_skips_bare_digit_content(tmp_path):
    store = HintStore(db_path=str(tmp_path / "typo.db"))
    bare = _make_hint("location", "63970")
    store.add(bare)
    for i in range(4):
        store.add(_make_hint("entity", f"Some entity name number {i}"))

    corrupted_ids = corrupt_hint_store_typo(store, seed=42, fraction=0.50, char_rate=0.20)
    assert bare.hint_id not in corrupted_ids
    # Some non-bare hints should have been corrupted.
    assert len(corrupted_ids) > 0


def test_corrupt_hint_store_deterministic(tmp_path):
    import shutil

    src = tmp_path / "src.db"
    a = HintStore(db_path=str(src))
    for i in range(6):
        a.add(_make_hint("entity", f"Sample content {i} with enough characters"))

    dst = tmp_path / "dst.db"
    shutil.copy2(src, dst)
    b = HintStore(db_path=str(dst))

    ids_a = corrupt_hint_store_typo(a, seed=42, fraction=0.5)
    ids_b = corrupt_hint_store_typo(b, seed=42, fraction=0.5)
    assert sorted(ids_a) == sorted(ids_b)
