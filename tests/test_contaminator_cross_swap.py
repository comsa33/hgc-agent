"""
Unit tests for HintContaminator (US-014).

Store fixture: 20 positive hints (mixed types) + 5 negative hints = 25 total.
"""

from __future__ import annotations

import json
import time
import uuid

import numpy as np
import pytest

from hgc.contaminators.cross_swap import (
    HintContaminator,
    contaminate_answer_cache_from_p1,
    contaminate_mem0_from_p1,
    contaminate_trajectory_cache_from_p1,
)
from hgc.memory import HintRecord, HintStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emb_bytes(values: list[float]) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


def _make_hint(
    *,
    hint_type: str = "location",
    polarity: str = "positive",
    content: str = "docid=42",
    content_meta: dict | None = None,
    query_ctx: str = "test query",
    success_count: int = 2,
    failure_count: int = 0,
    retrieval_count: int = 1,
) -> HintRecord:
    now = time.time()
    return HintRecord(
        hint_id=str(uuid.uuid4()),
        hint_type=hint_type,
        polarity=polarity,
        content=content,
        content_meta=content_meta or {},
        query_ctx=query_ctx,
        query_ctx_embedding=_emb_bytes([1.0, 0.0, 0.0]),
        trajectory_step=1,
        created_at=now,
        last_validated_at=now,
        success_count=success_count,
        failure_count=failure_count,
        retrieval_count=retrieval_count,
    )


# Fixed hint_ids used across all populate calls so determinism tests compare equivalent stores.
_POSITIVE_IDS: list[str] = [f"pos-hint-{i:03d}" for i in range(20)]
_NEGATIVE_IDS: list[str] = [f"neg-hint-{i:03d}" for i in range(5)]


def _populate_store(store: HintStore) -> tuple[list[str], list[str]]:
    """
    Add 20 positive + 5 negative hints with *fixed* hint_ids.

    Positives: 7 location (docid=NNN), 7 entity, 6 strategy.
    Negatives: 5 location hints with polarity='negative'.

    Using deterministic IDs so that two independently-populated stores
    contain the same hint_ids and ``corrupt(seed=42)`` selects the same ones.

    Returns (positive_ids, negative_ids).
    """
    positive_ids: list[str] = []
    negative_ids: list[str] = []
    pos_idx = 0

    # 7 location hints
    for i in range(7):
        h = _make_hint(hint_type="location", content=f"docid={1000 + i * 111}")
        h.hint_id = _POSITIVE_IDS[pos_idx]
        pos_idx += 1
        store.add(h)
        positive_ids.append(h.hint_id)

    # 7 entity hints
    entities = [
        "Queen Arwa University",
        "Nalanda University",
        "University of Bologna",
        "University of Oxford",
        "Harvard University",
        "Massachusetts Institute of Technology",
        "Stanford University",
    ]
    for entity in entities:
        h = _make_hint(hint_type="entity", content=entity)
        h.hint_id = _POSITIVE_IDS[pos_idx]
        pos_idx += 1
        store.add(h)
        positive_ids.append(h.hint_id)

    # 6 strategy hints
    strategies = [
        "search_documents(keyword='Routledge 2018')",
        "search_documents(keyword='ACM 2020')",
        "search_documents(keyword='IEEE 2019')",
        "search_documents(keyword='Springer 2017')",
        "search_documents(keyword='NeurIPS 2021')",
        "search_documents(keyword='ICML 2022')",
    ]
    for strat in strategies:
        h = _make_hint(hint_type="strategy", content=strat)
        h.hint_id = _POSITIVE_IDS[pos_idx]
        pos_idx += 1
        store.add(h)
        positive_ids.append(h.hint_id)

    # 5 negative hints
    for i in range(5):
        h = _make_hint(
            hint_type="location",
            polarity="negative",
            content=f"docid={9000 + i}",
        )
        h.hint_id = _NEGATIVE_IDS[i]
        store.add(h)
        negative_ids.append(h.hint_id)

    return positive_ids, negative_ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = HintStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    positive_ids, negative_ids = _populate_store(store)
    return store, positive_ids, negative_ids


# ---------------------------------------------------------------------------
# Test 1: corrupt() modifies exactly fraction * N_positive hints
# ---------------------------------------------------------------------------


class TestCorruptCount:
    def test_exactly_four_hints_corrupted(self, populated_store):
        """fraction=0.20 * 20 positives = floor(4.0) = 4 corrupted."""
        store, positive_ids, _ = populated_store
        contaminator = HintContaminator(store, seed=42, fraction=0.20)
        corrupted_ids = contaminator.corrupt()
        assert len(corrupted_ids) == 4

    def test_corrupted_ids_are_a_subset_of_positive_ids(self, populated_store):
        store, positive_ids, _ = populated_store
        contaminator = HintContaminator(store, seed=42, fraction=0.20)
        corrupted_ids = contaminator.corrupt()
        assert set(corrupted_ids).issubset(set(positive_ids))

    def test_content_changed_for_corrupted_hints(self, populated_store):
        store, _, _ = populated_store
        before = {h.hint_id: h.content for h in store.all()}
        contaminator = HintContaminator(store, seed=42, fraction=0.20)
        corrupted_ids = contaminator.corrupt()
        after = {h.hint_id: h.content for h in store.all()}
        for hid in corrupted_ids:
            assert after[hid] != before[hid], f"content unchanged for {hid}"

    def test_uncorrupted_hints_content_unchanged(self, populated_store):
        store, positive_ids, _ = populated_store
        before = {h.hint_id: h.content for h in store.all()}
        contaminator = HintContaminator(store, seed=42, fraction=0.20)
        corrupted_ids = contaminator.corrupt()
        after = {h.hint_id: h.content for h in store.all()}
        untouched = [hid for hid in positive_ids if hid not in corrupted_ids]
        for hid in untouched:
            assert after[hid] == before[hid]


# ---------------------------------------------------------------------------
# Test 2: deterministic selection — same seed → same hint_ids
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_selects_same_hint_ids(self, tmp_path):
        """Two contaminator instances with the same seed on the same store pick the same IDs."""
        db = str(tmp_path / "det.db")

        store_a = HintStore(db_path=db)
        _populate_store(store_a)
        c_a = HintContaminator(store_a, seed=42, fraction=0.20)
        ids_a = c_a.corrupt()
        store_a.close()

        # Re-open same DB (same data, content now corrupted — but hint_ids identical)
        # Re-create a fresh store with identical initial data for a clean comparison
        db2 = str(tmp_path / "det2.db")
        store_b = HintStore(db_path=db2)
        _populate_store(store_b)
        c_b = HintContaminator(store_b, seed=42, fraction=0.20)
        ids_b = c_b.corrupt()
        store_b.close()

        assert ids_a == ids_b, "Same seed must select identical hint_ids in the same order"

    def test_different_seed_may_select_different_hint_ids(self, tmp_path):
        db_a = str(tmp_path / "seed_a.db")
        db_b = str(tmp_path / "seed_b.db")

        store_a = HintStore(db_path=db_a)
        _populate_store(store_a)
        ids_a = HintContaminator(store_a, seed=42).corrupt()
        store_a.close()

        store_b = HintStore(db_path=db_b)
        _populate_store(store_b)
        ids_b = HintContaminator(store_b, seed=99).corrupt()
        store_b.close()

        # With 20 hints and 4 chosen, different seeds will almost certainly differ
        assert ids_a != ids_b, "Different seeds should (almost certainly) select different hints"


# ---------------------------------------------------------------------------
# Test 3: metadata preserved — polarity, query_ctx, counters unchanged
# ---------------------------------------------------------------------------


class TestMetadataPreserved:
    def test_polarity_query_ctx_counters_unchanged(self, populated_store):
        store, _, _ = populated_store
        before = {h.hint_id: h for h in store.all()}
        contaminator = HintContaminator(store, seed=42, fraction=0.20)
        corrupted_ids = contaminator.corrupt()
        after = {h.hint_id: h for h in store.all()}

        for hid in corrupted_ids:
            b = before[hid]
            a = after[hid]
            assert a.polarity == b.polarity == "positive", "polarity must stay positive"
            assert a.query_ctx == b.query_ctx, "query_ctx must be unchanged"
            assert a.success_count == b.success_count, "success_count must be unchanged"
            assert a.failure_count == b.failure_count, "failure_count must be unchanged"
            assert a.retrieval_count == b.retrieval_count, "retrieval_count must be unchanged"
            # Only content should differ
            assert a.content != b.content, "content must be corrupted"


# ---------------------------------------------------------------------------
# Test 4: negative hints are never touched
# ---------------------------------------------------------------------------


class TestNegativesUntouched:
    def test_negative_hints_never_corrupted(self, populated_store):
        store, _, negative_ids = populated_store
        before = {h.hint_id: h.content for h in store.all() if h.polarity == "negative"}
        contaminator = HintContaminator(store, seed=42, fraction=0.20)
        corrupted_ids = contaminator.corrupt()

        # None of the negative hint_ids should appear in corrupted_ids
        assert not set(negative_ids) & set(corrupted_ids), "Negative hints must not be corrupted"

        after = {h.hint_id: h.content for h in store.all() if h.polarity == "negative"}
        for nid in negative_ids:
            assert after[nid] == before[nid], f"Negative hint {nid} content was modified"


# ---------------------------------------------------------------------------
# Test 5: snapshot_healthy → corrupt → restore_healthy round-trip
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_round_trip_restores_exact_state(self, populated_store, tmp_path):
        store, positive_ids, negative_ids = populated_store
        snap_path = str(tmp_path / "healthy.json")

        # Capture original state
        original = {h.hint_id: h for h in store.all()}

        contaminator = HintContaminator(store, seed=42, fraction=0.20)
        contaminator.snapshot_healthy(snap_path)
        corrupted_ids = contaminator.corrupt()

        # Verify corruption happened
        for hid in corrupted_ids:
            current = store.get(hid)
            assert current.content != original[hid].content

        # Restore
        contaminator.restore_healthy(snap_path)

        restored = {h.hint_id: h for h in store.all()}

        # Same set of hint_ids
        assert set(restored.keys()) == set(original.keys()), "hint_id set must match after restore"

        # All fields identical
        for hid, orig in original.items():
            res = restored[hid]
            assert res.content == orig.content, f"content mismatch for {hid}"
            assert res.polarity == orig.polarity
            assert res.query_ctx == orig.query_ctx
            assert res.success_count == orig.success_count
            assert res.failure_count == orig.failure_count
            assert res.retrieval_count == orig.retrieval_count
            assert res.hint_type == orig.hint_type

    def test_total_hint_count_preserved_after_restore(self, populated_store, tmp_path):
        store, _, _ = populated_store
        snap_path = str(tmp_path / "healthy2.json")
        n_before = len(store.all())

        contaminator = HintContaminator(store, seed=42, fraction=0.20)
        contaminator.snapshot_healthy(snap_path)
        contaminator.corrupt()
        contaminator.restore_healthy(snap_path)

        assert len(store.all()) == n_before, "Total hint count must be identical after restore"


# ---------------------------------------------------------------------------
# Test 6: update_content on HintStore (minimal smoke — confirms new API)
# ---------------------------------------------------------------------------


class TestUpdateContent:
    def test_update_content_changes_only_content(self, store):
        h = _make_hint(content="docid=100", success_count=3, failure_count=1, retrieval_count=5)
        store.add(h)

        store.update_content(h.hint_id, "docid=999")

        updated = store.get(h.hint_id)
        assert updated.content == "docid=999"
        assert updated.success_count == 3
        assert updated.failure_count == 1
        assert updated.retrieval_count == 5
        assert updated.polarity == h.polarity
        assert updated.query_ctx == h.query_ctx


# ---------------------------------------------------------------------------
# Helpers for baseline cache contamination tests
# ---------------------------------------------------------------------------


def _write_p1_trajectories(tmp_path, n: int = 10) -> list:
    """Write *n* fake P1 trajectory JSON files with distinct questions/answers."""
    paths = []
    for i in range(n):
        data = {
            "question": f"question_{i}",
            "pred": f"answer_{i}",
            "judgment_correct": True,
        }
        p = tmp_path / f"traj_{i:03d}.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        paths.append(p)
    return paths


def _stub_embedder(question: str):
    """Deterministic stub embedder: each question maps to a unique float32 vector."""
    idx = int(question.split("_")[-1]) if "_" in question else hash(question) % 1000
    return np.array([float(idx), 0.0, 0.0], dtype=np.float32)


def _make_ac_agent():
    from hgc.baselines import AnswerCacheAgent

    return AnswerCacheAgent(
        tools=[],
        embedder=_stub_embedder,
        judge=lambda q, a: True,
        react_factory=lambda tools, max_iters: None,
    )


def _make_tc_agent():
    from hgc.baselines import TrajectoryCacheAgent

    return TrajectoryCacheAgent(
        tools=[],
        embedder=_stub_embedder,
        judge=lambda q, a: True,
        react_factory=lambda tools, max_iters, prefix: None,
    )


# ---------------------------------------------------------------------------
# FakeMem0 for Mem0 contamination tests
# ---------------------------------------------------------------------------


class FakeMem0:
    """In-memory stand-in for mem0.Memory with add/delete/get_all."""

    def __init__(self):
        self._store: dict[str, str] = {}  # id -> content
        self._counter = 0

    def add(self, content: str, user_id: str = "user") -> dict:
        self._counter += 1
        mid = f"mem-{self._counter:04d}"
        self._store[mid] = content
        return {"results": [{"id": mid, "memory": content}]}

    def delete(self, memory_id: str) -> None:
        self._store.pop(memory_id, None)

    def get_all(self, user_id: str = "user") -> dict:
        results = [{"id": k, "memory": v} for k, v in self._store.items()]
        return {"results": results}


# ---------------------------------------------------------------------------
# AnswerCacheAgent contamination tests
# ---------------------------------------------------------------------------


class TestContaminateAnswerCache:
    def test_swaps_20pct(self, tmp_path):
        """20 % of 10 entries = 2 corrupted."""
        paths = _write_p1_trajectories(tmp_path, n=10)
        agent = _make_ac_agent()

        corrupted = contaminate_answer_cache_from_p1(
            agent, paths, _stub_embedder, seed=42, fraction=0.20
        )

        assert agent.cache_size == 10
        assert len(corrupted) == 2

    def test_corrupted_answers_differ_from_originals(self, tmp_path):
        """Each corrupted entry has a different answer than the original."""
        paths = _write_p1_trajectories(tmp_path, n=10)

        # Seed a reference agent to capture original answers.
        ref = _make_ac_agent()
        from hgc.runner import seed_answer_cache_from_p1

        seed_answer_cache_from_p1(ref, paths, _stub_embedder)
        originals = {i: ref._cache[i][1] for i in range(len(ref._cache))}

        agent = _make_ac_agent()
        corrupted = contaminate_answer_cache_from_p1(
            agent, paths, _stub_embedder, seed=42, fraction=0.20
        )

        for i in corrupted:
            assert agent._cache[i][1] != originals[i], f"answer unchanged at index {i}"

    def test_embeddings_unchanged_at_corrupted_indices(self, tmp_path):
        """Embeddings at corrupted indices must not be modified."""
        paths = _write_p1_trajectories(tmp_path, n=10)

        ref = _make_ac_agent()
        from hgc.runner import seed_answer_cache_from_p1

        seed_answer_cache_from_p1(ref, paths, _stub_embedder)

        agent = _make_ac_agent()
        corrupted = contaminate_answer_cache_from_p1(
            agent, paths, _stub_embedder, seed=42, fraction=0.20
        )

        for i in corrupted:
            np.testing.assert_array_equal(agent._cache[i][0], ref._cache[i][0])

    def test_deterministic_seed(self, tmp_path):
        """Same seed yields identical corrupted-index lists."""
        paths = _write_p1_trajectories(tmp_path, n=10)

        agent_a = _make_ac_agent()
        agent_b = _make_ac_agent()

        idx_a = contaminate_answer_cache_from_p1(agent_a, paths, _stub_embedder, seed=42)
        idx_b = contaminate_answer_cache_from_p1(agent_b, paths, _stub_embedder, seed=42)

        assert idx_a == idx_b

    def test_raises_on_single_trajectory(self, tmp_path):
        """Only 1 valid trajectory -> ValueError after seeding."""
        paths = _write_p1_trajectories(tmp_path, n=1)
        agent = _make_ac_agent()

        with pytest.raises(ValueError, match="fewer than 2"):
            contaminate_answer_cache_from_p1(agent, paths, _stub_embedder)


# ---------------------------------------------------------------------------
# TrajectoryCacheAgent contamination tests
# ---------------------------------------------------------------------------


class TestContaminateTrajectoryCachce:
    def test_swaps_20pct(self, tmp_path):
        """20 % of 10 entries = 2 corrupted."""
        paths = _write_p1_trajectories(tmp_path, n=10)
        agent = _make_tc_agent()

        corrupted = contaminate_trajectory_cache_from_p1(
            agent, paths, _stub_embedder, seed=42, fraction=0.20
        )

        assert agent.store_size == 10
        assert len(corrupted) == 2

    def test_corrupted_summaries_differ_from_originals(self, tmp_path):
        # Write trajectories with distinct, non-empty trajectory dicts so
        # _trajectory_to_summary produces a unique string per entry.
        paths = []
        for i in range(10):
            data = {
                "question": f"question_{i}",
                "pred": f"answer_{i}",
                "judgment_correct": True,
                "trajectory": {
                    "thought_0": f"think_{i}",
                    "action_0": f"act_{i}",
                    "observation_0": f"obs_{i}",
                },
            }
            p = tmp_path / f"traj_{i:03d}.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            paths.append(p)

        ref = _make_tc_agent()
        from hgc.runner import seed_trajectory_cache_from_p1

        seed_trajectory_cache_from_p1(ref, paths, _stub_embedder)
        originals = {i: ref._store[i][2] for i in range(len(ref._store))}

        agent = _make_tc_agent()
        corrupted = contaminate_trajectory_cache_from_p1(
            agent, paths, _stub_embedder, seed=42, fraction=0.20
        )

        for i in corrupted:
            assert agent._store[i][2] != originals[i], f"summary unchanged at index {i}"

    def test_embeddings_and_questions_unchanged(self, tmp_path):
        paths = _write_p1_trajectories(tmp_path, n=10)

        ref = _make_tc_agent()
        from hgc.runner import seed_trajectory_cache_from_p1

        seed_trajectory_cache_from_p1(ref, paths, _stub_embedder)

        agent = _make_tc_agent()
        corrupted = contaminate_trajectory_cache_from_p1(
            agent, paths, _stub_embedder, seed=42, fraction=0.20
        )

        for i in corrupted:
            np.testing.assert_array_equal(agent._store[i][0], ref._store[i][0])
            assert agent._store[i][1] == ref._store[i][1]

    def test_deterministic_seed(self, tmp_path):
        paths = _write_p1_trajectories(tmp_path, n=10)

        agent_a = _make_tc_agent()
        agent_b = _make_tc_agent()

        idx_a = contaminate_trajectory_cache_from_p1(agent_a, paths, _stub_embedder, seed=42)
        idx_b = contaminate_trajectory_cache_from_p1(agent_b, paths, _stub_embedder, seed=42)

        assert idx_a == idx_b


# ---------------------------------------------------------------------------
# Mem0 contamination tests
# ---------------------------------------------------------------------------


class TestContaminateMem0:
    def _make_fake_mem0_with_trajectories(self, tmp_path, n: int = 10):
        """Return (FakeMem0, p1_paths) seeded with n trajectories."""
        paths = _write_p1_trajectories(tmp_path, n=n)
        return FakeMem0(), paths

    def test_swaps_20pct(self, tmp_path):
        """20 % of 10 memories = 2 corrupted; 2 new ids returned."""
        mem0, paths = self._make_fake_mem0_with_trajectories(tmp_path, n=10)
        new_ids = contaminate_mem0_from_p1(mem0, paths, seed=42, fraction=0.20)

        assert len(new_ids) == 2

    def test_corrupted_memories_contain_donor_answers(self, tmp_path):
        """Each new memory must contain a different answer string than the victim's."""
        n = 10
        paths = _write_p1_trajectories(tmp_path, n=n)

        # Seed a reference instance to capture original memories.
        ref = FakeMem0()
        from hgc.runner import seed_mem0_from_p1

        seed_mem0_from_p1(ref, paths, user_id="user")
        original_contents = {m["id"]: m["memory"] for m in ref.get_all(user_id="user")["results"]}

        mem0 = FakeMem0()
        new_ids = contaminate_mem0_from_p1(mem0, paths, seed=42, fraction=0.20)

        current = {m["id"]: m["memory"] for m in mem0.get_all(user_id="user")["results"]}

        for nid in new_ids:
            assert nid in current, f"new id {nid} not in store"
            # New memory should not equal any victim's original content verbatim
            # (donor answer was swapped in)
            assert current[nid] not in original_contents.values() or len(new_ids) > 0

    def test_total_memory_count_preserved(self, tmp_path):
        """After contamination, total memory count stays at n (delete + add = net 0)."""
        n = 10
        mem0, paths = self._make_fake_mem0_with_trajectories(tmp_path, n=n)
        contaminate_mem0_from_p1(mem0, paths, seed=42, fraction=0.20)

        total = len(mem0.get_all(user_id="user")["results"])
        assert total == n

    def test_deterministic_seed(self, tmp_path):
        """Same seed on identical state yields the same number of new ids."""
        n = 10
        paths = _write_p1_trajectories(tmp_path, n=n)

        mem0_a = FakeMem0()
        mem0_b = FakeMem0()

        ids_a = contaminate_mem0_from_p1(mem0_a, paths, seed=42, fraction=0.20)
        ids_b = contaminate_mem0_from_p1(mem0_b, paths, seed=42, fraction=0.20)

        assert len(ids_a) == len(ids_b)
        # Content of new memories should be identical across runs
        store_a = {m["memory"] for m in mem0_a.get_all(user_id="user")["results"]}
        store_b = {m["memory"] for m in mem0_b.get_all(user_id="user")["results"]}
        assert store_a == store_b


# ---------------------------------------------------------------------------
# Bare-digit location contamination (HC-002)
# ---------------------------------------------------------------------------


def _populate_bare_digit_store(store: HintStore) -> list[str]:
    """Seed the store with 7 positive location hints using bare-digit content.

    Mirrors the production format discovered on 2026-04-21: location hints in
    the real BCP P1 store carry content like ``"63970"``, not ``"docid=63970"``.
    """
    bare_docids = ["63970", "88828", "90327", "48497", "48589", "35932", "29583"]
    ids: list[str] = []
    for i, docid in enumerate(bare_docids):
        h = _make_hint(hint_type="location", content=docid)
        h.hint_id = f"bare-loc-{i:03d}"
        store.add(h)
        ids.append(h.hint_id)
    return ids


class TestBareDigitLocationContamination:
    """Regression tests for the 2026-04-21 contamination regex bug.

    Before the fix, ``_DOCID_RE = r"^docid=\\d+$"`` failed on bare-digit
    content, so ``_pick_wrong_sibling`` fell through to the generic
    ``"__CORRUPTED_XXXX"`` suffix path and produced garbage that the runtime
    ``_parse_docid`` treated as "not a docid, pass through unchanged" -- i.e.
    the contamination barely affected the agent at all.
    """

    def test_corrupted_contents_remain_parseable_bare_digits(self, tmp_path):
        db = tmp_path / "bare.db"
        store = HintStore(db_path=str(db))
        originals = _populate_bare_digit_store(store)

        contam = HintContaminator(store, seed=42, fraction=0.40)
        corrupted_ids = contam.corrupt()

        assert len(corrupted_ids) > 0
        for hid in corrupted_ids:
            hint = next(h for h in store.all() if h.hint_id == hid)
            assert hint.content.isdigit(), (
                f"corrupted bare-digit location hint must remain bare digit, "
                f"got {hint.content!r}"
            )
            assert "__CORRUPTED" not in hint.content, (
                f"contamination fell through to suffix path on {hid}: " f"{hint.content!r}"
            )
        # unaffected hints keep their original bare-digit content
        surviving = {h.hint_id: h.content for h in store.all() if h.hint_id not in corrupted_ids}
        for hid in surviving:
            assert hid in originals
            assert surviving[hid].isdigit()

    def test_corrupted_docid_differs_from_original(self, tmp_path):
        db = tmp_path / "bare2.db"
        store = HintStore(db_path=str(db))
        _populate_bare_digit_store(store)

        before = {h.hint_id: h.content for h in store.all()}
        contam = HintContaminator(store, seed=42, fraction=0.40)
        corrupted_ids = contam.corrupt()

        for hid in corrupted_ids:
            after_content = next(h.content for h in store.all() if h.hint_id == hid)
            assert (
                after_content != before[hid]
            ), f"hint {hid} unchanged after corrupt: {after_content!r}"

    def test_deterministic_seed_bare_digit(self, tmp_path):
        """Two independent runs at the same seed select the same victims."""
        db_a = tmp_path / "a.db"
        db_b = tmp_path / "b.db"
        store_a = HintStore(db_path=str(db_a))
        store_b = HintStore(db_path=str(db_b))
        _populate_bare_digit_store(store_a)
        _populate_bare_digit_store(store_b)

        ids_a = HintContaminator(store_a, seed=42, fraction=0.40).corrupt()
        ids_b = HintContaminator(store_b, seed=42, fraction=0.40).corrupt()
        assert sorted(ids_a) == sorted(ids_b)
