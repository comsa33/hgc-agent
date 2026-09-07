"""Unit tests for HintRecord + HintStore (US-001)."""

import time
import uuid

import numpy as np
import pytest

from hgc.memory import HintRecord, HintStore, _confidence, _cosine_sim, _recency

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """In-memory SQLite store (file in tmp_path so tests are isolated)."""
    s = HintStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _emb(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _emb_bytes(values: list[float]) -> bytes:
    return _emb(values).tobytes()


def _make_record(
    *,
    hint_type="location",
    polarity="positive",
    content="docid=42",
    content_meta=None,
    query_ctx="test query",
    embedding_values=None,
    trajectory_step=1,
    created_at=None,
    last_validated_at=None,
    success_count=0,
    failure_count=0,
    retrieval_count=0,
) -> HintRecord:
    if embedding_values is None:
        embedding_values = [1.0, 0.0, 0.0]
    if content_meta is None:
        content_meta = {}
    now = time.time()
    return HintRecord(
        hint_id=str(uuid.uuid4()),
        hint_type=hint_type,
        polarity=polarity,
        content=content,
        content_meta=content_meta,
        query_ctx=query_ctx,
        query_ctx_embedding=_emb_bytes(embedding_values),
        trajectory_step=trajectory_step,
        created_at=created_at if created_at is not None else now,
        last_validated_at=last_validated_at if last_validated_at is not None else now,
        success_count=success_count,
        failure_count=failure_count,
        retrieval_count=retrieval_count,
    )


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors(self):
        a = _emb([1.0, 2.0, 3.0])
        assert abs(_cosine_sim(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = _emb([1.0, 0.0])
        b = _emb([0.0, 1.0])
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_zero_vector(self):
        a = _emb([0.0, 0.0])
        b = _emb([1.0, 2.0])
        assert _cosine_sim(a, b) == 0.0


class TestConfidence:
    def test_all_success(self):
        assert _confidence(10, 0) == 1.0

    def test_all_failure(self):
        assert _confidence(0, 10) == 0.0

    def test_half(self):
        assert abs(_confidence(1, 1) - 0.5) < 1e-9

    def test_zero_zero(self):
        # max(1, 0+0) = 1 → 0/1 = 0
        assert _confidence(0, 0) == 0.0


class TestRecency:
    def test_just_created(self):
        now = time.time()
        r = _recency(now, t_half_days=30)
        assert 0.99 < r <= 1.0

    def test_half_life(self):
        t_half = 30
        old = time.time() - t_half * 86400
        r = _recency(old, t_half_days=t_half)
        assert abs(r - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Add + get round-trip
# ---------------------------------------------------------------------------


class TestAddGet:
    def test_roundtrip_fields(self, store):
        rec = _make_record(
            hint_type="strategy",
            polarity="negative",
            content="search('foo')",
            content_meta={"keyword": "foo"},
            query_ctx="find foo",
            embedding_values=[0.5, 0.5, 0.0],
            trajectory_step=3,
            success_count=2,
            failure_count=1,
            retrieval_count=4,
        )
        original_id = rec.hint_id
        returned_id = store.add(rec)
        assert returned_id == original_id

        fetched = store.get(original_id)
        assert fetched is not None
        assert fetched.hint_id == original_id
        assert fetched.hint_type == "strategy"
        assert fetched.polarity == "negative"
        assert fetched.content == "search('foo')"
        assert fetched.content_meta == {"keyword": "foo"}
        assert fetched.query_ctx == "find foo"
        assert fetched.trajectory_step == 3
        assert fetched.success_count == 2
        assert fetched.failure_count == 1
        assert fetched.retrieval_count == 4
        # Embedding round-trip
        emb_back = np.frombuffer(fetched.query_ctx_embedding, dtype=np.float32)
        np.testing.assert_array_almost_equal(emb_back, [0.5, 0.5, 0.0])

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("does-not-exist") is None

    def test_auto_id_when_empty(self, store):
        rec = _make_record()
        rec.hint_id = ""
        returned_id = store.add(rec)
        assert returned_id != ""
        assert store.get(returned_id) is not None


# ---------------------------------------------------------------------------
# Search ranking
# ---------------------------------------------------------------------------


class TestSearch:
    def _populate(self, store):
        """
        Three hints:
          h_high — embedding=[1,0,0], success=9, failure=1  (high sim + conf)
          h_mid  — embedding=[0.7,0.7,0], success=1, failure=1  (mid sim + conf)
          h_low  — embedding=[0,1,0], success=0, failure=0  (low sim, zero conf)
        Query embedding = [1, 0, 0]
        """
        h_high = _make_record(embedding_values=[1.0, 0.0, 0.0], success_count=9, failure_count=1)
        h_mid = _make_record(
            embedding_values=[0.7071, 0.7071, 0.0], success_count=1, failure_count=1
        )
        h_low = _make_record(embedding_values=[0.0, 1.0, 0.0], success_count=0, failure_count=0)
        store.add(h_high)
        store.add(h_mid)
        store.add(h_low)
        return h_high, h_mid, h_low

    def test_ranking_respects_similarity_and_confidence(self, store):
        h_high, h_mid, h_low = self._populate(store)
        q = _emb([1.0, 0.0, 0.0])
        results = store.search(q, k=3, alpha=1.0, beta=0.5, gamma=0.0, theta_pos=0.0)
        ids = [h.hint_id for h in results]
        assert ids[0] == h_high.hint_id, "highest sim+conf should be first"
        assert ids[1] == h_mid.hint_id

    def test_returns_at_most_k(self, store):
        self._populate(store)
        q = _emb([1.0, 0.0, 0.0])
        results = store.search(q, k=2, theta_pos=0.0)
        assert len(results) <= 2

    def test_positive_hints_filtered_below_theta_pos(self, store):
        # hint similarity to query = 0 (orthogonal), theta_pos=0.3 → filtered out
        h = _make_record(embedding_values=[0.0, 1.0, 0.0], polarity="positive")
        store.add(h)
        q = _emb([1.0, 0.0, 0.0])
        results = store.search(q, k=5, theta_pos=0.3)
        assert all(r.hint_id != h.hint_id for r in results)

    def test_negative_hints_filtered_below_theta_neg(self, store):
        # similarity ~ 0.707 < theta_neg=0.8 → filtered out
        h = _make_record(
            embedding_values=[0.7071, 0.7071, 0.0],
            polarity="negative",
        )
        store.add(h)
        q = _emb([1.0, 0.0, 0.0])
        results = store.search(q, k=5, theta_neg=0.8)
        assert all(r.hint_id != h.hint_id for r in results)

    def test_negative_hint_included_above_theta_neg(self, store):
        # similarity ≈ 1.0 > theta_neg=0.6 → included
        h = _make_record(
            embedding_values=[1.0, 0.0, 0.0],
            polarity="negative",
        )
        store.add(h)
        q = _emb([1.0, 0.0, 0.0])
        results = store.search(q, k=5, theta_neg=0.6)
        assert any(r.hint_id == h.hint_id for r in results)

    def test_retrieval_count_incremented(self, store):
        h = _make_record(embedding_values=[1.0, 0.0, 0.0], retrieval_count=0)
        store.add(h)
        q = _emb([1.0, 0.0, 0.0])
        store.search(q, k=1, theta_pos=0.0)
        fetched = store.get(h.hint_id)
        assert fetched.retrieval_count == 1

    def test_empty_store_returns_empty(self, store):
        q = _emb([1.0, 0.0, 0.0])
        assert store.search(q, k=5) == []


# ---------------------------------------------------------------------------
# update_on_outcome
# ---------------------------------------------------------------------------


class TestUpdateOnOutcome:
    def test_correct_increments_success_count(self, store):
        h = _make_record(success_count=2, failure_count=1)
        store.add(h)
        store.update_on_outcome([h.hint_id], correct=True)
        fetched = store.get(h.hint_id)
        assert fetched.success_count == 3
        assert fetched.failure_count == 1

    def test_correct_updates_last_validated_at(self, store):
        old_ts = time.time() - 1000
        h = _make_record(last_validated_at=old_ts, success_count=0)
        store.add(h)
        before = time.time()
        store.update_on_outcome([h.hint_id], correct=True)
        fetched = store.get(h.hint_id)
        assert fetched.last_validated_at >= before

    def test_incorrect_increments_failure_count(self, store):
        h = _make_record(success_count=3, failure_count=1)
        store.add(h)
        store.update_on_outcome([h.hint_id], correct=False)
        fetched = store.get(h.hint_id)
        assert fetched.failure_count == 2
        assert fetched.success_count == 3

    def test_incorrect_flips_polarity_when_confidence_drops_below_025(self, store):
        # success=0, failure=3 → new conf = 0/4 = 0.0 < 0.25
        h = _make_record(polarity="positive", success_count=0, failure_count=3)
        store.add(h)
        store.update_on_outcome([h.hint_id], correct=False)
        fetched = store.get(h.hint_id)
        assert fetched.polarity == "negative"

    def test_incorrect_does_not_flip_when_confidence_above_025(self, store):
        # success=3, failure=0 → new conf = 3/4 = 0.75 >= 0.25, no flip
        h = _make_record(polarity="positive", success_count=3, failure_count=0)
        store.add(h)
        store.update_on_outcome([h.hint_id], correct=False)
        fetched = store.get(h.hint_id)
        assert fetched.polarity == "positive"

    def test_nonexistent_hint_id_is_skipped(self, store):
        # Should not raise
        store.update_on_outcome(["no-such-id"], correct=True)

    def test_multiple_hint_ids(self, store):
        h1 = _make_record(success_count=1)
        h2 = _make_record(success_count=5)
        store.add(h1)
        store.add(h2)
        store.update_on_outcome([h1.hint_id, h2.hint_id], correct=True)
        assert store.get(h1.hint_id).success_count == 2
        assert store.get(h2.hint_id).success_count == 6


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


class TestPrune:
    def test_prune_rule1_failure_gt5_success0(self, store):
        h = _make_record(success_count=0, failure_count=6)
        store.add(h)
        deleted = store.prune()
        assert deleted >= 1
        assert store.get(h.hint_id) is None

    def test_prune_rule1_spares_hint_with_success(self, store):
        h = _make_record(success_count=1, failure_count=6)
        store.add(h)
        store.prune()
        assert store.get(h.hint_id) is not None

    def test_prune_rule2_old_unretrieved(self, store):
        old_ts = time.time() - 181 * 86400  # > 180 days
        h = _make_record(created_at=old_ts, retrieval_count=0)
        store.add(h)
        deleted = store.prune()
        assert deleted >= 1
        assert store.get(h.hint_id) is None

    def test_prune_rule2_spares_old_but_retrieved(self, store):
        old_ts = time.time() - 181 * 86400
        h = _make_record(created_at=old_ts, retrieval_count=1)
        store.add(h)
        store.prune()
        assert store.get(h.hint_id) is not None

    def test_prune_rule3_negative_hint_older_than_90_days(self, store):
        old_ts = time.time() - 91 * 86400
        h = _make_record(polarity="negative", created_at=old_ts)
        store.add(h)
        deleted = store.prune()
        assert deleted >= 1
        assert store.get(h.hint_id) is None

    def test_prune_rule3_spares_recent_negative(self, store):
        h = _make_record(polarity="negative")  # just created
        store.add(h)
        store.prune()
        assert store.get(h.hint_id) is not None

    def test_prune_returns_count(self, store):
        h1 = _make_record(success_count=0, failure_count=6)
        h2 = _make_record(success_count=0, failure_count=6)
        store.add(h1)
        store.add(h2)
        deleted = store.prune()
        assert deleted == 2

    def test_prune_empty_store(self, store):
        assert store.prune() == 0


# ---------------------------------------------------------------------------
# Scope filter (US-019)
# ---------------------------------------------------------------------------


class TestScopeFilter:
    """Verify typed-scope retrieval policy:
    - location hints: STRICT (only same scope_id returned)
    - entity hints:   LOOSE (cross-scope OK)
    - strategy hints: LOOSE (cross-scope OK)
    """

    def _make_scoped(self, hint_type: str, scope_id: str) -> HintRecord:
        rec = _make_record(hint_type=hint_type, embedding_values=[1.0, 0.0, 0.0])
        rec.scope_id = scope_id
        return rec

    def test_location_hint_cross_scope_filtered_out(self, store):
        """Location hint with scope_id='A' is NOT returned when searching scope_id='B'."""
        h = self._make_scoped("location", "A")
        store.add(h)
        q = _emb([1.0, 0.0, 0.0])
        results = store.search(q, k=5, theta_pos=0.0, scope_id="B")
        assert all(r.hint_id != h.hint_id for r in results)

    def test_entity_hint_cross_scope_returned(self, store):
        """Entity hint with scope_id='A' IS returned when searching scope_id='B'."""
        h = self._make_scoped("entity", "A")
        store.add(h)
        q = _emb([1.0, 0.0, 0.0])
        results = store.search(q, k=5, theta_pos=0.0, scope_id="B")
        assert any(r.hint_id == h.hint_id for r in results)

    def test_strategy_hint_cross_scope_returned(self, store):
        """Strategy hint with scope_id='A' IS returned when searching scope_id='B'."""
        h = self._make_scoped("strategy", "A")
        store.add(h)
        q = _emb([1.0, 0.0, 0.0])
        results = store.search(q, k=5, theta_pos=0.0, scope_id="B")
        assert any(r.hint_id == h.hint_id for r in results)

    def test_location_hint_same_scope_returned(self, store):
        """Location hint with scope_id='A' IS returned when searching scope_id='A'."""
        h = self._make_scoped("location", "A")
        store.add(h)
        q = _emb([1.0, 0.0, 0.0])
        results = store.search(q, k=5, theta_pos=0.0, scope_id="A")
        assert any(r.hint_id == h.hint_id for r in results)
