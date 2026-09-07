"""Unit tests for src/hgc/datasets/bcp.py.

All tests use temporary directories and a FakeLM — no real API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hgc.datasets.bcp import BCPDataset, _hash_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RECORDS = [
    {
        "query_id": str(i),
        "query": f"Sample question number {i}?",
        "answer": f"Answer {i}",
        "gold_docs": [],
        "negative_docs": [],
        "evidence_docs": [],
    }
    for i in range(1, 6)  # 5 records with query_ids 1-5
]


class FakeLM:
    """Returns a predetermined string and records how many times it was called."""

    def __init__(self, response: str = "paraphrased text") -> None:
        self._response = response
        self.call_count = 0

    def __call__(self, prompt: str) -> list[str]:  # noqa: ARG002
        self.call_count += 1
        return [self._response]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_bcp(tmp_path: Path) -> dict:
    """Create a temporary 5-record JSONL file and return path info."""
    jsonl_path = tmp_path / "queries.jsonl"
    with jsonl_path.open("w") as f:
        for rec in _SAMPLE_RECORDS:
            f.write(json.dumps(rec) + "\n")

    cache_path = tmp_path / "paraphrase_cache.json"

    return {
        "queries_path": str(jsonl_path),
        "cache_path": str(cache_path),
        "data_dir": str(tmp_path),
        "tmp_path": tmp_path,
    }


def make_ds(paths: dict) -> BCPDataset:
    return BCPDataset(
        queries_path=paths["queries_path"],
        cache_path=paths["cache_path"],
        data_dir=paths["data_dir"],
    )


# ---------------------------------------------------------------------------
# Test 1: deterministic selection — two calls yield identical query_id ordering
# ---------------------------------------------------------------------------


class TestSelect100Determinism:
    def test_same_seed_same_order(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        result_a = ds.select_n(n=5, seed=42)
        result_b = ds.select_n(n=5, seed=42)
        ids_a = [r["query_id"] for r in result_a]
        ids_b = [r["query_id"] for r in result_b]
        assert ids_a == ids_b

    def test_returns_correct_count(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        result = ds.select_n(n=3, seed=42)
        assert len(result) == 3

    def test_all_records_included_when_n_equals_total(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        result = ds.select_n(n=5, seed=42)
        returned_ids = {r["query_id"] for r in result}
        expected_ids = {r["query_id"] for r in _SAMPLE_RECORDS}
        assert returned_ids == expected_ids


# ---------------------------------------------------------------------------
# Test 2: different seeds produce different orderings
# ---------------------------------------------------------------------------


class TestSelect100DifferentSeeds:
    def test_different_seeds_different_order(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        ids_42 = [r["query_id"] for r in ds.select_n(n=5, seed=42)]
        ids_99 = [r["query_id"] for r in ds.select_n(n=5, seed=99)]
        # With 5 items it's possible (but very unlikely) they're the same;
        # we assert they differ as the spec requires.
        assert ids_42 != ids_99


# ---------------------------------------------------------------------------
# Test 3: paraphrase() with FakeLM caches result — second call skips LM
# ---------------------------------------------------------------------------


class TestParaphraseCache:
    def test_second_call_does_not_invoke_lm(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        lm = FakeLM("rephrased version")

        first = ds.paraphrase("What is the capital of France?", lm=lm)
        assert lm.call_count == 1
        assert first == "rephrased version"

        second = ds.paraphrase("What is the capital of France?", lm=lm)
        assert lm.call_count == 1  # no additional LM call
        assert second == first

    def test_different_texts_each_call_lm(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        lm = FakeLM("rephrased")

        ds.paraphrase("Question A?", lm=lm)
        ds.paraphrase("Question B?", lm=lm)
        assert lm.call_count == 2


# ---------------------------------------------------------------------------
# Test 4: paraphrase() cache persists across BCPDataset instances
# ---------------------------------------------------------------------------


class TestParaphraseCachePersistence:
    def test_cache_persists_across_instances(self, tmp_bcp):
        lm = FakeLM("cached paraphrase")

        # First instance writes to cache
        ds1 = make_ds(tmp_bcp)
        result1 = ds1.paraphrase("Who wrote Hamlet?", lm=lm)
        assert lm.call_count == 1

        # Second instance reads from disk — should not call LM again
        lm2 = FakeLM("should not be called")
        ds2 = make_ds(tmp_bcp)
        result2 = ds2.paraphrase("Who wrote Hamlet?", lm=lm2)

        assert lm2.call_count == 0
        assert result2 == result1

    def test_cache_file_is_valid_json(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        lm = FakeLM("a paraphrase")
        ds.paraphrase("Some question?", lm=lm)

        cache_path = Path(tmp_bcp["cache_path"])
        assert cache_path.exists()
        with cache_path.open() as f:
            data = json.load(f)
        assert isinstance(data, dict)
        key = _hash_key("Some question?")
        assert key in data
        assert data[key] == "a paraphrase"


# ---------------------------------------------------------------------------
# Test 5: load_or_build() writes dataset_path with expected structure
# ---------------------------------------------------------------------------


class TestLoadOrBuild:
    def test_writes_dataset_file(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        lm = FakeLM("para")
        result = ds.load_or_build(n=5, seed=42, paraphrase=True, lm=lm)

        snapshot_path = ds._dataset_path(n=5, seed=42)
        assert snapshot_path.exists()

        with snapshot_path.open() as f:
            on_disk = json.load(f)

        assert on_disk == result

    def test_n_aware_snapshots_are_independent(self, tmp_bcp):
        """Growing n must NOT silently reuse a smaller snapshot."""
        ds = make_ds(tmp_bcp)
        lm = FakeLM("para")
        result_n3 = ds.load_or_build(n=3, seed=42, paraphrase=True, lm=lm)
        result_n5 = ds.load_or_build(n=5, seed=42, paraphrase=True, lm=lm)

        assert len(result_n3["queries"]) == 3
        assert len(result_n5["queries"]) == 5
        assert ds._dataset_path(n=3, seed=42) != ds._dataset_path(n=5, seed=42)
        # Smaller-n snapshot must be a prefix of the larger-n snapshot under
        # the same seed: deterministic shuffle is truncated by ``select_n``.
        ids_n3 = [q["query_id"] for q in result_n3["queries"]]
        ids_n5 = [q["query_id"] for q in result_n5["queries"]]
        assert ids_n5[: len(ids_n3)] == ids_n3

    def test_result_has_queries_and_paraphrased_keys(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        lm = FakeLM("para")
        result = ds.load_or_build(n=5, seed=42, paraphrase=True, lm=lm)

        assert "queries" in result
        assert "paraphrased" in result

    def test_queries_and_paraphrased_are_aligned(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        lm = FakeLM("para")
        result = ds.load_or_build(n=5, seed=42, paraphrase=True, lm=lm)

        assert len(result["queries"]) == len(result["paraphrased"])
        assert len(result["queries"]) == 5

    def test_second_call_loads_from_disk(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        lm = FakeLM("para")
        ds.load_or_build(n=5, seed=42, paraphrase=True, lm=lm)
        calls_after_first = lm.call_count

        lm2 = FakeLM("should not be called")
        ds2 = make_ds(tmp_bcp)
        ds2.load_or_build(n=5, seed=42, paraphrase=True, lm=lm2)

        assert lm2.call_count == 0
        assert calls_after_first == 5  # one per query on first build

    def test_paraphrase_false_uses_original_text(self, tmp_bcp):
        ds = make_ds(tmp_bcp)
        result = ds.load_or_build(n=5, seed=42, paraphrase=False)

        for q, p in zip(result["queries"], result["paraphrased"], strict=False):
            assert p == q["query"]


# ---------------------------------------------------------------------------
