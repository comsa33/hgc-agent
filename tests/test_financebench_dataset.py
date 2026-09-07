"""Unit tests for src/hgc/datasets/financebench.py and make_tools_financebench.

All tests use mocks/fixtures — NO actual HuggingFace downloads.
"""


from __future__ import annotations

# HGC-009 delivered: hgc.factories now ships make_tools_* factories
from pathlib import Path

import pytest

from hgc.datasets.financebench import (
    FinanceBenchDataset,
    _build_docs_from_evidence,
    _row_to_record,
)
from hgc.factories import make_tools_financebench

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_EVIDENCE_ROW_0 = [
    {
        "evidence_text": "Revenue was $10B in fiscal year 2021.",
        "doc_name": "JNJ_2021_10K",
        "evidence_page_num": 42,
        "evidence_text_full_page": "Full page 42 text with revenue details. Revenue was $10B.",
    },
    {
        "evidence_text": "Operating income was $3B.",
        "doc_name": "JNJ_2021_10K",
        "evidence_page_num": 55,
        "evidence_text_full_page": "Full page 55 text with operating income details.",
    },
]

_EVIDENCE_ROW_1 = [
    {
        "evidence_text": "Net income was $1.5B.",
        "doc_name": "AAPL_2022_10K",
        "evidence_page_num": 10,
        "evidence_text_full_page": "Full page 10 with net income information.",
    },
]

_ROW_0 = {
    "financebench_id": "financebench_id_00001",
    "company": "Johnson & Johnson",
    "doc_name": "JNJ_2021_10K",
    "doc_type": "10k",
    "doc_period": 2021,
    "doc_link": "https://example.com/jnj_2021_10k.pdf",
    "question": "What was JNJ's revenue in fiscal year 2021?",
    "answer": "$10B",
    "justification": "Revenue details are on page 42.",
    "evidence": _EVIDENCE_ROW_0,
    "question_type": "domain-relevant",
    "question_reasoning": "Information extraction",
    "gics_sector": "Health Care",
    "dataset_subset_label": "OPEN_SOURCE",
}

_ROW_1 = {
    "financebench_id": "financebench_id_00002",
    "company": "Apple",
    "doc_name": "AAPL_2022_10K",
    "doc_type": "10k",
    "doc_period": 2022,
    "doc_link": "https://example.com/aapl_2022_10k.pdf",
    "question": "What was Apple's net income in 2022?",
    "answer": "$1.5B",
    "justification": "Net income on page 10.",
    "evidence": _EVIDENCE_ROW_1,
    "question_type": "metrics-generated",
    "question_reasoning": "Numerical reasoning",
    "gics_sector": "Information Technology",
    "dataset_subset_label": "OPEN_SOURCE",
}

_FAKE_HF_DATASET = [_ROW_0, _ROW_1]


@pytest.fixture()
def tmp_ds(tmp_path: Path) -> FinanceBenchDataset:
    """Return a FinanceBenchDataset pointing at a temp directory."""
    return FinanceBenchDataset(cache_dir=str(tmp_path / "financebench"))


def _patch_hf(ds: FinanceBenchDataset):
    """Inject fake HF data so no network call is made."""
    ds._hf_data = _FAKE_HF_DATASET
    return ds


# ---------------------------------------------------------------------------
# Test 1: test_loader_schema
# ---------------------------------------------------------------------------


class TestLoaderSchema:
    def test_query_record_has_required_keys(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        assert len(records) == 2
        for rec in records:
            assert "query_id" in rec
            assert "question" in rec
            assert "answer" in rec
            assert "docs" in rec
            assert "doc_name" in rec
            assert "doc_period" in rec
            assert "question_type" in rec

    def test_query_id_format(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        ids = [r["query_id"] for r in records]
        assert "financebench_0" in ids
        assert "financebench_1" in ids

    def test_query_id_prefixed_financebench(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        for rec in records:
            assert rec["query_id"].startswith("financebench_")

    def test_doc_name_populated(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        doc_names = {r["doc_name"] for r in records}
        assert "JNJ_2021_10K" in doc_names
        assert "AAPL_2022_10K" in doc_names

    def test_doc_period_as_string(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        for rec in records:
            assert isinstance(rec["doc_period"], str)

    def test_question_type_preserved(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        types = {r["question_type"] for r in records}
        assert "domain-relevant" in types
        assert "metrics-generated" in types


# ---------------------------------------------------------------------------
# Test 2: test_select_N_deterministic
# ---------------------------------------------------------------------------


class TestSelectNDeterministic:
    def test_same_seed_same_order(self, tmp_ds):
        _patch_hf(tmp_ds)
        ids_a = [r["query_id"] for r in tmp_ds.select_N(seed=42, n=2)]
        ids_b = [r["query_id"] for r in tmp_ds.select_N(seed=42, n=2)]
        assert ids_a == ids_b

    def test_different_seeds_may_differ(self, tmp_ds):
        _patch_hf(tmp_ds)
        ids_42 = [r["query_id"] for r in tmp_ds.select_N(seed=42, n=2)]
        ids_99 = [r["query_id"] for r in tmp_ds.select_N(seed=99, n=2)]
        assert isinstance(ids_42, list)
        assert isinstance(ids_99, list)

    def test_returns_n_records(self, tmp_ds):
        _patch_hf(tmp_ds)
        result = tmp_ds.select_N(seed=42, n=1)
        assert len(result) == 1

    def test_returns_all_when_n_equals_total(self, tmp_ds):
        _patch_hf(tmp_ds)
        result = tmp_ds.select_N(seed=42, n=2)
        assert len(result) == 2
        returned_ids = {r["query_id"] for r in result}
        assert "financebench_0" in returned_ids
        assert "financebench_1" in returned_ids


# ---------------------------------------------------------------------------
# Test 3: test_docs_from_evidence_pages
# ---------------------------------------------------------------------------


class TestDocsFromEvidencePages:
    def test_docs_built_from_evidence_text_full_page(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        jnj_rec = next(r for r in records if r["doc_name"] == "JNJ_2021_10K")
        # Two distinct evidence pages → 2 docs
        assert len(jnj_rec["docs"]) == 2

    def test_docs_have_docid_and_text(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        for rec in records:
            for doc in rec["docs"]:
                assert "docid" in doc
                assert "text" in doc
                assert doc["text"].strip() != ""

    def test_docid_uses_doc_name_and_page_num(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        jnj_rec = next(r for r in records if r["doc_name"] == "JNJ_2021_10K")
        doc_ids = {d["docid"] for d in jnj_rec["docs"]}
        assert "JNJ_2021_10K_page_42" in doc_ids
        assert "JNJ_2021_10K_page_55" in doc_ids

    def test_doc_text_is_full_page_text(self):
        evidence = [
            {
                "evidence_text_full_page": "Full page content here.",
                "evidence_page_num": 5,
            }
        ]
        docs = _build_docs_from_evidence("DOC_NAME", evidence)
        assert len(docs) == 1
        assert docs[0]["text"] == "Full page content here."
        assert docs[0]["docid"] == "DOC_NAME_page_5"


# ---------------------------------------------------------------------------
# Test 4: test_fallback_to_justification
# ---------------------------------------------------------------------------


class TestFallbackToJustification:
    def test_fallback_when_evidence_empty(self):
        row = {
            "doc_name": "XYZ_2020_10K",
            "doc_period": 2020,
            "question": "What is the revenue?",
            "answer": "$5B",
            "justification": "Revenue is stated on page 3.",
            "evidence": [],
            "question_type": "novel-generated",
        }
        rec = _row_to_record(row, row_idx=7)
        assert len(rec["docs"]) == 1
        assert rec["docs"][0]["text"] == "Revenue is stated on page 3."
        assert "fallback" in rec["docs"][0]["docid"]

    def test_fallback_when_evidence_pages_all_blank(self):
        row = {
            "doc_name": "XYZ_2020_10K",
            "doc_period": 2020,
            "question": "What is the revenue?",
            "answer": "$5B",
            "justification": "Revenue is stated on page 3.",
            "evidence": [
                {"evidence_text_full_page": "   ", "evidence_page_num": 1},
                {"evidence_text_full_page": "", "evidence_page_num": 2},
            ],
            "question_type": "novel-generated",
        }
        rec = _row_to_record(row, row_idx=8)
        assert len(rec["docs"]) == 1
        assert rec["docs"][0]["text"] == "Revenue is stated on page 3."

    def test_fallback_uses_question_when_justification_also_absent(self):
        row = {
            "doc_name": "XYZ_2020_10K",
            "doc_period": 2020,
            "question": "What is the revenue?",
            "answer": "$5B",
            "justification": "",
            "evidence": [],
            "question_type": "novel-generated",
        }
        rec = _row_to_record(row, row_idx=9)
        assert len(rec["docs"]) == 1
        assert rec["docs"][0]["text"] == "What is the revenue?"


# ---------------------------------------------------------------------------
# Test 5: make_tools_financebench — returns 4 callables and can search
# ---------------------------------------------------------------------------


class TestMakeToolsFinanceBench:
    def _make_qr(self):
        return {
            "query_id": "financebench_0",
            "question": "What was JNJ's revenue?",
            "answer": "$10B",
            "doc_name": "JNJ_2021_10K",
            "doc_period": "2021",
            "question_type": "domain-relevant",
            "docs": [
                {
                    "docid": "JNJ_2021_10K_page_42",
                    "text": "Full page 42 text with revenue details. Revenue was $10B.",
                },
                {
                    "docid": "JNJ_2021_10K_page_55",
                    "text": "Full page 55 text with operating income details.",
                },
            ],
        }

    def test_returns_four_callables(self):
        qr = self._make_qr()
        tools = make_tools_financebench(qr)
        assert len(tools) == 4
        for t in tools:
            assert callable(t)

    def test_list_document_ids(self):
        qr = self._make_qr()
        list_ids, _, _, _ = make_tools_financebench(qr)
        ids = list_ids()
        assert len(ids) == 2
        assert "JNJ_2021_10K_page_42" in ids
        assert "JNJ_2021_10K_page_55" in ids

    def test_get_document_snippet(self):
        qr = self._make_qr()
        _, _, get_snippet, _ = make_tools_financebench(qr)
        snippet = get_snippet("JNJ_2021_10K_page_42", 0, 50)
        assert "revenue" in snippet.lower()

    def test_get_document_snippet_unknown_docid(self):
        qr = self._make_qr()
        _, _, get_snippet, _ = make_tools_financebench(qr)
        result = get_snippet("nonexistent_id")
        assert "ERROR" in result

    def test_search_documents_finds_hit(self):
        qr = self._make_qr()
        _, _, _, search = make_tools_financebench(qr)
        hits = search("revenue")
        assert len(hits) >= 1
        assert all("docid" in h and "snippet" in h for h in hits)

    def test_search_documents_no_hit(self):
        qr = self._make_qr()
        _, _, _, search = make_tools_financebench(qr)
        hits = search("xyzzy_not_in_any_doc")
        assert hits == []

    def test_get_document_title(self):
        qr = self._make_qr()
        _, get_title, _, _ = make_tools_financebench(qr)
        title = get_title("JNJ_2021_10K_page_42")
        assert isinstance(title, str)
        assert len(title) > 0

    def test_get_document_title_unknown_docid(self):
        qr = self._make_qr()
        _, get_title, _, _ = make_tools_financebench(qr)
        result = get_title("nonexistent")
        assert "ERROR" in result
