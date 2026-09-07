"""Unit tests for src/hgc/datasets/qasper.py and make_tools_qasper.

All tests use mocks/fixtures — NO actual HuggingFace downloads.
"""


from __future__ import annotations

# HGC-009 delivered: hgc.factories now ships make_tools_* factories
from pathlib import Path

import pytest

from hgc.datasets.qasper import (
    QASPERDataset,
    _build_docs,
    _extract_answer,
)
from hgc.factories import make_tools_qasper

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_PAPER_1 = {
    "id": "paper_abc",
    "title": "A Study of Things",
    "abstract": "We study things.",
    "full_text": {
        "section_name": ["Introduction", "Method", "FLOAT SELECTED Table 1", "Results"],
        "paragraphs": [
            ["This paper introduces X.", "We propose a new method."],
            ["Our method does Y.", "It achieves Z."],
            ["Table 1 caption here."],
            ["Results show improvement."],
        ],
    },
    "qas": {
        "question": ["What does the method do?", "What do results show?"],
        "answers": [
            {
                "answer": [
                    {
                        "extractive_spans": ["Our method does Y."],
                        "free_form_answer": "",
                        "yes_no": None,
                        "unanswerable": False,
                    }
                ]
            },
            {
                "answer": [
                    {
                        "extractive_spans": [],
                        "free_form_answer": "Results show improvement.",
                        "yes_no": None,
                        "unanswerable": False,
                    }
                ]
            },
        ],
    },
    "figures_and_tables": [],
}

_PAPER_2 = {
    "id": "paper_xyz",
    "title": "Another Study",
    "abstract": "We study other things.",
    "full_text": {
        "section_name": ["Background", "Conclusion"],
        "paragraphs": [
            ["Background paragraph one.", "Background paragraph two."],
            ["We conclude that A is true."],
        ],
    },
    "qas": {
        "question": ["Is A true?", "What is the topic?"],
        "answers": [
            {
                "answer": [
                    {
                        "extractive_spans": [],
                        "free_form_answer": "",
                        "yes_no": True,
                        "unanswerable": False,
                    }
                ]
            },
            {
                "answer": [
                    {
                        "extractive_spans": [],
                        "free_form_answer": "",
                        "yes_no": None,
                        "unanswerable": True,
                    }
                ]
            },
        ],
    },
    "figures_and_tables": [],
}

_FAKE_HF_DATASET = [_PAPER_1, _PAPER_2]


@pytest.fixture()
def tmp_ds(tmp_path: Path) -> QASPERDataset:
    """Return a QASPERDataset pointing at a temp directory."""
    return QASPERDataset(cache_dir=str(tmp_path / "qasper"), split="validation")


def _patch_hf(ds: QASPERDataset):
    """Patch datasets.load_dataset so ds._load_hf() returns the fixture."""
    ds._hf_data = _FAKE_HF_DATASET
    return ds


# ---------------------------------------------------------------------------
# Test 1: test_loader_schema
# ---------------------------------------------------------------------------


class TestLoaderSchema:
    def test_query_record_has_required_keys(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        assert len(records) == 4  # 2 questions per paper × 2 papers
        for rec in records:
            assert "query_id" in rec
            assert "question" in rec
            assert "answer" in rec
            assert "docs" in rec
            assert "paper_id" in rec
            assert "answer_type" in rec

    def test_query_id_format(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        # query_id should encode paper_id + question index
        ids = [r["query_id"] for r in records]
        assert "paper_abc_q0" in ids
        assert "paper_abc_q1" in ids
        assert "paper_xyz_q0" in ids
        assert "paper_xyz_q1" in ids

    def test_paper_id_field_populated(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        for rec in records:
            assert rec["paper_id"] in ("paper_abc", "paper_xyz")


# ---------------------------------------------------------------------------
# Test 2: test_select_N_deterministic
# ---------------------------------------------------------------------------


class TestSelectNDeterministic:
    def test_same_seed_same_query_ids(self, tmp_ds):
        _patch_hf(tmp_ds)
        ids_a = [r["query_id"] for r in tmp_ds.select_N(n=4, seed=42)]
        # Reset hf_data (already cached) for second call
        ids_b = [r["query_id"] for r in tmp_ds.select_N(n=4, seed=42)]
        assert ids_a == ids_b

    def test_different_seeds_may_differ(self, tmp_ds):
        _patch_hf(tmp_ds)
        ids_42 = [r["query_id"] for r in tmp_ds.select_N(n=4, seed=42)]
        ids_99 = [r["query_id"] for r in tmp_ds.select_N(n=4, seed=99)]
        # With 4 records it's unlikely but possible to be equal; at least verify types
        assert isinstance(ids_42, list)
        assert isinstance(ids_99, list)

    def test_returns_n_records(self, tmp_ds):
        _patch_hf(tmp_ds)
        result = tmp_ds.select_N(n=2, seed=42)
        assert len(result) == 2

    def test_returns_all_when_n_equals_total(self, tmp_ds):
        _patch_hf(tmp_ds)
        result = tmp_ds.select_N(n=4, seed=42)
        assert len(result) == 4
        returned_ids = {r["query_id"] for r in result}
        assert "paper_abc_q0" in returned_ids
        assert "paper_xyz_q1" in returned_ids


# ---------------------------------------------------------------------------
# Test 3: test_docs_per_query
# ---------------------------------------------------------------------------


class TestDocsPerQuery:
    def test_docs_list_maps_to_paper_sections(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        # paper_abc has 4 sections, but 1 is FLOAT SELECTED → 3 docs
        paper_abc_recs = [r for r in records if r["paper_id"] == "paper_abc"]
        assert len(paper_abc_recs) == 2
        # Both questions share the same paper, so same docs
        assert len(paper_abc_recs[0]["docs"]) == len(paper_abc_recs[1]["docs"])

    def test_float_sections_excluded(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        paper_abc_docs = [r for r in records if r["paper_id"] == "paper_abc"][0]["docs"]
        doc_ids = [d["docid"] for d in paper_abc_docs]
        # Section index 2 is "FLOAT SELECTED Table 1" — should be excluded
        assert "qasper_paper_abc_sec_2" not in doc_ids

    def test_docs_have_docid_and_text(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        for rec in records:
            for doc in rec["docs"]:
                assert "docid" in doc
                assert "text" in doc
                assert doc["docid"].startswith("qasper_")

    def test_doc_text_joins_paragraphs(self):
        docs = _build_docs(
            "p1",
            {
                "section_name": ["Intro"],
                "paragraphs": [["Para one.", "Para two."]],
            },
        )
        assert len(docs) == 1
        assert "Para one." in docs[0]["text"]
        assert "Para two." in docs[0]["text"]
        assert "\n\n" in docs[0]["text"]


# ---------------------------------------------------------------------------
# Test 4: test_answer_extraction_priority
# ---------------------------------------------------------------------------


class TestAnswerExtractionPriority:
    def test_extractive_takes_priority(self):
        ann_list = [
            {
                "extractive_spans": ["span text"],
                "free_form_answer": "free form",
                "yes_no": True,
                "unanswerable": False,
            }
        ]
        answer, ans_type = _extract_answer(ann_list)
        assert answer == "span text"
        assert ans_type == "extractive"

    def test_abstractive_when_no_extractive(self):
        ann_list = [
            {
                "extractive_spans": [],
                "free_form_answer": "free form answer",
                "yes_no": None,
                "unanswerable": False,
            }
        ]
        answer, ans_type = _extract_answer(ann_list)
        assert answer == "free form answer"
        assert ans_type == "abstractive"

    def test_yesno_when_no_extractive_or_abstractive(self):
        ann_list = [
            {
                "extractive_spans": [],
                "free_form_answer": "",
                "yes_no": True,
                "unanswerable": False,
            }
        ]
        answer, ans_type = _extract_answer(ann_list)
        assert answer == "yes"
        assert ans_type == "yesno"

    def test_yesno_false(self):
        ann_list = [
            {
                "extractive_spans": [],
                "free_form_answer": "",
                "yes_no": False,
                "unanswerable": False,
            }
        ]
        answer, ans_type = _extract_answer(ann_list)
        assert answer == "no"
        assert ans_type == "yesno"

    def test_unanswerable_fallback(self):
        ann_list = [
            {
                "extractive_spans": [],
                "free_form_answer": "",
                "yes_no": None,
                "unanswerable": True,
            }
        ]
        answer, ans_type = _extract_answer(ann_list)
        assert answer == "UNANSWERABLE"
        assert ans_type == "unanswerable"

    def test_empty_list_returns_unanswerable(self):
        answer, ans_type = _extract_answer([])
        assert answer == "UNANSWERABLE"
        assert ans_type == "unanswerable"

    def test_answer_types_from_fixture_papers(self, tmp_ds):
        _patch_hf(tmp_ds)
        records = tmp_ds.load_all()
        by_id = {r["query_id"]: r for r in records}
        assert by_id["paper_abc_q0"]["answer_type"] == "extractive"
        assert by_id["paper_abc_q1"]["answer_type"] == "abstractive"
        assert by_id["paper_xyz_q0"]["answer_type"] == "yesno"
        assert by_id["paper_xyz_q1"]["answer_type"] == "unanswerable"


# ---------------------------------------------------------------------------
# Test 5: make_tools_qasper — returns 4 callables and can search
# ---------------------------------------------------------------------------


class TestMakeToolsQasper:
    def _make_qr(self):
        return {
            "query_id": "paper_abc_q0",
            "question": "What does the method do?",
            "answer": "Our method does Y.",
            "paper_id": "paper_abc",
            "answer_type": "extractive",
            "docs": [
                {
                    "docid": "qasper_paper_abc_sec_0",
                    "text": "This paper introduces X. We propose a new method.",
                },
                {"docid": "qasper_paper_abc_sec_1", "text": "Our method does Y. It achieves Z."},
                {"docid": "qasper_paper_abc_sec_3", "text": "Results show improvement."},
            ],
        }

    def test_returns_four_callables(self):
        qr = self._make_qr()
        tools = make_tools_qasper(qr)
        assert len(tools) == 4
        for t in tools:
            assert callable(t)

    def test_list_document_ids(self):
        qr = self._make_qr()
        list_ids, _, _, _ = make_tools_qasper(qr)
        ids = list_ids()
        assert len(ids) == 3
        assert "qasper_paper_abc_sec_0" in ids

    def test_get_document_snippet(self):
        qr = self._make_qr()
        _, _, get_snippet, _ = make_tools_qasper(qr)
        snippet = get_snippet("qasper_paper_abc_sec_1", 0, 50)
        assert "Our method" in snippet

    def test_get_document_snippet_unknown_docid(self):
        qr = self._make_qr()
        _, _, get_snippet, _ = make_tools_qasper(qr)
        result = get_snippet("nonexistent_id")
        assert "ERROR" in result

    def test_search_documents_finds_hit(self):
        qr = self._make_qr()
        _, _, _, search = make_tools_qasper(qr)
        hits = search("method")
        assert len(hits) >= 1
        assert all("docid" in h and "snippet" in h for h in hits)

    def test_search_documents_no_hit(self):
        qr = self._make_qr()
        _, _, _, search = make_tools_qasper(qr)
        hits = search("xyzzy_not_in_any_doc")
        assert hits == []

    def test_get_document_title(self):
        qr = self._make_qr()
        _, get_title, _, _ = make_tools_qasper(qr)
        title = get_title("qasper_paper_abc_sec_0")
        assert isinstance(title, str)
        assert len(title) > 0
