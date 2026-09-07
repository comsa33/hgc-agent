"""Unit tests for hgc.gate.docid_check (G2) and parse_docid helper."""

from __future__ import annotations

import time
import uuid

import numpy as np

from hgc.gate import DocidCheck, parse_docid, resolve_docid
from hgc.memory import HintRecord


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


def test_parse_docid_bare_digits():
    assert parse_docid("63970") == "63970"


def test_parse_docid_legacy_form():
    assert parse_docid("docid=12345") == "12345"


def test_parse_docid_inline_in_text():
    assert parse_docid("pointed to docid=5412 in the corpus") == "5412"


def test_parse_docid_returns_none_for_non_docid_text():
    assert parse_docid("Queen Arwa University") is None


def test_docid_check_passes_when_docid_present():
    h = _make_hint("location", "5412")
    assert DocidCheck()(h, {"5412": "doc text"}) is True


def test_docid_check_fails_when_docid_absent():
    h = _make_hint("location", "5412")
    assert DocidCheck()(h, {"9999": "some other doc"}) is False


def test_docid_check_passes_non_location_hint():
    h = _make_hint("entity", "some entity name")
    assert DocidCheck()(h, {}) is True


def test_docid_check_passes_location_hint_with_unparseable_content():
    # Not a docid pattern — let G3 decide via content verification.
    h = _make_hint("location", "section: Introduction")
    assert DocidCheck()(h, {}) is True


# --- string docids (long-context / RAG backbones) -------------------------
# These cells mint the location hint straight from the docid the backbone
# consumed, so the content is a doc_map key verbatim and never parses as
# digits. Before resolve_docid existed, G2 passed every one of them.

_QASPER_DOCID = "qasper_1909.00015_q0_oracle"
_FINBENCH_DOCID = "AMAZON_2019_10K_page_37"


def test_docid_check_passes_string_docid_present_in_pool():
    h = _make_hint("location", _QASPER_DOCID)
    assert DocidCheck()(h, {_QASPER_DOCID: "paper text"}) is True


def test_docid_check_fails_string_docid_absent_from_pool():
    h = _make_hint("location", _FINBENCH_DOCID)
    assert DocidCheck()(h, {"GENERALMILLS_2022_10K_page_44": "other doc"}) is False


def test_docid_check_fails_contaminated_string_docid():
    # cross_swap suffixes a corrupted hint, which drops it out of the pool.
    h = _make_hint("location", _QASPER_DOCID + "__CORRUPTED_0042")
    assert DocidCheck()(h, {_QASPER_DOCID: "paper text"}) is False


def test_docid_check_defers_to_g3_when_doc_pool_unknown():
    # Empty pool means we cannot judge membership — leave it to G3.
    h = _make_hint("location", _QASPER_DOCID)
    assert DocidCheck()(h, {}) is True


def test_resolve_docid_prefers_exact_key_over_digit_parse():
    # A string key that also contains digits must resolve as itself, not as
    # whatever the digit parser scrapes out of it.
    doc_map = {"AMAZON_2019_10K_page_37": "text"}
    assert resolve_docid("AMAZON_2019_10K_page_37", doc_map) == "AMAZON_2019_10K_page_37"


def test_resolve_docid_falls_back_to_numeric_extraction():
    assert resolve_docid("docid=63970", {"63970": "text"}) == "63970"


def test_resolve_docid_returns_none_when_unresolvable():
    assert resolve_docid("section: Introduction", {"63970": "text"}) is None
