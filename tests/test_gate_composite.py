"""End-to-end tests for hgc.gate.composite.CompositeGate."""

from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

from hgc.gate import CompositeGate, DocidCheck, ScopeFilter
from hgc.memory import HintRecord


def _make_hint(
    hint_type: str,
    content: str,
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


@pytest.fixture
def always_yes_verifier():
    return lambda question, answer, doc_text: True


@pytest.fixture
def always_no_verifier():
    return lambda question, answer, doc_text: False


def test_passes_when_all_three_agree(always_yes_verifier):
    hint = _make_hint("location", "5412", scope_id="scope-a")
    gate = CompositeGate(support_verifier=always_yes_verifier)
    decision = gate(
        question="What is X?",
        answer="Madrid",
        hints=[hint],
        docs=[{"docid": "5412", "text": "Madrid is the capital."}],
        scope_id="scope-a",
    )
    assert decision.passed is True
    assert decision.reason == "verified"
    assert decision.evidence_hint is hint


def test_rejects_on_scope_mismatch(always_yes_verifier):
    hint = _make_hint("location", "5412", scope_id="scope-b")
    gate = CompositeGate(support_verifier=always_yes_verifier)
    decision = gate(
        question="q",
        answer="a",
        hints=[hint],
        docs=[{"docid": "5412", "text": "anything"}],
        scope_id="scope-a",
    )
    assert decision.passed is False
    assert decision.reason == "no_location_hint_in_scope"


def test_rejects_on_missing_docid(always_yes_verifier):
    hint = _make_hint("location", "5412", scope_id="scope-a")
    gate = CompositeGate(support_verifier=always_yes_verifier)
    decision = gate(
        question="q",
        answer="a",
        hints=[hint],
        docs=[{"docid": "9999", "text": "unrelated doc"}],
        scope_id="scope-a",
    )
    assert decision.passed is False
    assert decision.reason == "no_hint_passed_all_components"


def test_rejects_on_support_verifier_no(always_no_verifier):
    hint = _make_hint("location", "5412", scope_id="scope-a")
    gate = CompositeGate(support_verifier=always_no_verifier)
    decision = gate(
        question="q",
        answer="a",
        hints=[hint],
        docs=[{"docid": "5412", "text": "contradicting content"}],
        scope_id="scope-a",
    )
    assert decision.passed is False


def test_rejects_when_no_location_hints(always_yes_verifier):
    hint = _make_hint("entity", "some entity")
    gate = CompositeGate(support_verifier=always_yes_verifier)
    decision = gate(
        question="q",
        answer="a",
        hints=[hint],
        docs=[{"docid": "5412", "text": "anything"}],
        scope_id="scope-a",
    )
    assert decision.passed is False
    assert decision.reason == "no_location_hint_in_scope"


def test_ablation_g3_disabled_accepts_without_verifier_call(always_no_verifier):
    # With G3 off, the gate should accept on G1+G2 alone even if G3 would say no.
    hint = _make_hint("location", "5412", scope_id="scope-a")
    gate = CompositeGate(support_verifier=always_no_verifier, enabled={"g1", "g2"})
    decision = gate(
        question="q",
        answer="a",
        hints=[hint],
        docs=[{"docid": "5412", "text": "doc text"}],
        scope_id="scope-a",
    )
    assert decision.passed is True


def test_ablation_g2_disabled_does_not_require_docid_presence(always_yes_verifier):
    hint = _make_hint("location", "5412", scope_id="scope-a")
    gate = CompositeGate(support_verifier=always_yes_verifier, enabled={"g1", "g3"})
    # docid 5412 is NOT in docs, but with G2 disabled and G3=yes we still need a doc to
    # hand to G3. Without doc text, G3 is skipped -> no candidate passes.
    decision = gate(
        question="q",
        answer="a",
        hints=[hint],
        docs=[{"docid": "9999", "text": "unrelated"}],
        scope_id="scope-a",
    )
    # With G2 off and docid 5412 not in doc_map, doc_text is empty and G3 is skipped.
    # This documents the current behaviour — no hint passes.
    assert decision.passed is False


def test_ablation_g1_disabled_accepts_cross_scope_hint(always_yes_verifier):
    hint = _make_hint("location", "5412", scope_id="scope-b")
    gate = CompositeGate(
        scope_filter=ScopeFilter(),
        docid_check=DocidCheck(),
        support_verifier=always_yes_verifier,
        enabled={"g2", "g3"},
    )
    decision = gate(
        question="q",
        answer="a",
        hints=[hint],
        docs=[{"docid": "5412", "text": "doc"}],
        scope_id="scope-a",
    )
    assert decision.passed is True


def test_returns_first_passing_hint_as_evidence(always_yes_verifier):
    h1 = _make_hint("location", "1111", scope_id="scope-a")
    h2 = _make_hint("location", "2222", scope_id="scope-a")
    gate = CompositeGate(support_verifier=always_yes_verifier)
    decision = gate(
        question="q",
        answer="a",
        hints=[h1, h2],
        docs=[
            {"docid": "9999", "text": "no"},  # h1 fails G2
            {"docid": "2222", "text": "doc for h2"},  # h2 passes
        ],
        scope_id="scope-a",
    )
    assert decision.passed is True
    assert decision.evidence_hint is h2


# ---------------------------------------------------------------------------
# stage_counts: which component dropped which candidate
# ---------------------------------------------------------------------------


def test_stage_counts_census_a_mixed_rejection(always_no_verifier):
    """One hint per failure mode; the census attributes each to its stage."""
    doc_map = [{"docid": "5412", "text": "Madrid is the capital."}]
    hints = [
        _make_hint("location", "5412", scope_id="other-scope"),  # dropped by G1
        _make_hint("location", "9999", scope_id="scope-a"),  # dropped by G2
        _make_hint("location", "5412", scope_id="scope-a"),  # reaches G3, LM says no
    ]
    gate = CompositeGate(support_verifier=always_no_verifier)
    decision = gate(
        question="What is X?",
        answer="Madrid",
        hints=hints,
        docs=doc_map,
        scope_id="scope-a",
    )
    assert decision.passed is False
    assert decision.stage_counts == {
        "g1_scope": 1,
        "g2_docid": 1,
        "g3_doc_unresolved": 0,
        "g3_llm": 1,
    }


def test_stage_counts_on_accept_records_prior_drops(always_yes_verifier):
    """An accept still reports the hints that fell before the accepting one."""
    hints = [
        _make_hint("location", "9999", scope_id="scope-a"),  # dropped by G2
        _make_hint("location", "5412", scope_id="scope-a"),  # accepted
    ]
    gate = CompositeGate(support_verifier=always_yes_verifier)
    decision = gate(
        question="What is X?",
        answer="Madrid",
        hints=hints,
        docs=[{"docid": "5412", "text": "Madrid is the capital."}],
        scope_id="scope-a",
    )
    assert decision.passed is True
    assert decision.stage_counts == {
        "g1_scope": 0,
        "g2_docid": 1,
        "g3_doc_unresolved": 0,
        "g3_llm": 0,
    }


def test_stage_counts_g3_doc_unresolved_when_g2_disabled(always_yes_verifier):
    """With G2 ablated, an unresolvable docid is caught by G3's doc guard —
    the exact mechanism the LC ablation needs to attribute."""
    hint = _make_hint("location", "9999", scope_id="scope-a")
    gate = CompositeGate(support_verifier=always_yes_verifier, enabled={"g1", "g3"})
    decision = gate(
        question="What is X?",
        answer="Madrid",
        hints=[hint],
        docs=[{"docid": "5412", "text": "Madrid is the capital."}],
        scope_id="scope-a",
    )
    assert decision.passed is False
    assert decision.stage_counts == {
        "g1_scope": 0,
        "g2_docid": 0,
        "g3_doc_unresolved": 1,
        "g3_llm": 0,
    }
