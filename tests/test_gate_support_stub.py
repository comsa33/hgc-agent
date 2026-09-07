"""Unit tests for hgc.gate.support_verifier.SupportVerifier (G3).

All tests monkeypatch the DSPy Predict so no LM calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace

from hgc.gate import SupportVerifier


def _patch_predict(monkeypatch, returns: bool | Exception):
    """Replace the DSPy Predict call with a stub that returns or raises."""

    def stub_predict(**kwargs):  # noqa: ARG001
        if isinstance(returns, Exception):
            raise returns
        return SimpleNamespace(supports=returns)

    def _build_stub(self):  # noqa: ARG001
        return stub_predict

    monkeypatch.setattr(SupportVerifier, "_build", _build_stub)


def test_yes_decision(monkeypatch):
    _patch_predict(monkeypatch, True)
    verifier = SupportVerifier()
    assert verifier("What is X?", "Madrid", "Madrid is a city in Spain.") is True


def test_no_decision(monkeypatch):
    _patch_predict(monkeypatch, False)
    verifier = SupportVerifier()
    assert verifier("What is X?", "Barcelona", "Madrid is a city in Spain.") is False


def test_llm_error_returns_false_by_default(monkeypatch):
    _patch_predict(monkeypatch, RuntimeError("LM failure"))
    verifier = SupportVerifier()
    assert verifier("q", "a", "doc") is False


def test_custom_on_error_callback(monkeypatch):
    _patch_predict(monkeypatch, RuntimeError("boom"))
    called = {}

    def on_err(exc):
        called["exc"] = exc
        return True  # test explicit True fallback

    verifier = SupportVerifier(on_error=on_err)
    assert verifier("q", "a", "doc") is True
    assert isinstance(called["exc"], RuntimeError)


def test_doc_truncation(monkeypatch):
    captured = {}

    def stub_predict(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(supports=True)

    def _build_stub(self):  # noqa: ARG001
        return stub_predict

    monkeypatch.setattr(SupportVerifier, "_build", _build_stub)
    verifier = SupportVerifier(max_doc_chars=10)
    long_doc = "x" * 500
    verifier("q", "a", long_doc)
    assert captured["document"] == "x" * 10


def test_unknown_predicate_is_rejected():
    """A typo in HGC_VERIFIER_PREDICATE should fail loudly, not fall back."""
    import pytest

    with pytest.raises(ValueError, match="unknown verifier predicate"):
        SupportVerifier(predicate="grounded")


def test_predicate_selects_a_different_question():
    """The two predicates must reach the LM as genuinely different prompts.

    Built without stubbing (signature construction makes no LM call) so the
    test would catch a branch that silently returns the released signature.
    """
    support = SupportVerifier()._build().signature
    answerhood = SupportVerifier(predicate="answerhood")._build().signature

    assert "supports the proposed answer" in support.instructions
    assert "correct answer to the question" in answerhood.instructions
    assert support.instructions != answerhood.instructions

    support_desc = support.output_fields["supports"].json_schema_extra["desc"]
    answerhood_desc = answerhood.output_fields["supports"].json_schema_extra["desc"]
    assert "consistent with" in support_desc
    assert "does not answer the question is not support" in answerhood_desc


def _patch_two_stage(monkeypatch, eligible: bool, supports: bool):
    """Stub both stages so no LM call is made."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        SupportVerifier, "_build_eligibility",
        lambda self: (lambda **kw: SimpleNamespace(addresses_question=eligible)),
    )
    monkeypatch.setattr(
        SupportVerifier, "_build",
        lambda self: (lambda **kw: SimpleNamespace(supports=supports)),
    )


def test_two_stage_rejects_ineligible_even_when_grounded(monkeypatch):
    """The whole point: a verbatim quotation must not skip the eligibility judge."""
    _patch_two_stage(monkeypatch, eligible=False, supports=True)
    v = SupportVerifier(predicate="two_stage")
    # Answer appears verbatim in the document, so containment would accept it.
    assert v("What gathered the data?", "the three phase dialog", "we used the three phase dialog") is False


def test_two_stage_accepts_when_both_stages_pass(monkeypatch):
    _patch_two_stage(monkeypatch, eligible=True, supports=True)
    v = SupportVerifier(predicate="two_stage")
    assert v("q", "Madrid", "Madrid is the capital.") is True


def test_two_stage_still_defers_to_grounding(monkeypatch):
    """Eligible but ungrounded must still be rejected by stage two."""
    _patch_two_stage(monkeypatch, eligible=True, supports=False)
    v = SupportVerifier(predicate="two_stage")
    assert v("q", "Barcelona", "Madrid is the capital.") is False


def test_single_stage_predicates_never_call_eligibility(monkeypatch):
    called = {"n": 0}

    def boom(self):
        called["n"] += 1
        return lambda **kw: None

    monkeypatch.setattr(SupportVerifier, "_build_eligibility", boom)
    _patch_predict(monkeypatch, True)
    SupportVerifier()("q", "a", "doc")
    SupportVerifier(predicate="answerhood")("q", "a", "doc")
    assert called["n"] == 0
