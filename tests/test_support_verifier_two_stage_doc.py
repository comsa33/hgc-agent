"""``two_stage_doc``: the eligibility judge sees the document; ``two_stage`` does not.

The judge under ``two_stage`` rejects correct BCP hits it cannot confirm from
the question and answer alone. This variant hands it the document. The tests
pin what each predicate passes to the judge and that both run ahead of the
containment fast path, which is the property the two-stage protocol exists for.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hgc.gate.support_verifier import SupportVerifier

_DOC = "Alpha Labs was founded in 2003. " * 200  # longer than max_doc_chars


class _Judge:
    def __init__(self, verdict: bool):
        self.verdict = verdict
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(addresses_question=self.verdict)


@pytest.mark.parametrize(
    "predicate, expects_document",
    [("two_stage", False), ("two_stage_doc", True)],
)
def test_what_the_eligibility_judge_is_shown(predicate, expects_document):
    v = SupportVerifier(predicate=predicate, max_doc_chars=100)
    judge = _Judge(verdict=False)
    v._eligibility = judge
    # Answer is verbatim in the document, so containment alone would accept.
    assert v("who founded it?", "Alpha Labs was founded in 2003.", _DOC) is False
    assert len(judge.calls) == 1
    call = judge.calls[0]
    assert call["question"] == "who founded it?"
    assert ("document" in call) is expects_document
    if expects_document:
        assert len(call["document"]) == 100  # truncated like the G3 prompt


def test_eligible_answer_then_falls_through_to_containment():
    v = SupportVerifier(predicate="two_stage_doc")
    v._eligibility = _Judge(verdict=True)
    assert v("who founded it?", "Alpha Labs was founded in 2003.", _DOC) is True


def test_released_predicate_never_consults_the_judge():
    v = SupportVerifier(predicate="support")
    judge = _Judge(verdict=False)
    v._eligibility = judge
    assert v("q", "Alpha Labs was founded in 2003.", _DOC) is True
    assert judge.calls == []


def test_unknown_predicate_is_rejected():
    with pytest.raises(ValueError, match="predicate"):
        SupportVerifier(predicate="three_stage")
