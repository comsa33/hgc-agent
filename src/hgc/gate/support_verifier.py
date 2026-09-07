"""G3: LLM-based support verifier.

Wraps a DSPy :class:`dspy.Predict` with a boolean output field. One-shot
yes/no decision: does this document contain evidence supporting the proposed
answer to this question? Errors degrade to a conservative ``False`` so that
contamination is not accepted on LLM failure.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

_PREDICATES = ("support", "answerhood", "two_stage", "two_stage_doc")


class SupportVerifier:
    """Callable wrapper: ``(question, answer, doc_text) -> bool``.

    The wrapped DSPy predict is lazy-instantiated so that the module imports
    cleanly without requiring a configured LM at construction time; the first
    call triggers ``dspy.Predict`` instantiation under the currently
    configured LM.

    A fast containment precheck fires before the LLM call: if the proposed
    answer (or a short proper-noun-rich span of it) appears as a substring of
    the document text (case-insensitive, whitespace-normalised), the verifier
    accepts without an LLM call. This eliminates BCP-style false rejections
    where the cached extractive answer is literally present in the retrieved
    document but the zero-shot ``dspy.Predict`` judge refused to confirm.

    Parameters
    ----------
    on_error:
        Fallback called with the raised exception when the LLM call fails.
        Default returns False (conservative — reject the cache hit).
    max_doc_chars:
        Truncate ``doc_text`` to this many characters before the prompt.
    containment_min_chars:
        Minimum length of the answer string for containment to count as
        support. Too-short answers (e.g., "yes"/"no") risk spurious matches
        and are forwarded to the LLM check.
    predicate:
        Which question the LLM check asks. ``"support"`` (released default)
        asks whether the document carries evidence consistent with the
        proposed answer. ``"answerhood"`` asks the stricter question of
        whether the document shows the answer to be responsive to the
        query — the two come apart when a poisoned answer is lifted verbatim
        out of the victim's own document, which is grounded but answers
        nothing. ``"two_stage"`` keeps the released grounding check but puts
        an independent eligibility judge in front of it, disqualifying
        answers that do not address the question before grounding is
        considered at all. This mirrors the two-stage protocol the FACTS
        Grounding leaderboard uses, and unlike the other two it also runs
        ahead of the containment fast path, so a verbatim quotation cannot
        skip it. ``"two_stage_doc"`` is the same protocol with the source
        document shown to the eligibility judge as well: on the agentic cell
        the judge that sees only question and answer turns away correct
        hits it cannot confirm from the pair alone, and this variant asks
        whether the document lets it confirm them without also letting a
        quotation from that document back in.
    """

    def __init__(
        self,
        on_error: Callable[[Exception], bool] | None = None,
        max_doc_chars: int = 2500,
        containment_min_chars: int = 5,
        lm: Any | None = None,
        predicate: str = "support",
    ) -> None:
        if predicate not in _PREDICATES:
            raise ValueError(
                f"unknown verifier predicate {predicate!r}; expected one of {sorted(_PREDICATES)}"
            )
        self._predict = None
        self._eligibility = None
        self._on_error = on_error or (lambda _exc: False)
        self._max_doc_chars = max_doc_chars
        self._containment_min_chars = containment_min_chars
        self._predicate = predicate
        # Optional dedicated LM for the G3 verification call. When set, the
        # DSPy predict runs under this LM (e.g. an open-weight verifier)
        # instead of the globally configured backbone LM. None → global LM
        # (released default, unchanged).
        self._lm = lm

    @staticmethod
    def _normalise(s: str) -> str:
        return " ".join(s.lower().split())

    def _containment_pass(self, answer: str, doc_text: str) -> bool:
        ans = self._normalise(answer)
        if len(ans) < self._containment_min_chars:
            return False
        # Check the full document — containment is O(n), no reason to truncate
        # like we do for the LLM prompt (which has a context budget).
        doc = self._normalise(doc_text)
        return ans in doc

    def _eligible(self, question: str, answer: str, doc_text: str = "") -> bool:
        """Stage one: does the answer address the question at all?

        Under ``two_stage_doc`` the judge also receives the (truncated)
        document; under ``two_stage`` it sees the question and answer only.
        """
        if self._eligibility is None:
            self._eligibility = self._build_eligibility()
        kwargs = {"question": question, "proposed_answer": answer}
        if self._predicate == "two_stage_doc":
            kwargs["document"] = doc_text[: self._max_doc_chars]
        try:
            if self._lm is not None:
                import dspy

                ctx = dspy.context(lm=self._lm)
            else:
                ctx = nullcontext()
            with ctx:
                result = self._eligibility(**kwargs)
            return bool(result.addresses_question)
        except Exception as exc:  # noqa: BLE001 — same conservative policy as G3
            return self._on_error(exc)

    def _build(self):
        import dspy

        class _SupportSig(dspy.Signature):
            """Decide if the document supports the proposed answer for the question."""

            question: str = dspy.InputField()
            proposed_answer: str = dspy.InputField()
            document: str = dspy.InputField(desc="Document text, possibly truncated.")
            supports: bool = dspy.OutputField(
                desc=(
                    "True only when the document contains explicit or implicit "
                    "evidence consistent with the proposed answer. A paraphrase "
                    "of the answer still counts as support."
                )
            )

        class _AnswerhoodSig(dspy.Signature):
            """Decide if the document shows the proposed answer to be a correct answer to the question."""

            question: str = dspy.InputField()
            proposed_answer: str = dspy.InputField()
            document: str = dspy.InputField(desc="Document text, possibly truncated.")
            supports: bool = dspy.OutputField(
                desc=(
                    "True only when the document shows the proposed answer to be "
                    "a correct and responsive answer to the question. Text the "
                    "document contains but that does not answer the question is "
                    "not support. A paraphrase of a correct answer still counts."
                )
            )

        sig = _AnswerhoodSig if self._predicate == "answerhood" else _SupportSig
        return dspy.Predict(sig)

    def _build_eligibility(self):
        import dspy

        class _EligibilitySig(dspy.Signature):
            """Decide if the proposed answer addresses what the question asks for."""

            question: str = dspy.InputField()
            proposed_answer: str = dspy.InputField()
            addresses_question: bool = dspy.OutputField(
                desc=(
                    "True only when the proposed answer responds to what the "
                    "question asks. Text on the same topic that answers a "
                    "different question, or states a related fact without "
                    "answering, is False. Judge responsiveness only; whether "
                    "the answer is factually right is decided separately."
                )
            )

        class _EligibilityDocSig(dspy.Signature):
            """Decide if the proposed answer addresses what the question asks for, using the document to understand what the answer refers to."""

            question: str = dspy.InputField()
            proposed_answer: str = dspy.InputField()
            document: str = dspy.InputField(desc="Source document, possibly truncated.")
            addresses_question: bool = dspy.OutputField(
                desc=(
                    "True only when the proposed answer responds to what the "
                    "question asks. Use the document to resolve what the answer "
                    "refers to, not as evidence of responsiveness: text copied "
                    "from the document that answers a different question, or "
                    "states a related fact without answering, is False even "
                    "though it appears in the document. Judge responsiveness "
                    "only; whether the answer is factually right is decided "
                    "separately."
                )
            )

        if self._predicate == "two_stage_doc":
            return dspy.Predict(_EligibilityDocSig)
        return dspy.Predict(_EligibilitySig)

    def __call__(self, question: str, answer: str, doc_text: str) -> bool:
        # Two-stage protocol: eligibility first, ahead of every other path.
        # Putting it before containment is the point — the attack this guards
        # against is a verbatim quotation, which is exactly what containment
        # waves through.
        if self._predicate in ("two_stage", "two_stage_doc") and not self._eligible(
            question, answer, doc_text
        ):
            return False

        # Fast path: cached answer appears verbatim in doc text → accept.
        if self._containment_pass(answer, doc_text):
            return True

        if self._predict is None:
            self._predict = self._build()
        try:
            # Run the verification call under the dedicated gate LM when one is
            # configured (leaving the backbone's globally configured LM
            # untouched); otherwise a nullcontext falls through to the global LM.
            if self._lm is not None:
                import dspy

                ctx = dspy.context(lm=self._lm)
            else:
                ctx = nullcontext()
            with ctx:
                result = self._predict(
                    question=question,
                    proposed_answer=answer,
                    document=doc_text[: self._max_doc_chars],
                )
            return bool(result.supports)
        except Exception as exc:  # noqa: BLE001 — we want to catch any LM error
            return self._on_error(exc)
