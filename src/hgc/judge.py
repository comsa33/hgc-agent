"""LLM-as-Judge module for HGC.

Grades predicted answers against gold answers using the HLE / BrowseComp
semantic-equivalence prompt style (Wei et al. 2025, Phan et al. 2025).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_GRADING_PROMPT_TMPL = """\
You are a grader for a question-answering system.

Question: {question}
Correct answer: {gold_answer}
Predicted answer: {model_answer}

Is the predicted answer semantically equivalent to the correct answer?
Ignore minor formatting differences (whitespace, punctuation, capitalisation,
diacritics) and irrelevant wrapping text; focus on semantic equivalence.
Return strict JSON with exactly these two keys and no other text:

{{"correct": "yes" | "no", "reasoning": "<short explanation, under 300 chars>"}}
"""


@dataclass
class Verdict:
    """Structured result from LLMJudge."""

    correct: bool
    confidence: float
    reasoning: str
    raw_response: str


def _make_default_lm(
    deployment: str | None,
    api_version: str | None,
    endpoint: str | None,
    api_key: str | None,
    temperature: float,
) -> Any:
    """Construct a dspy.LM backed by the Azure GPT-4o judge deployment."""
    import dspy  # deferred so tests can import without dspy installed

    dep = (
        deployment
        or os.environ.get("AZURE_OPENAI_JUDGE_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or "gpt-4o"
    )
    resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
    resolved_endpoint = (endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
    resolved_version = api_version or os.environ.get(
        "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
    )

    return dspy.LM(
        model=f"azure/{dep}",
        api_key=resolved_key,
        api_base=resolved_endpoint,
        api_version=resolved_version,
        temperature=temperature,
        max_tokens=512,
        # Mirror configure_lm / _gate_lm: DSPy caches LM calls on disk by
        # default, and a cached judgment returns in ~2ms while a fresh one
        # takes ~1.5s. That 500x gap lands inside the per-query timing we
        # report, so the judge must be uncached like every other LM here.
        cache=False,
    )


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from *text* if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Split on the opening fence line
        parts = stripped.split("```")
        # parts[0] is empty, parts[1] contains the body (possibly with "json\n" prefix)
        body = parts[1] if len(parts) > 1 else ""
        if body.startswith("json"):
            body = body[4:]
        return body.strip()
    return stripped


def _parse_verdict(raw: str) -> Verdict:
    """Parse LLM output into a Verdict; fall back on any parse failure."""
    text = _strip_fences(raw)
    try:
        data = json.loads(text)
        correct_str = str(data.get("correct", "no")).strip().lower()
        correct = correct_str == "yes"
        reasoning = str(data.get("reasoning", ""))[:499]
        return Verdict(correct=correct, confidence=1.0, reasoning=reasoning, raw_response=raw)
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        logger.warning("LLMJudge: JSON parse failed: %s | raw=%r", exc, raw[:200])
        return Verdict(correct=False, confidence=0.0, reasoning="parse_failure", raw_response=raw)


class LLMJudge:
    """Answer-grading module using an LLM with the HLE / BrowseComp prompt style.

    Usage — three-argument form::

        judge = LLMJudge()
        verdict = judge.judge(question, gold_answer, model_answer)

    Usage — bound form (drop-in replacement for containment judge)::

        judge = LLMJudge()
        judge.bind_gold(question, gold_answer)
        verdict = judge(model_answer)
    """

    def __init__(
        self,
        deployment: str | None = None,
        api_version: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        temperature: float = 0,
        lm: Any = None,
    ) -> None:
        self._deployment = deployment
        self._api_version = api_version
        self._endpoint = endpoint
        self._api_key = api_key
        self._temperature = temperature
        self._lm = lm  # may be None — constructed lazily on first judge() call

        # State for the bound-gold convenience API
        self._bound_question: str | None = None
        self._bound_gold: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lm(self) -> Any:
        if self._lm is None:
            self._lm = _make_default_lm(
                self._deployment,
                self._api_version,
                self._endpoint,
                self._api_key,
                self._temperature,
            )
        return self._lm

    def _call_lm(self, prompt: str) -> str:
        lm = self._get_lm()
        raw = lm(prompt)
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        if hasattr(raw, "completions"):
            raw = raw.completions[0]
        return str(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind_gold(self, question: str, gold_answer: str) -> LLMJudge:
        """Store (question, gold_answer) for use with __call__; returns self for chaining."""
        self._bound_question = question
        self._bound_gold = gold_answer
        return self

    def judge(self, question: str, gold_answer: str, model_answer: str) -> Verdict:
        """Primary grading API.  Returns a Verdict dataclass."""
        prompt = _GRADING_PROMPT_TMPL.format(
            question=question,
            gold_answer=gold_answer,
            model_answer=model_answer,
        )
        try:
            raw = self._call_lm(prompt)
        except Exception as exc:
            logger.warning("LLMJudge: LM call failed: %s", exc)
            return Verdict(correct=False, confidence=0.0, reasoning="lm_error", raw_response="")

        return _parse_verdict(raw)

    def __call__(self, answer: str) -> Verdict:
        """Convenience call using the last-bound (question, gold) pair.

        Raises ValueError if bind_gold() has not been called.
        """
        if self._bound_question is None or self._bound_gold is None:
            raise ValueError("LLMJudge: call bind_gold(question, gold) before using __call__.")
        return self.judge(self._bound_question, self._bound_gold, answer)
