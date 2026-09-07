"""Unit tests for src/hgc/extraction.py.

All tests use a FakeLM — no real API calls are made.
"""

from __future__ import annotations

import json

from hgc.extraction import HintExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_RESPONSE = json.dumps(
    {
        "location": [{"content": "docid=5412", "content_meta": {"docid": "5412"}}],
        "entity": [{"content": "Queen Arwa University", "content_meta": {}}],
        "strategy": [
            {
                "content": "search_documents('Routledge 2018')",
                "content_meta": {"keyword": "Routledge 2018"},
            }
        ],
    }
)

_SAMPLE_TRAJECTORY = {
    "thought_0": "I should search for the university.",
    "tool_name_0": "search_documents",
    "tool_args_0": {"keyword": "Routledge 2018"},
    "observation_0": [{"docid": "5412", "snippet": "...Routledge published in 2018..."}],
    "thought_1": "Found docid 5412. Let me read it.",
    "tool_name_1": "get_document_snippet",
    "tool_args_1": {"docid": "5412", "char_start": 0, "char_len": 500},
    "observation_1": "Queen Arwa University was founded in ...",
}


class FakeLM:
    """Returns a predetermined string regardless of the prompt."""

    def __init__(self, response: str) -> None:
        self._response = response

    def __call__(self, prompt: str) -> list[str]:  # noqa: ARG002
        return [self._response]


class MalformedLM(FakeLM):
    """Returns invalid JSON."""

    def __init__(self) -> None:
        super().__init__("this is not json {{{")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHintExtractorShape:
    """extract() returns the correct dict shape."""

    def test_returns_three_keys(self):
        extractor = HintExtractor(lm=FakeLM(_GOOD_RESPONSE))
        result = extractor.extract(
            "Who founded Queen Arwa University?", _SAMPLE_TRAJECTORY, correct=True
        )
        assert set(result.keys()) == {"location", "entity", "strategy"}

    def test_location_parsed(self):
        extractor = HintExtractor(lm=FakeLM(_GOOD_RESPONSE))
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        assert len(result["location"]) == 1
        assert result["location"][0]["content"] == "docid=5412"

    def test_entity_parsed(self):
        extractor = HintExtractor(lm=FakeLM(_GOOD_RESPONSE))
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        assert len(result["entity"]) == 1
        assert "Queen Arwa University" in result["entity"][0]["content"]

    def test_strategy_parsed(self):
        extractor = HintExtractor(lm=FakeLM(_GOOD_RESPONSE))
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        assert len(result["strategy"]) == 1
        assert "Routledge 2018" in result["strategy"][0]["content"]

    def test_content_meta_is_dict(self):
        extractor = HintExtractor(lm=FakeLM(_GOOD_RESPONSE))
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        for key in ("location", "entity", "strategy"):
            for item in result[key]:
                assert isinstance(item["content_meta"], dict)


class TestPolarity:
    """Polarity handling for correct / incorrect trajectories."""

    def test_correct_true_no_negative_polarity(self):
        extractor = HintExtractor(lm=FakeLM(_GOOD_RESPONSE))
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        for key in ("location", "entity", "strategy"):
            for item in result[key]:
                assert item["content_meta"].get("polarity") != "negative"

    def test_correct_false_sets_negative_polarity(self):
        extractor = HintExtractor(lm=FakeLM(_GOOD_RESPONSE))
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=False)
        # Every hint item should carry polarity=negative in content_meta
        all_items = result["location"] + result["entity"] + result["strategy"]
        assert len(all_items) > 0, "Expected at least one hint"
        for item in all_items:
            assert (
                item["content_meta"].get("polarity") == "negative"
            ), f"Expected polarity=negative, got: {item['content_meta']}"

    def test_correct_true_preserves_existing_meta(self):
        extractor = HintExtractor(lm=FakeLM(_GOOD_RESPONSE))
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        # docid meta should be preserved
        loc = result["location"][0]
        assert loc["content_meta"].get("docid") == "5412"


class TestMalformedJSON:
    """Falls back to empty lists on parse failure."""

    def test_malformed_returns_empty_lists(self):
        extractor = HintExtractor(lm=MalformedLM())
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        assert result == {"location": [], "entity": [], "strategy": []}

    def test_malformed_does_not_raise(self):
        extractor = HintExtractor(lm=MalformedLM())
        # Should not raise any exception
        extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=False)

    def test_empty_trajectory(self):
        extractor = HintExtractor(lm=FakeLM(_GOOD_RESPONSE))
        result = extractor.extract("test query", {}, correct=True)
        assert set(result.keys()) == {"location", "entity", "strategy"}

    def test_lm_exception_returns_empty(self):
        class ErrorLM:
            def __call__(self, prompt):
                raise RuntimeError("LM unavailable")

        extractor = HintExtractor(lm=ErrorLM())
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        assert result == {"location": [], "entity": [], "strategy": []}


class TestMarkdownFences:
    """Handles LLM responses wrapped in markdown code fences."""

    def test_strips_json_code_fence(self):
        fenced = f"```json\n{_GOOD_RESPONSE}\n```"
        extractor = HintExtractor(lm=FakeLM(fenced))
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        assert len(result["location"]) == 1

    def test_strips_plain_code_fence(self):
        fenced = f"```\n{_GOOD_RESPONSE}\n```"
        extractor = HintExtractor(lm=FakeLM(fenced))
        result = extractor.extract("test query", _SAMPLE_TRAJECTORY, correct=True)
        assert len(result["location"]) == 1
