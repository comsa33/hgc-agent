"""AC-LC baseline: the cache is read directly, never through AnswerCacheAgent.run().

AnswerCacheAgent.run() falls back to a tool-less ReAct rollout on a miss and
judges it. The long-context AC baseline then discards that answer and calls the
long-context backbone instead, so every miss used to pay for a wasted rollout
and carried its latency into the reported per-query time. ACRAGAgent fixed this
for the RAG cells in Round 5; these tests pin the same behaviour for the LC
cells and fail loudly if the double call comes back.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import hgc.baselines as baselines
import hgc.factories as factories
import hgc.longctx_backbone as longctx_backbone


class _StubEmbedder:
    """Two orthogonal directions so cache hits are controllable by keyword."""

    def embed(self, text: str) -> np.ndarray:
        vec = np.array([1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0], dtype=np.float32)
        return vec / np.linalg.norm(vec)


class _StubBackbone:
    """Stands in for LongCtxBackbone; counts how often it is invoked."""

    calls = 0

    def __init__(self, llm=None):  # noqa: ARG002
        pass

    def run(self, question: str, docs=None) -> dict:  # noqa: ARG002
        type(self).calls += 1
        return {
            "answer": "backbone answer",
            "trajectory": {},
            "tokens": 42,
            "wall_time": 0.01,
            "n_iters": 1,
        }


@pytest.fixture
def lc_env(monkeypatch):
    """Swap the long-context backbone for a stub and make the real
    AnswerCacheAgent.run() an immediate test failure."""
    _StubBackbone.calls = 0
    monkeypatch.setattr(longctx_backbone, "LongCtxBackbone", _StubBackbone)
    monkeypatch.setattr(factories, "_make_longctx_llm", lambda: None)

    def _forbidden(self, question):  # noqa: ARG001
        raise AssertionError("AnswerCacheAgent.run() was called — the AC-LC double-call regressed")

    monkeypatch.setattr(baselines.AnswerCacheAgent, "run", _forbidden)
    return _StubBackbone


def _write_p1_trajectory(dirpath: Path, qid: str, question: str, pred: str, correct: bool) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"trajectory_q{qid}.json").write_text(
        json.dumps(
            {"query_id": qid, "question": question, "pred": pred, "judgment_correct": correct}
        ),
        encoding="utf-8",
    )


def test_cache_miss_calls_backbone_exactly_once(lc_env, tmp_path):
    factory = factories.build_ac_lc_factory_financebench(_StubEmbedder(), tmp_path / "empty")
    agent = factory({"answer": "backbone answer", "docs": []}, [])

    result = agent.run("alpha question")

    assert lc_env.calls == 1
    assert result["cache_hit"] is False
    assert result["path"] == "cache_miss"
    assert result["tokens"] == 42
    assert result["wall_time"] > 0


def test_cache_hit_serves_seeded_answer_without_touching_backbone(lc_env, tmp_path):
    p1 = tmp_path / "P1-LC"
    _write_p1_trajectory(p1, "1", "alpha question", "seeded alpha answer", correct=True)

    factory = factories.build_ac_lc_factory_financebench(_StubEmbedder(), p1)
    agent = factory({"answer": "seeded alpha answer", "docs": []}, [])

    result = agent.run("alpha question, rephrased")

    assert lc_env.calls == 0
    assert result["cache_hit"] is True
    assert result["path"] == "cache_hit"
    assert result["answer"] == "seeded alpha answer"
    assert result["tokens"] == 0
    assert result["judgment_correct"] is True


def test_judge_incorrect_p1_answers_are_not_seeded(lc_env, tmp_path):
    p1 = tmp_path / "P1-LC"
    _write_p1_trajectory(p1, "1", "alpha question", "wrong alpha answer", correct=False)

    factory = factories.build_ac_lc_factory_financebench(_StubEmbedder(), p1)
    agent = factory({"answer": "backbone answer", "docs": []}, [])

    result = agent.run("alpha question")

    assert lc_env.calls == 1
    assert result["cache_hit"] is False
