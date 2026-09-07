"""Unit tests for hgc.hgc_rag.HGCRAGAgent — four-path coverage with stubs."""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock

import numpy as np

from hgc.gate import CompositeGate
from hgc.hgc_rag import ACRAGAgent, HGCRAGAgent
from hgc.memory import HintRecord


def _stub_embed(text: str) -> np.ndarray:
    h = abs(hash(text)) % 10_000
    v = np.array([h, h + 1, h + 2], dtype=np.float32)
    return v / np.linalg.norm(v)


class _ACStub:
    """Minimal AnswerCacheAgent-shaped stub: HGCRAGAgent reads ``_cache``,
    ``_sim_threshold``, ``_embedder`` directly (bypassing AC's ReAct fallback).
    """

    def __init__(self, hit: bool, answer: str = "Madrid"):
        self._embedder = _stub_embed
        self._sim_threshold = 0.85
        # When hit=True, seed cache with the same embedding the question will
        # produce; when hit=False, leave cache empty so no entry can match.
        if hit:
            self._cache = [(_stub_embed("q"), answer)]
        else:
            self._cache = []


def _ac_result(hit: bool, answer: str = "Madrid"):
    return {
        "answer": answer,
        "trajectory": {},
        "tokens": 0,
        "wall_time": 0.01,
        "n_iters": 0,
        "cache_hit": hit,
    }


def _rag_result(answer: str = "Barcelona", tokens: int = 3000):
    return {
        "answer": answer,
        "trajectory": {"retrieved_docids": ["9999"]},
        "tokens": tokens,
        "wall_time": 0.2,
        "n_iters": 1,
        "cache_hit": False,
    }


def _make_hint(content: str, hint_type: str = "location", scope_id: str = "scope-a"):
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
    h.scope_id = scope_id  # type: ignore[attr-defined]
    return h


def _params(scope_id: str = "scope-a") -> dict:
    return {
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.3,
        "theta_pos": 0.3,
        "theta_neg": 0.6,
        "scope_id": scope_id,
    }


def _make_agent(ac, rag, store, gate=None, params=None):
    return HGCRAGAgent(
        ac_agent=ac,
        rag_backbone=rag,
        store=store,
        embedder=_stub_embed,
        hint_retrieval_params=params if params is not None else _params(),
        gate=gate if gate is not None else CompositeGate(),
        top_k_hints=3,
    )


def test_cache_miss_runs_rag_fallback():
    ac = _ACStub(hit=False)
    rag = MagicMock()
    rag.run.return_value = _rag_result(answer="Barcelona")
    store = MagicMock()
    store.search.return_value = []

    agent = _make_agent(ac, rag, store)
    result = agent.run("q", docs=[{"docid": "5412", "text": "doc"}])

    assert result["path"] == "cache_miss"
    assert result["cache_hit"] is False
    assert result["answer"] == "Barcelona"
    rag.run.assert_called_once()


def test_cache_hit_no_hint_trusts_ac():
    ac = _ACStub(hit=True, answer="Madrid")
    rag = MagicMock()
    rag.run.return_value = _rag_result()
    store = MagicMock()
    store.search.return_value = []

    agent = _make_agent(ac, rag, store)
    result = agent.run("q", docs=[{"docid": "5412", "text": "doc"}])

    assert result["path"] == "cache_hit_no_hint"
    assert result["cache_hit"] is True
    assert result["answer"] == "Madrid"
    assert result["tokens"] == 0
    rag.run.assert_not_called()


def test_cache_verified_when_gate_passes():
    ac = _ACStub(hit=True, answer="Madrid")
    rag = MagicMock()
    rag.run.return_value = _rag_result()
    hint = _make_hint("5412")
    store = MagicMock()
    store.search.return_value = [hint]
    gate = CompositeGate(support_verifier=lambda q, a, d: True)

    agent = _make_agent(ac, rag, store, gate=gate)
    result = agent.run(
        "q",
        docs=[{"docid": "5412", "text": "Madrid is the capital of Spain."}],
    )

    assert result["path"] == "cache_verified"
    assert result["cache_hit"] is True
    assert result["answer"] == "Madrid"
    rag.run.assert_not_called()


def _stubbed_decision(passed: bool, evidence_hint=None):
    from hgc.gate import GateDecision

    return GateDecision(
        passed=passed,
        reason="verified" if passed else "no_hint_passed_all_components",
        evidence_hint=evidence_hint,
    )


def test_cache_verified_tokens_include_gate_rag(monkeypatch):
    """HGCRAGAgent: gate LLM tokens must appear in cache_verified result."""
    import dspy

    class _StubLM:
        history: list[dict] = []

    stub_lm = _StubLM()
    monkeypatch.setattr(dspy.settings, "lm", stub_lm, raising=False)

    ac = _ACStub(hit=True, answer="Madrid")
    rag = MagicMock()
    rag.run.return_value = _rag_result()
    hint = _make_hint("5412")
    store = MagicMock()
    store.search.return_value = [hint]

    def _gate(question, answer, hints, docs, scope_id):
        stub_lm.history.append({"usage": {"total_tokens": 873}})
        return _stubbed_decision(passed=True, evidence_hint=hints[0])

    agent = _make_agent(ac, rag, store, gate=_gate)
    result = agent.run("q", docs=[{"docid": "5412", "text": "Madrid is the capital."}])

    assert result["path"] == "cache_verified"
    assert result["tokens"] == 873
    assert result["gate_tokens"] == 873


def test_cache_fallback_tokens_sum_gate_and_rag(monkeypatch):
    """HGCRAGAgent: cache_fallback must sum gate + RAG backbone tokens."""
    import dspy

    class _StubLM:
        history: list[dict] = []

    stub_lm = _StubLM()
    monkeypatch.setattr(dspy.settings, "lm", stub_lm, raising=False)

    ac = _ACStub(hit=True, answer="Madrid")
    rag = MagicMock()

    def _rag_run(question, docs=None):
        # Real RAGBackbone uses LangChain, which does not write to dspy.lm.history;
        # RAG tokens reach HGCRAGAgent via rag_result["tokens"] only.
        return _rag_result(answer="Barcelona", tokens=3000)

    rag.run.side_effect = _rag_run
    hint = _make_hint("5412")
    store = MagicMock()
    store.search.return_value = [hint]

    def _gate(question, answer, hints, docs, scope_id):
        stub_lm.history.append({"usage": {"total_tokens": 700}})
        return _stubbed_decision(passed=False)

    agent = _make_agent(ac, rag, store, gate=_gate)
    result = agent.run("q", docs=[{"docid": "5412", "text": "Paris is the capital of France."}])

    assert result["path"] == "cache_fallback"
    assert result["tokens"] == 700 + 3000
    assert result["gate_tokens"] == 700


def test_ac_rag_cache_hit_returns_cached_no_rag():
    ac = _ACStub(hit=True, answer="Madrid")
    rag = MagicMock()
    rag.run.return_value = _rag_result()

    agent = ACRAGAgent(ac_agent=ac, rag_backbone=rag)
    result = agent.run("q", docs=[{"docid": "5412", "text": "doc"}])

    assert result["path"] == "cache_hit"
    assert result["cache_hit"] is True
    assert result["answer"] == "Madrid"
    assert result["tokens"] == 0
    rag.run.assert_not_called()


def test_ac_rag_cache_miss_runs_rag_and_sums_tokens():
    ac = _ACStub(hit=False)
    rag = MagicMock()
    rag.run.return_value = _rag_result(answer="Barcelona", tokens=2100)

    agent = ACRAGAgent(ac_agent=ac, rag_backbone=rag)
    result = agent.run("q", docs=[{"docid": "5412", "text": "doc"}])

    assert result["path"] == "cache_miss"
    assert result["cache_hit"] is False
    assert result["answer"] == "Barcelona"
    assert result["tokens"] == 2100
    rag.run.assert_called_once()


def test_cache_fallback_on_gate_rejection_runs_rag():
    ac = _ACStub(hit=True, answer="Madrid")
    rag = MagicMock()
    rag.run.return_value = _rag_result(answer="Barcelona")
    hint = _make_hint("5412")
    store = MagicMock()
    store.search.return_value = [hint]
    gate = CompositeGate(support_verifier=lambda q, a, d: False)

    agent = _make_agent(ac, rag, store, gate=gate)
    result = agent.run(
        "q",
        docs=[{"docid": "5412", "text": "Paris is the capital of France."}],
    )

    assert result["path"] == "cache_fallback"
    assert result["cache_hit"] is False
    assert result["answer"] == "Barcelona"
    rag.run.assert_called_once()
    assert "gate_reject_reason" in result
