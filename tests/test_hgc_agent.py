"""Unit tests for hgc.hgc_agent.HGCAgent — four-path coverage with stubs.

All tests use MagicMock for AC/Core/Store so no API calls are made.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock

import numpy as np

from hgc.gate import CompositeGate
from hgc.hgc_agent import HGCAgent
from hgc.memory import HintRecord


def _stub_embed(text: str) -> np.ndarray:
    h = abs(hash(text)) % 10_000
    v = np.array([h, h + 1, h + 2], dtype=np.float32)
    return v / np.linalg.norm(v)


def _ac_result(hit: bool, answer: str = "Madrid") -> dict:
    return {
        "answer": answer,
        "trajectory": {},
        "judgment": True,
        "tokens": 0,
        "wall_time": 0.01,
        "n_iters": 0,
        "cache_hit": hit,
    }


def _core_result(answer: str = "Barcelona", tokens: int = 5000, n_iters: int = 8) -> dict:
    return {
        "answer": answer,
        "trajectory": {"thought_0": "..."},
        "judgment": False,
        "tokens": tokens,
        "wall_time": 0.5,
        "n_iters": n_iters,
        "cache_hit": False,
    }


def _make_hint(content: str, hint_type: str = "location", scope_id: str = "scope-a") -> HintRecord:
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


def _make_core_stub(answer: str = "Barcelona", scope_id: str = "scope-a"):
    core = MagicMock()
    core.run.return_value = _core_result(answer=answer)
    core._alpha = 1.0
    core._beta = 0.5
    core._gamma = 0.3
    core._theta_pos = 0.3
    core._theta_neg = 0.6
    core._scope_id = scope_id
    return core


def _make_ac_stub(hit: bool, answer: str = "Madrid"):
    ac = MagicMock()
    ac.run.return_value = _ac_result(hit=hit, answer=answer)
    return ac


def _make_store_stub(location_hints: list[HintRecord]):
    store = MagicMock()
    store.search.return_value = location_hints
    return store


def _make_agent(ac, core, store, gate=None):
    return HGCAgent(
        ac_agent=ac,
        core_agent=core,
        store=store,
        embedder=_stub_embed,
        gate=gate if gate is not None else CompositeGate(),
        top_k_hints=3,
    )


# ---------------------------------------------------------------------------
# Four primary paths
# ---------------------------------------------------------------------------


def test_cache_miss_path(monkeypatch):
    """AC miss -> HGCCore runs, path=cache_miss, cache_hit=False."""
    import dspy

    class _StubLM:
        history: list[dict] = []

    stub_lm = _StubLM()
    monkeypatch.setattr(dspy.settings, "lm", stub_lm, raising=False)

    ac = _make_ac_stub(hit=False)
    core = _make_core_stub(answer="Barcelona")

    def _core_run(question, docs=None):
        stub_lm.history.append({"usage": {"total_tokens": 5000}})
        return _core_result(answer="Barcelona", tokens=5000)

    core.run.side_effect = _core_run
    store = _make_store_stub([])

    agent = _make_agent(ac, core, store)
    result = agent.run("What is X?", docs=[])

    assert result["path"] == "cache_miss"
    assert result["cache_hit"] is False
    assert result["answer"] == "Barcelona"
    assert result["tokens"] == 5000
    assert result["gate_tokens"] == 0
    core.run.assert_called_once()


def test_cache_hit_no_hint_path():
    """AC hit, store returns zero location hints -> trust AC, path=cache_hit_no_hint."""
    ac = _make_ac_stub(hit=True, answer="Madrid")
    core = _make_core_stub()
    store = _make_store_stub([])

    agent = _make_agent(ac, core, store)
    result = agent.run("What is X?", docs=[])

    assert result["path"] == "cache_hit_no_hint"
    assert result["cache_hit"] is True
    assert result["answer"] == "Madrid"
    assert result["tokens"] == 0
    assert result["n_iters"] == 0
    core.run.assert_not_called()


def test_cache_verified_path_with_passing_gate():
    """AC hit + location hint + passing gate -> path=cache_verified."""
    ac = _make_ac_stub(hit=True, answer="Madrid")
    core = _make_core_stub()
    hint = _make_hint("5412", scope_id="scope-a")
    store = _make_store_stub([hint])
    gate = CompositeGate(support_verifier=lambda q, a, d: True)

    agent = _make_agent(ac, core, store, gate=gate)
    result = agent.run(
        "What is X?",
        docs=[{"docid": "5412", "text": "Madrid is the capital."}],
    )

    assert result["path"] == "cache_verified"
    assert result["cache_hit"] is True
    assert result["answer"] == "Madrid"
    assert result["tokens"] == 0
    core.run.assert_not_called()


def test_cache_fallback_path_when_gate_rejects():
    """AC hit + location hint + rejecting gate -> core runs, path=cache_fallback."""
    ac = _make_ac_stub(hit=True, answer="Madrid")
    core = _make_core_stub(answer="Barcelona")
    hint = _make_hint("5412", scope_id="scope-a")
    store = _make_store_stub([hint])
    gate = CompositeGate(support_verifier=lambda q, a, d: False)  # reject

    agent = _make_agent(ac, core, store, gate=gate)
    result = agent.run(
        "What is X?",
        docs=[{"docid": "5412", "text": "Paris is the capital of France."}],
    )

    assert result["path"] == "cache_fallback"
    assert result["cache_hit"] is False
    assert result["answer"] == "Barcelona"
    core.run.assert_called_once()
    assert "gate_reject_reason" in result


# ---------------------------------------------------------------------------
# Gate-specific flows
# ---------------------------------------------------------------------------


def test_cache_fallback_when_docid_absent():
    """Hint references a docid not in the current doc pool -> gate rejects via G2."""
    ac = _make_ac_stub(hit=True, answer="Madrid")
    core = _make_core_stub()
    hint = _make_hint("5412", scope_id="scope-a")
    store = _make_store_stub([hint])

    agent = _make_agent(ac, core, store)  # default full gate (G1+G2+G3)
    result = agent.run(
        "What is X?",
        docs=[{"docid": "9999", "text": "unrelated"}],
    )

    assert result["path"] == "cache_fallback"
    core.run.assert_called_once()


def test_cache_fallback_when_scope_mismatch():
    """Hint belongs to a different scope -> gate rejects via G1."""
    ac = _make_ac_stub(hit=True, answer="Madrid")
    core = _make_core_stub(scope_id="scope-a")
    hint = _make_hint("5412", scope_id="other-scope")
    store = _make_store_stub([hint])

    agent = _make_agent(ac, core, store)
    result = agent.run(
        "What is X?",
        docs=[{"docid": "5412", "text": "Madrid is the capital."}],
    )

    # Scope filter drops the hint -> no location hints -> cache_hit_no_hint (trust AC).
    # This documents the intended behaviour: the scope filter runs inside the gate,
    # so a scope-mismatched hint appears to the gate as "no candidate" and trips
    # the 'no_location_hint_in_scope' reason; the agent then falls back to core.
    assert result["path"] == "cache_fallback"
    assert result["gate_reject_reason"] == "no_location_hint_in_scope"


def test_evidence_hint_reported_on_pass():
    """On cache_verified the evidence_hint id is preserved in the result."""
    ac = _make_ac_stub(hit=True, answer="Madrid")
    core = _make_core_stub()
    hint = _make_hint("5412", scope_id="scope-a")
    store = _make_store_stub([hint])
    gate = CompositeGate(support_verifier=lambda q, a, d: True)

    agent = _make_agent(ac, core, store, gate=gate)
    result = agent.run(
        "What is X?",
        docs=[{"docid": "5412", "text": "Madrid is the capital."}],
    )

    assert result["gate_evidence"] == hint.hint_id


def test_cache_verified_tokens_include_gate(monkeypatch):
    """Gate LLM calls made between AC and core must land in the returned tokens.

    Regression: pre-fix, cache_verified/cache_hit_no_hint hard-coded tokens=0,
    silently dropping the G3 SupportVerifier LLM cost when containment precheck
    missed.
    """
    import dspy

    class _StubLM:
        history: list[dict] = []

    stub_lm = _StubLM()
    monkeypatch.setattr(dspy.settings, "lm", stub_lm, raising=False)

    ac = _make_ac_stub(hit=True, answer="Madrid")
    core = _make_core_stub()
    hint = _make_hint("5412", scope_id="scope-a")
    store = _make_store_stub([hint])

    def _gate_with_fake_llm_call(question, answer, hints, docs, scope_id):
        # Simulate G3 LLM call by appending to lm.history between AC and result.
        stub_lm.history.append({"usage": {"total_tokens": 873}})
        return _stubbed_decision(passed=True, evidence_hint=hints[0])

    agent = _make_agent(ac, core, store, gate=_gate_with_fake_llm_call)
    result = agent.run(
        "What is X?",
        docs=[{"docid": "5412", "text": "Madrid is the capital."}],
    )

    assert result["path"] == "cache_verified"
    assert result["tokens"] == 873
    assert result["gate_tokens"] == 873


def test_cache_fallback_tokens_sum_gate_and_core(monkeypatch):
    """cache_fallback must sum G3 tokens AND core backbone tokens."""
    import dspy

    class _StubLM:
        history: list[dict] = []

    stub_lm = _StubLM()
    monkeypatch.setattr(dspy.settings, "lm", stub_lm, raising=False)

    ac = _make_ac_stub(hit=True, answer="Madrid")
    # core_stub records a fake core-side LLM call into the same history.
    core = _make_core_stub(answer="Barcelona")

    def _core_run(question, docs=None):
        stub_lm.history.append({"usage": {"total_tokens": 5000}})
        return _core_result(answer="Barcelona", tokens=5000)

    core.run.side_effect = _core_run

    hint = _make_hint("5412", scope_id="scope-a")
    store = _make_store_stub([hint])

    def _gate_rejecting_with_llm(question, answer, hints, docs, scope_id):
        stub_lm.history.append({"usage": {"total_tokens": 700}})
        return _stubbed_decision(passed=False)

    agent = _make_agent(ac, core, store, gate=_gate_rejecting_with_llm)
    result = agent.run(
        "What is X?",
        docs=[{"docid": "5412", "text": "unrelated"}],
    )

    assert result["path"] == "cache_fallback"
    assert result["tokens"] == 700 + 5000
    assert result["gate_tokens"] == 700


def _stubbed_decision(passed: bool, evidence_hint=None):
    from hgc.gate import GateDecision

    return GateDecision(
        passed=passed,
        reason="verified" if passed else "no_hint_passed_all_components",
        evidence_hint=evidence_hint,
    )


def test_gate_ablation_g3_only_accepts_on_support_alone():
    """Gate configured with only G3 active still gates on the verifier alone."""
    ac = _make_ac_stub(hit=True, answer="Madrid")
    core = _make_core_stub()
    hint = _make_hint("5412", scope_id="wrong-scope")  # G1 would reject
    store = _make_store_stub([hint])
    gate = CompositeGate(support_verifier=lambda q, a, d: True, enabled={"g3"})

    agent = _make_agent(ac, core, store, gate=gate)
    result = agent.run(
        "What is X?",
        docs=[{"docid": "5412", "text": "Madrid is the capital."}],
    )

    # G1 off => scope mismatch does not drop the hint; G3 yes => verified.
    assert result["path"] == "cache_verified"
