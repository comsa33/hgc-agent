"""Unit tests for src/hgc/runner.py.

All tests use FakeAgent and FakeJudge — no real Azure API calls.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from hgc.judge import Verdict
from hgc.runner import PhaseResult, PhaseRunner

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAgent:
    """Fake agent that always returns a fixed result dict."""

    def __init__(self, answer: str = "X", tokens: int = 100, n_iters: int = 3) -> None:
        self.answer = answer
        self.tokens = tokens
        self.n_iters = n_iters
        self.call_count = 0

    def run(self, question: str) -> dict:
        self.call_count += 1
        return {
            "answer": self.answer,
            "trajectory": {"thought_0": "thinking", "thought_1": "more", "thought_2": "done"},
            "judgment": True,
            "tokens": self.tokens,
            "wall_time": 0.1,
            "n_iters": self.n_iters,
            "cache_hit": False,
            # HGC-style keys (optional)
            "retrieved_positive_hints": [],
            "retrieved_negative_hints": [],
            "added_hints": [],
        }


class FakeJudge:
    """Fake judge that always returns correct=True with canned reasoning."""

    def judge(self, question: str, gold: str, pred: str) -> Verdict:
        return Verdict(
            correct=True,
            confidence=1.0,
            reasoning="looks good",
            raw_response='{"correct": "yes", "reasoning": "looks good"}',
        )


def _make_dataset(n: int = 3) -> dict:
    queries = [
        {"query_id": str(i), "query": f"What is {i}?", "answer": f"Answer {i}"} for i in range(n)
    ]
    paraphrased = [f"Tell me about {i}?" for i in range(n)]
    return {"queries": queries, "paraphrased": paraphrased}


def _make_runner(tmp_path: Path, dataset: dict | None = None) -> PhaseRunner:
    if dataset is None:
        dataset = _make_dataset()

    def tool_factory(qr: dict) -> list:
        return []

    return PhaseRunner(
        dataset=dataset,
        judge=FakeJudge(),
        out_dir=tmp_path / "results",
        tool_factory=tool_factory,
    )


def _fake_agent_factory(qr: dict, tools: list) -> FakeAgent:
    return FakeAgent()


# ---------------------------------------------------------------------------
# Test: per-query trajectory JSON is written with expected keys
# ---------------------------------------------------------------------------


def test_run_phase_writes_trajectory_json(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    results = runner.run_phase("P0", _fake_agent_factory)

    assert len(results) == 3

    phase_dir = tmp_path / "results" / "P0"
    assert phase_dir.exists()

    for qr in _make_dataset()["queries"]:
        traj_file = phase_dir / f"trajectory_q{qr['query_id']}.json"
        assert traj_file.exists(), f"Missing trajectory file: {traj_file}"

        data = json.loads(traj_file.read_text())
        # Required keys per acceptance criteria
        for key in [
            "query_id",
            "question",
            "gold",
            "phase",
            "pred",
            "judgment_correct",
            "tokens",
            "time_s",
            "n_iters",
            "n_retrieved_positive",
            "n_retrieved_negative",
            "n_added_hints",
            "trajectory",
        ]:
            assert key in data, f"Missing key {key!r} in trajectory JSON"

        assert data["phase"] == "P0"
        assert data["pred"] == "X"
        assert data["judgment_correct"] is True
        assert data["tokens"] == 100
        assert data["n_iters"] == 3


# ---------------------------------------------------------------------------
# Test: resume logic — re-running skips queries with existing trajectory files
# ---------------------------------------------------------------------------


class _CallTrackingAgentFactory:
    """Agent factory that counts how many times an agent was created and run."""

    def __init__(self) -> None:
        self.agents_created: list[FakeAgent] = []

    def __call__(self, qr: dict, tools: list) -> FakeAgent:
        agent = FakeAgent()
        self.agents_created.append(agent)
        return agent

    @property
    def total_run_calls(self) -> int:
        return sum(a.call_count for a in self.agents_created)


def test_resume_skips_existing_trajectories(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    factory1 = _CallTrackingAgentFactory()

    # First run — all 3 queries execute
    runner.run_phase("P0", factory1)
    assert factory1.total_run_calls == 3

    # Second run — all trajectories already exist, so agent.run is NOT called
    factory2 = _CallTrackingAgentFactory()
    results2 = runner.run_phase("P0", factory2)

    assert (
        factory2.total_run_calls == 0
    ), "agent.run should not be called when trajectory files already exist"
    assert len(results2) == 3  # still returns results from disk


# ---------------------------------------------------------------------------
# Test: exception in agent.run is captured in PhaseResult.error, loop continues
# ---------------------------------------------------------------------------


class _RaisingAgent:
    """Agent that raises an exception on run()."""

    def run(self, question: str) -> dict:
        raise RuntimeError("boom")


class _MixedFactory:
    """First query raises, remaining queries succeed."""

    def __call__(self, qr: dict, tools: list):
        if qr["query_id"] == "0":
            return _RaisingAgent()
        return FakeAgent()


def test_exception_captured_and_loop_continues(tmp_path: Path) -> None:
    dataset = _make_dataset(n=3)
    runner = _make_runner(tmp_path, dataset=dataset)

    results = runner.run_phase("P0", _MixedFactory())

    assert len(results) == 3

    # First query should have an error recorded
    first = next(r for r in results if r.query_id == "0")
    assert first.error != "", "Expected error to be recorded for raising agent"
    assert "boom" in first.error

    # Other queries should be successful
    for r in results:
        if r.query_id != "0":
            assert r.error == "", f"Expected no error for query {r.query_id}"
            assert r.pred == "X"


# ---------------------------------------------------------------------------
# Test: write_summary produces correct CSV
# ---------------------------------------------------------------------------


def test_write_summary_produces_correct_csv(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    results = runner.run_phase("P0", _fake_agent_factory)
    csv_path = runner.write_summary("P0", results)

    assert csv_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 3

    expected_columns = {
        "query_id",
        "pred",
        "judgment_correct",
        "containment_correct",
        "tokens",
        "time_s",
        "agent_wall_time",
        "judge_time_s",
        "n_iters",
        "n_retrieved_positive",
        "n_retrieved_negative",
        "n_added_hints",
        "error",
    }
    assert expected_columns == set(reader.fieldnames or [])

    # The runner now records the agent's own span and the judge call
    # separately, because time_s spans both and cannot answer "what did the
    # gate cost".
    for row in rows:
        assert abs(float(row["agent_wall_time"]) - 0.1) < 1e-6
        assert float(row["judge_time_s"]) >= 0.0

    for row in rows:
        assert row["tokens"] == "100"
        assert row["n_iters"] == "3"
        assert row["judgment_correct"] == "True"


# ---------------------------------------------------------------------------
# Test: on_paraphrased flag uses paraphrased questions
# ---------------------------------------------------------------------------


def test_on_paraphrased_uses_paraphrased_questions(tmp_path: Path) -> None:
    dataset = _make_dataset(n=2)
    runner = _make_runner(tmp_path, dataset=dataset)

    runner.run_phase("P3", _fake_agent_factory, on_paraphrased=True)

    phase_dir = tmp_path / "results" / "P3"
    for i in range(2):
        traj_file = phase_dir / f"trajectory_q{i}.json"
        data = json.loads(traj_file.read_text())
        assert data["question"] == f"Tell me about {i}?"


# ---------------------------------------------------------------------------
# Test: trajectory JSON written atomically (no .tmp files left behind)
# ---------------------------------------------------------------------------


def test_atomic_write_no_tmp_files(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    runner.run_phase("P0", _fake_agent_factory)

    phase_dir = tmp_path / "results" / "P0"
    tmp_files = list(phase_dir.glob("*.tmp"))
    assert tmp_files == [], f"Found leftover .tmp files: {tmp_files}"


# ---------------------------------------------------------------------------
# Test: resume re-runs errored trajectories
# ---------------------------------------------------------------------------


def test_resume_reruns_errored_trajectories(tmp_path: Path) -> None:
    """Errored trajectory files (non-empty 'error' field) must be deleted and rerun."""
    runner = _make_runner(tmp_path)

    # First run: produce 3 trajectories
    runner.run_phase("P0", _fake_agent_factory)

    # Corrupt the first trajectory by writing an error into it
    phase_dir = tmp_path / "results" / "P0"
    traj_files = sorted(phase_dir.glob("trajectory_q*.json"))
    assert len(traj_files) == 3
    first_traj = traj_files[0]
    data = json.loads(first_traj.read_text(encoding="utf-8"))
    data["error"] = "Connection error."
    data["pred"] = ""
    data["tokens"] = 0
    first_traj.write_text(json.dumps(data), encoding="utf-8")

    # Second run: should rerun only the errored trajectory
    factory2 = _CallTrackingAgentFactory()
    results2 = runner.run_phase("P0", factory2)

    assert (
        factory2.total_run_calls == 1
    ), "Only the errored trajectory should be re-run, not the clean ones"
    assert len(results2) == 3  # all 3 results still returned

    # The rerun result should have no error
    rerun_result = next(r for r in results2 if r.query_id == data["query_id"])
    assert rerun_result.error == "", "Rerun trajectory should have no error"
    assert rerun_result.pred == "X", "Rerun trajectory should have the agent answer"


def test_resume_skips_clean_trajectories_after_error_fix(tmp_path: Path) -> None:
    """Clean trajectories are still skipped; only errored ones are retried."""
    runner = _make_runner(tmp_path)

    # First run
    runner.run_phase("P0", _fake_agent_factory)

    # Mark all 3 as errored
    phase_dir = tmp_path / "results" / "P0"
    for f in sorted(phase_dir.glob("trajectory_q*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["error"] = "Connection error."
        f.write_text(json.dumps(d), encoding="utf-8")

    # Rerun: all 3 should be retried
    factory2 = _CallTrackingAgentFactory()
    runner.run_phase("P0", factory2)
    assert factory2.total_run_calls == 3, "All 3 errored trajectories should be rerun"


# ---------------------------------------------------------------------------
# Test: PhaseResult dataclass has all expected fields
# ---------------------------------------------------------------------------


def test_phase_result_fields() -> None:
    r = PhaseResult(
        phase="P0",
        query_id="1",
        gold="Answer 1",
        pred="Answer 1",
        judgment_correct=True,
        judgment_reasoning="looks good",
        containment_correct=True,
        tokens=100,
        time_s=0.5,
        n_iters=3,
        n_retrieved_positive=2,
        n_retrieved_negative=1,
        n_added_hints=4,
        error="",
    )
    assert r.phase == "P0"
    assert r.judgment_correct is True
    assert r.n_added_hints == 4
    assert r.error == ""


def test_write_summary_leaves_timing_blank_for_legacy_results(tmp_path: Path) -> None:
    """Rows restored from pre-split trajectories must stay empty, not 0.0.

    None of the trajectories written before agent_wall_time/judge_time_s
    existed carries those keys. Writing 0.0 would make an unmeasured query
    read as an instantaneous one and silently drag any mean downward.
    """
    runner = _make_runner(tmp_path)
    legacy = PhaseResult(
        phase="P0",
        query_id="1",
        gold="Answer 1",
        pred="Answer 1",
        judgment_correct=True,
        judgment_reasoning="",
        containment_correct=True,
        tokens=100,
        time_s=0.5,
        n_iters=3,
        n_retrieved_positive=0,
        n_retrieved_negative=0,
        n_added_hints=0,
        error="",
    )
    assert legacy.agent_wall_time is None
    assert legacy.judge_time_s is None

    csv_path = runner.write_summary("P0", [legacy])
    with csv_path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))

    assert row["agent_wall_time"] == ""
    assert row["judge_time_s"] == ""
    assert row["time_s"] == "0.5"
