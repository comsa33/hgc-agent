"""Phase runner orchestrator for HGC experiments.

Implements US-013: executes each experimental phase (P0/P0'/P0-RAG/P1/P2/P3/
P3-AC/P3-TC/P3-M0/P3-RAG/P4) on the query dataset, persisting per-query
trajectory JSONs and summary CSVs with resume support.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PhaseResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Per-query result record for a single experimental phase."""

    phase: str
    query_id: str
    gold: str
    pred: str
    judgment_correct: bool
    judgment_reasoning: str
    containment_correct: bool
    tokens: int
    time_s: float
    n_iters: int
    n_retrieved_positive: int
    n_retrieved_negative: int
    n_added_hints: int
    error: str = ""
    # Timing split out of ``time_s``. ``time_s`` spans agent construction
    # through the judge verdict, so it cannot answer "what did the gate cost".
    # These two are None on rows restored from trajectories written before the
    # split existed — nullable, not 0.0, so absent never reads as instant.
    agent_wall_time: float | None = None
    judge_time_s: float | None = None


def _opt_float(value: Any) -> float | None:
    """Coerce to float, or None when the value is absent or unparseable.

    Used for the timing fields, which are missing from every trajectory
    written before the agent/judge split existed. Those rows must stay empty
    rather than collapse to 0.0, which would read as a zero-latency query.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Seeding helpers for baseline agents (P3-AC, P3-TC, P3-M0)
# ---------------------------------------------------------------------------


def seed_answer_cache_from_p1(
    ac_agent: Any,
    p1_trajectories: list[Path],
    embedder: Callable[[str], Any],
) -> int:
    """Seed an AnswerCacheAgent with (question_embedding, answer) pairs from P1 trajectories.

    Reads each P1 trajectory JSON, extracts (question, pred) and seeds the cache
    by appending (embed(question), pred) to ac_agent._cache.

    Parameters
    ----------
    ac_agent:
        AnswerCacheAgent instance whose ``_cache`` list will be populated.
    p1_trajectories:
        List of Path objects pointing to P1 per-query trajectory JSON files.
    embedder:
        Callable[str] -> np.ndarray for embedding questions.

    Returns
    -------
    int
        Number of entries seeded (only correct P1 answers are seeded).
    """
    seeded = 0
    for traj_path in sorted(p1_trajectories):
        if not traj_path.exists():
            logger.warning("seed_answer_cache_from_p1: missing %s, skipping", traj_path)
            continue
        try:
            data = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("seed_answer_cache_from_p1: failed to read %s: %s", traj_path, exc)
            continue

        question = data.get("question", "")
        answer = data.get("pred", data.get("answer", ""))
        judgment = data.get("judgment_correct", data.get("judgment", False))

        if not question or not answer:
            continue
        # Only seed correct answers (mirrors AnswerCacheAgent.run() behavior)
        if not judgment:
            continue

        try:
            q_emb = embedder(question)
        except Exception as exc:
            logger.warning("seed_answer_cache_from_p1: embed failed for %s: %s", traj_path, exc)
            continue
        ac_agent._cache.append((q_emb, answer))
        seeded += 1

    logger.info("seed_answer_cache_from_p1: seeded %d entries", seeded)
    return seeded


def seed_trajectory_cache_from_p1(
    tc_agent: Any,
    p1_trajectories: list[Path],
    embedder: Callable[[str], Any],
) -> int:
    """Seed a TrajectoryCacheAgent with (embedding, question, trajectory_summary) from P1.

    Reads each P1 trajectory JSON, extracts the trajectory dict, converts it
    to a summary string, and seeds tc_agent._store.

    Parameters
    ----------
    tc_agent:
        TrajectoryCacheAgent instance whose ``_store`` list will be populated.
    p1_trajectories:
        List of Path objects pointing to P1 per-query trajectory JSON files.
    embedder:
        Callable[str] -> np.ndarray for embedding questions.

    Returns
    -------
    int
        Number of entries seeded (only correct P1 trajectories are seeded).
    """
    from hgc.baselines import _trajectory_to_summary

    seeded = 0
    for traj_path in sorted(p1_trajectories):
        if not traj_path.exists():
            logger.warning("seed_trajectory_cache_from_p1: missing %s, skipping", traj_path)
            continue
        try:
            data = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("seed_trajectory_cache_from_p1: failed to read %s: %s", traj_path, exc)
            continue

        question = data.get("question", "")
        trajectory = data.get("trajectory", {})
        judgment = data.get("judgment_correct", data.get("judgment", False))

        if not question:
            continue
        # Only seed correct trajectories (mirrors TrajectoryCacheAgent.run() behavior)
        if not judgment:
            continue

        try:
            q_emb = embedder(question)
        except Exception as exc:
            logger.warning("seed_trajectory_cache_from_p1: embed failed for %s: %s", traj_path, exc)
            continue
        traj_summary = _trajectory_to_summary(trajectory) if isinstance(trajectory, dict) else ""
        tc_agent._store.append((q_emb, question, traj_summary))
        seeded += 1

    logger.info("seed_trajectory_cache_from_p1: seeded %d entries", seeded)
    return seeded


def seed_mem0_from_p1(
    mem0_like: Any,
    p1_trajectories: list[Path],
    user_id: str = "user",
) -> int:
    """Seed a Mem0ReActAgent or a raw mem0.Memory object from P1 (question, answer) pairs.

    Accepts either:
    - A ``Mem0ReActAgent`` instance (duck-typed: has ``._mem0`` and ``._user_id``).
    - A raw ``mem0.Memory`` instance that has an ``.add(content, user_id=...)`` method.

    Parameters
    ----------
    mem0_like:
        Mem0ReActAgent or raw mem0.Memory instance.
    p1_trajectories:
        List of Path objects pointing to P1 per-query trajectory JSON files.
    user_id:
        Fallback user_id used when ``mem0_like`` is a raw Memory object (i.e. has
        no ``._user_id`` attribute).  When ``mem0_like`` is a Mem0ReActAgent its
        ``._user_id`` takes precedence.

    Returns
    -------
    int
        Number of memories stored.
    """
    # Duck-type: prefer ._mem0 (Mem0ReActAgent), fall back to direct .add (raw Memory).
    if hasattr(mem0_like, "_mem0"):
        mem0_obj = mem0_like._mem0
        uid = getattr(mem0_like, "_user_id", user_id)
    elif hasattr(mem0_like, "add"):
        mem0_obj = mem0_like
        uid = user_id
    else:
        raise TypeError(
            f"seed_mem0_from_p1: expected Mem0ReActAgent or mem0.Memory, got {type(mem0_like)}"
        )

    seeded = 0
    for traj_path in sorted(p1_trajectories):
        if not traj_path.exists():
            logger.warning("seed_mem0_from_p1: missing %s, skipping", traj_path)
            continue
        try:
            data = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("seed_mem0_from_p1: failed to read %s: %s", traj_path, exc)
            continue

        question = data.get("question", "")
        answer = data.get("pred", data.get("answer", ""))

        if not question or not answer:
            continue

        content = f"Q: {question}\nA: {answer}"
        try:
            mem0_obj.add(content, user_id=uid)
            seeded += 1
        except Exception as exc:
            logger.warning("seed_mem0_from_p1: mem0.add failed: %s", exc)

    logger.info("seed_mem0_from_p1: seeded %d memories", seeded)
    return seeded


# ---------------------------------------------------------------------------
# PhaseRunner
# ---------------------------------------------------------------------------


class PhaseRunner:
    """Orchestrates experiment phases on the BCP dataset.

    Parameters
    ----------
    dataset:
        Dataset dict with keys ``queries`` and ``paraphrased`` (from
        ``BCPDataset.load_or_build()``).
    judge:
        LLMJudge instance (must expose ``.judge(question, gold, pred) -> Verdict``).
    out_dir:
        Root output directory; phase results go to ``out_dir/{phase_name}/``.
    tool_factory:
        Callable ``tool_factory(query_record) -> list[Callable]`` that returns
        per-query tool closures.
    """

    def __init__(
        self,
        dataset: dict,
        judge: Any,
        out_dir: Path,
        tool_factory: Callable[[dict], list],
    ) -> None:
        self._dataset = dataset
        self._judge = judge
        self._out_dir = Path(out_dir)
        self._tool_factory = tool_factory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phase_dir(self, phase_name: str) -> Path:
        p = self._out_dir / phase_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _traj_path(self, phase_name: str, query_id: str) -> Path:
        return self._phase_dir(phase_name) / f"trajectory_q{query_id}.json"

    @staticmethod
    def _write_atomic(path: Path, data: dict) -> None:
        """Write *data* as JSON to *path* using atomic rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _containment_correct(gold: str, pred: str) -> bool:
        """Simple substring containment check (gold in pred, case-insensitive)."""
        return gold.strip().lower() in pred.lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_phase(
        self,
        phase_name: str,
        agent_factory: Callable[[dict, list], Any],
        on_paraphrased: bool = False,
    ) -> list[PhaseResult]:
        """Execute a single phase on all queries.

        Parameters
        ----------
        phase_name:
            Identifier written into result files (e.g. ``"P0"``, ``"P3-AC"``).
        agent_factory:
            Callable ``(query_record, tools) -> agent`` where ``agent`` exposes
            a ``run(question)`` or ``run(question, docs)`` method returning a
            result dict.
        on_paraphrased:
            If True, use ``dataset["paraphrased"]`` questions instead of
            ``dataset["queries"][i]["query"]``.

        Returns
        -------
        list[PhaseResult]
        """
        queries: list[dict] = self._dataset["queries"]
        paraphrased: list[str] = self._dataset.get("paraphrased", [])

        results: list[PhaseResult] = []

        for idx, qr in enumerate(queries):
            query_id = str(qr.get("query_id", idx))
            gold = qr.get("answer", "")

            if on_paraphrased and idx < len(paraphrased):
                question = paraphrased[idx]
            else:
                question = qr.get("query", "")

            traj_path = self._traj_path(phase_name, query_id)

            # Resume: skip if trajectory already written (but retry errored ones)
            if traj_path.exists():
                try:
                    existing = json.loads(traj_path.read_text(encoding="utf-8"))
                    prior_error = existing.get("error", "")
                    if prior_error:
                        # Errored trajectory — delete and rerun
                        logger.info(
                            "run_phase[%s] q%s: retrying (prior error: %s)",
                            phase_name,
                            query_id,
                            prior_error[:120],
                        )
                        traj_path.unlink()
                    else:
                        # Clean trajectory — skip
                        logger.info(
                            "run_phase[%s] q%s: skipping (trajectory exists)",
                            phase_name,
                            query_id,
                        )
                        results.append(
                            PhaseResult(
                                phase=phase_name,
                                query_id=query_id,
                                gold=gold,
                                pred=existing.get("pred", ""),
                                judgment_correct=bool(existing.get("judgment_correct", False)),
                                judgment_reasoning=existing.get("judgment_reasoning", ""),
                                containment_correct=bool(
                                    existing.get("containment_correct", False)
                                ),
                                tokens=int(existing.get("tokens", 0)),
                                time_s=float(existing.get("time_s", 0.0)),
                                n_iters=int(existing.get("n_iters", 0)),
                                n_retrieved_positive=int(existing.get("n_retrieved_positive", 0)),
                                n_retrieved_negative=int(existing.get("n_retrieved_negative", 0)),
                                n_added_hints=int(existing.get("n_added_hints", 0)),
                                error=existing.get("error", ""),
                                agent_wall_time=_opt_float(existing.get("agent_wall_time")),
                                judge_time_s=_opt_float(existing.get("judge_time_s")),
                            )
                        )
                        continue
                except Exception as exc:
                    logger.warning(
                        "run_phase[%s] q%s: failed to reload existing trajectory: %s",
                        phase_name,
                        query_id,
                        exc,
                    )
                    # Treat unreadable trajectory as errored — rerun
                    traj_path.unlink(missing_ok=True)

            # Build per-query tools and agent
            tools = self._tool_factory(qr)
            t0 = time.time()
            error_str = ""
            pred = ""
            judgment_correct = False
            judgment_reasoning = ""
            tokens = 0
            n_iters = 0
            n_retrieved_positive = 0
            n_retrieved_negative = 0
            n_added_hints = 0
            trajectory: dict = {}
            agent_wall_time: float | None = None
            judge_time_s: float | None = None

            try:
                agent = agent_factory(qr, tools)

                # Support both HGC-style (question only) and NaiveRAG-style
                # (question + docs)
                if _agent_wants_docs(agent):
                    docs = _collect_docs(qr)
                    run_result = agent.run(question, docs)
                else:
                    run_result = agent.run(question)

                pred = str(run_result.get("answer", ""))
                tokens = int(run_result.get("tokens", 0))
                gate_tokens = int(run_result.get("gate_tokens", 0) or 0)
                path = run_result.get("path")
                cache_hit = run_result.get("cache_hit")
                gate_reject_reason = run_result.get("gate_reject_reason")
                gate_stage_counts = run_result.get("gate_stage_counts")
                n_iters = int(run_result.get("n_iters", 0))
                trajectory = run_result.get("trajectory", {}) or {}
                # Every agent already measures its own monotonic span; the
                # runner used to drop it on the floor.
                agent_wall_time = _opt_float(run_result.get("wall_time"))

                # Hints fields (only HGCCoreAgent populates these)
                n_retrieved_positive = len(run_result.get("retrieved_positive_hints", []))
                n_retrieved_negative = len(run_result.get("retrieved_negative_hints", []))
                n_added_hints = len(run_result.get("added_hints", []))

                # Judge — timed separately so it can be subtracted out of
                # ``time_s``. Judging is evaluation scaffolding, not part of
                # what a deployed system would pay.
                _judge_t0 = time.monotonic()
                verdict = self._judge.judge(question, gold, pred)
                judge_time_s = time.monotonic() - _judge_t0
                judgment_correct = bool(verdict.correct)
                judgment_reasoning = verdict.reasoning

            except Exception as exc:
                error_str = str(exc)
                gate_tokens = 0
                path = None
                cache_hit = None
                gate_reject_reason = None
                gate_stage_counts = None
                logger.warning("run_phase[%s] q%s: exception: %s", phase_name, query_id, exc)

            time_s = time.time() - t0
            containment = self._containment_correct(gold, pred)

            # Write per-query trajectory JSON (atomic)
            traj_record: dict = {
                "query_id": query_id,
                "question": question,
                "gold": gold,
                "phase": phase_name,
                "pred": pred,
                "judgment_correct": judgment_correct,
                "judgment_reasoning": judgment_reasoning,
                "containment_correct": containment,
                "tokens": tokens,
                "gate_tokens": gate_tokens,
                "path": path,
                "cache_hit": cache_hit,
                "gate_reject_reason": gate_reject_reason,
                "gate_stage_counts": gate_stage_counts,
                "time_s": time_s,
                "agent_wall_time": agent_wall_time,
                "judge_time_s": judge_time_s,
                "n_iters": n_iters,
                "n_retrieved_positive": n_retrieved_positive,
                "n_retrieved_negative": n_retrieved_negative,
                "n_added_hints": n_added_hints,
                "error": error_str,
                "trajectory": trajectory,
            }
            self._write_atomic(traj_path, traj_record)

            result = PhaseResult(
                phase=phase_name,
                query_id=query_id,
                gold=gold,
                pred=pred,
                judgment_correct=judgment_correct,
                judgment_reasoning=judgment_reasoning,
                containment_correct=containment,
                tokens=tokens,
                time_s=time_s,
                n_iters=n_iters,
                n_retrieved_positive=n_retrieved_positive,
                n_retrieved_negative=n_retrieved_negative,
                n_added_hints=n_added_hints,
                error=error_str,
                agent_wall_time=agent_wall_time,
                judge_time_s=judge_time_s,
            )
            results.append(result)

        return results

    def write_summary(self, phase_name: str, results: list[PhaseResult]) -> Path:
        """Write ``results/{phase_name}/summary.csv`` with one row per query.

        Parameters
        ----------
        phase_name:
            Phase identifier (must match the directory created by run_phase).
        results:
            List of PhaseResult objects returned by run_phase.

        Returns
        -------
        Path
            Absolute path to the written CSV file.
        """
        phase_dir = self._phase_dir(phase_name)
        csv_path = phase_dir / "summary.csv"

        columns = [
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
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "query_id": r.query_id,
                        "pred": r.pred[:100],
                        "judgment_correct": r.judgment_correct,
                        "containment_correct": r.containment_correct,
                        "tokens": r.tokens,
                        "time_s": round(r.time_s, 3),
                        "agent_wall_time": (
                            "" if r.agent_wall_time is None else round(r.agent_wall_time, 3)
                        ),
                        "judge_time_s": (
                            "" if r.judge_time_s is None else round(r.judge_time_s, 3)
                        ),
                        "n_iters": r.n_iters,
                        "n_retrieved_positive": r.n_retrieved_positive,
                        "n_retrieved_negative": r.n_retrieved_negative,
                        "n_added_hints": r.n_added_hints,
                        "error": r.error[:200] if r.error else "",
                    }
                )

        logger.info("write_summary[%s]: wrote %d rows to %s", phase_name, len(results), csv_path)
        return csv_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _agent_wants_docs(agent: Any) -> bool:
    """Return True if the agent's run() method accepts a 'docs' parameter.

    NaiveRAGAgent.run(question, docs) takes docs; all others take only question.
    Detect by inspecting the run() signature.
    """
    import inspect

    try:
        sig = inspect.signature(agent.run)
        return "docs" in sig.parameters
    except (TypeError, ValueError):
        return False


def _collect_docs(qr: dict) -> list[dict]:
    """Collect all document dicts from a query record.

    BCP records split docs into ``gold_docs`` / ``evidence_docs`` / ``negative_docs``.
    QASPER and FinanceBench records ship a single unified ``docs`` list. Return
    whichever shape is present — BCP wins when both keys exist.
    """
    bcp_docs = qr.get("gold_docs", []) + qr.get("evidence_docs", []) + qr.get("negative_docs", [])
    if bcp_docs:
        return bcp_docs
    return list(qr.get("docs", []))
