"""Regenerate smoke_summary.json for each experiment dir post-rerun.

Reads per-phase trajectory_q*.json files and produces the aggregate
smoke_summary.json by merging rerun phases with original non-rerun phases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hgc.runner import PhaseResult  # noqa: E402
from hgc.smoke_common import phase_summary  # noqa: E402

from hgc.paths import results_root

RESULTS_DIR = results_root(Path(__file__).resolve().parent.parent)

EXPERIMENTS = {
    "bcp_gpt41_seed42": [
        "P0",
        "P1",
        "P3",
        "P3-AC",
        "P3-Hybrid",
        "P4-AC",
        "P4-Hybrid",
        "E4-Hybrid",
        "T4-Hybrid",
    ],
    "bcp_gpt41_seed7": ["P1", "P3-AC", "P3-Hybrid", "P4-AC", "P4-Hybrid"],
    "bcp_gpt41_seed100": ["P1", "P3-AC", "P3-Hybrid", "P4-AC", "P4-Hybrid"],
    "bcp_gpt4o": ["P1", "P3-AC", "P3-Hybrid", "P4-AC", "P4-Hybrid"],
    "base_qasper_rag": ["P1-RAG", "V-RAG", "AC-RAG", "HGC-RAG", "C-AC-RAG", "C-HGC-RAG"],
    "base_qasper_oracle": ["P1-LC", "V-LC", "AC-LC", "HGC-LC", "C-AC-LC", "C-HGC-LC"],
    "base_finbench_lc": ["P1-LC", "V-LC", "AC-LC", "HGC-LC", "C-AC-LC", "C-HGC-LC"],
}


def regenerate(exp_name: str, phases: list[str]) -> None:
    out_dir = RESULTS_DIR / exp_name
    if not out_dir.exists():
        print(f"SKIP (missing dir): {exp_name}")
        return
    all_summaries = []
    for phase_name in phases:
        phase_dir = out_dir / phase_name
        if not phase_dir.exists():
            continue
        traj_files = sorted(phase_dir.glob("trajectory_q*.json"))
        results = []
        for tf in traj_files:
            try:
                data = json.loads(tf.read_text(encoding="utf-8"))
                if data.get("error"):
                    continue
                results.append(
                    PhaseResult(
                        phase=phase_name,
                        query_id=str(data.get("query_id", "")),
                        gold=data.get("gold", ""),
                        pred=data.get("pred", ""),
                        judgment_correct=bool(data.get("judgment_correct", False)),
                        judgment_reasoning=data.get("judgment_reasoning", ""),
                        containment_correct=bool(data.get("containment_correct", False)),
                        tokens=int(data.get("tokens", 0)),
                        time_s=float(data.get("time_s", 0.0)),
                        n_iters=int(data.get("n_iters", 0)),
                        n_retrieved_positive=int(data.get("n_retrieved_positive", 0)),
                        n_retrieved_negative=int(data.get("n_retrieved_negative", 0)),
                        n_added_hints=int(data.get("n_added_hints", 0)),
                        error=data.get("error", ""),
                    )
                )
            except Exception:
                continue
        if results:
            all_summaries.append(phase_summary(phase_name, results))

    if not all_summaries:
        print(f"EMPTY (no trajectory files found): {exp_name}")
        return
    smoke_summary_path = out_dir / "smoke_summary.json"
    smoke_summary_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))
    print(f"OK  {exp_name}: {len(all_summaries)} phases regenerated")


if __name__ == "__main__":
    for exp, phases in EXPERIMENTS.items():
        regenerate(exp, phases)
