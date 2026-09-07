"""Per-path accuracy aggregation for Appendix A and H.

For each main-paper experimental cell, count trajectories falling into each path
{cache_verified, cache_fallback, cache_miss, cache_hit_no_hint} and compute
within-path accuracy (judge-correct rate).

The HGC path is recorded in trajectory_q*.json under either:
  - top-level 'path' field (HGCAgent / HGCRAGAgent), or
  - trajectory['path'] (older format)
For AC/Vanilla phases, all queries default to 'ac_only' or 'vanilla'.

Output: hgc/analysis/per_path_accuracy.csv with columns
  experiment, phase, path, n, n_correct, accuracy_pct
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from hgc.paths import results_root

CODE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = results_root(CODE_ROOT)
OUT_CSV = CODE_ROOT / "analysis" / "per_path_accuracy.csv"

MAIN_CELLS = [
    ("bcp_gpt41_seed42", "P3-AC"),
    ("bcp_gpt41_seed42", "P3-Hybrid"),
    ("bcp_gpt41_seed42", "P4-AC"),
    ("bcp_gpt41_seed42", "P4-Hybrid"),
    ("bcp_gpt41_seed7", "P3-AC"),
    ("bcp_gpt41_seed7", "P3-Hybrid"),
    ("bcp_gpt41_seed7", "P4-AC"),
    ("bcp_gpt41_seed7", "P4-Hybrid"),
    ("bcp_gpt41_seed100", "P3-AC"),
    ("bcp_gpt41_seed100", "P3-Hybrid"),
    ("bcp_gpt41_seed100", "P4-AC"),
    ("bcp_gpt41_seed100", "P4-Hybrid"),
    ("bcp_gpt4o", "P3-AC"),
    ("bcp_gpt4o", "P3-Hybrid"),
    ("bcp_gpt4o", "P4-AC"),
    ("bcp_gpt4o", "P4-Hybrid"),
    # Long-context and RAG cells come from the reruns on the fixed code
    # (base_*), which is what the paper's tables report. The pre-fix
    # directories are superseded and are not part of the release.
    ("base_qasper_oracle", "AC-LC"),
    ("base_qasper_oracle", "HGC-LC"),
    ("base_qasper_oracle", "C-AC-LC"),
    ("base_qasper_oracle", "C-HGC-LC"),
    ("base_qasper_rag", "AC-RAG"),
    ("base_qasper_rag", "HGC-RAG"),
    ("base_qasper_rag", "C-AC-RAG"),
    ("base_qasper_rag", "C-HGC-RAG"),
    ("base_finbench_lc", "AC-LC"),
    ("base_finbench_lc", "HGC-LC"),
    ("base_finbench_lc", "C-AC-LC"),
    ("base_finbench_lc", "C-HGC-LC"),
]


def get_path(d: dict) -> str:
    """Extract path label from a trajectory record."""
    p = d.get("path")
    if p:
        return p
    traj = d.get("trajectory") or {}
    p = traj.get("path") if isinstance(traj, dict) else None
    if p:
        return p
    # Fallback by phase prefix
    phase = d.get("phase", "")
    if "AC" in phase and "Hybrid" not in phase and "HGC" not in phase:
        return "ac_only"
    return "unknown"


def main() -> None:
    rows = []
    for exp, phase in MAIN_CELLS:
        phase_dir = RESULTS / exp / phase
        if not phase_dir.exists():
            continue
        path_buckets: dict[str, list[bool]] = {}
        for tp in sorted(phase_dir.glob("trajectory_q*.json")):
            try:
                d = json.loads(tp.read_text())
            except Exception:
                continue
            path = get_path(d)
            correct = bool(d.get("judgment_correct"))
            path_buckets.setdefault(path, []).append(correct)

        for path, results in sorted(path_buckets.items()):
            n = len(results)
            n_correct = sum(results)
            acc = 100 * n_correct / n if n else 0
            rows.append(
                {
                    "experiment": exp,
                    "phase": phase,
                    "path": path,
                    "n": n,
                    "n_correct": n_correct,
                    "accuracy_pct": round(acc, 1),
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["experiment", "phase", "path", "n", "n_correct", "accuracy_pct"],
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print()

    # Pretty print per-cell summary
    print(f"{'Experiment':<28} {'Phase':<12} {'Path':<22} {'n':>4} {'corr':>5} {'acc%':>6}")
    print("-" * 86)
    for r in rows:
        print(
            f"{r['experiment']:<28} {r['phase']:<12} {r['path']:<22} "
            f"{r['n']:>4} {r['n_correct']:>5} {r['accuracy_pct']:>6.1f}"
        )


if __name__ == "__main__":
    main()
