"""Sample (question, gold, predicted, judge_verdict) rows for human validation of the LLM judge.

Stratified sample: 25 judge="yes" + 25 judge="no" per benchmark, four benchmark
cells total (BCP, QASPER-Oracle, FinanceBench-LC, QASPER-RAG) = 200 rows.
QASPER-RAG was added after the original three-cell sampling pass; running this
script regenerates the four-cell sample, but the held-out human labels for
QASPER-RAG live in analysis/judge_validation/qasper_rag_labeled.csv with a
phase-agnostic schema (see compute_judge_agreement.py for caveats).

Usage:
    python sample_for_judge_validation.py

Writes to: hgc/analysis/judge_validation/human_labels.csv
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from hgc.paths import results_root

RESULTS = results_root(Path(__file__).resolve().parents[1])

# NOTE: the cells below name the pre-fix cohort the validation sample was
# actually drawn from. They are kept as a record of provenance, not as a live
# path: those directories are superseded by the base_* reruns and are not
# part of the anonymous release, so re-running this script there will find no
# long-context or RAG cells. The labels it produced ship in
# analysis/judge_validation/ and are what the agreement figures are computed
# from; regenerating the sample is not needed to check them.
OUT_DIR = Path(__file__).resolve().parents[1] / "analysis" / "judge_validation"
OUT_FILE = OUT_DIR / "human_labels.csv"

BENCHMARK_PHASES = {
    "bcp": [
        ("bcp_gpt41_seed42", "P3-AC"),
        ("bcp_gpt41_seed42", "P3-Hybrid"),
        ("bcp_gpt41_seed42", "P4-AC"),
        ("bcp_gpt41_seed42", "P4-Hybrid"),
    ],
    "qasper": [
        ("qasper_oracle_gpt41_seed42", "AC-LC"),
        ("qasper_oracle_gpt41_seed42", "HGC-LC"),
        ("qasper_oracle_gpt41_seed42", "C-AC-LC"),
        ("qasper_oracle_gpt41_seed42", "C-HGC-LC"),
    ],
    "financebench": [
        ("finbench_lc_gpt41_seed42", "AC-LC"),
        ("finbench_lc_gpt41_seed42", "HGC-LC"),
        ("finbench_lc_gpt41_seed42", "C-AC-LC"),
        ("finbench_lc_gpt41_seed42", "C-HGC-LC"),
    ],
    "qasper_rag": [
        ("qasper-rag_gpt41_seed42", "AC-RAG"),
        ("qasper-rag_gpt41_seed42", "HGC-RAG"),
        ("qasper-rag_gpt41_seed42", "C-AC-RAG"),
        ("qasper-rag_gpt41_seed42", "C-HGC-RAG"),
    ],
}

PER_BENCH_YES = 25
PER_BENCH_NO = 25
SEED = 20260424


def load_rows(dataset_dir: str, phase: str) -> list[dict]:
    phase_dir = RESULTS / dataset_dir / phase
    rows = []
    for p in sorted(phase_dir.glob("trajectory_q*.json")):
        with open(p) as f:
            d = json.load(f)
        rows.append(
            {
                "query_id": d.get("query_id", ""),
                "phase": phase,
                "question": (d.get("question") or "").strip(),
                "gold_answer": (d.get("gold") or "").strip(),
                "predicted_answer": (d.get("pred") or "").strip(),
                "judge_verdict": "yes" if d.get("judgment_correct") else "no",
                "judge_reasoning": (d.get("judgment_reasoning") or "").strip(),
            }
        )
    return rows


def stratified_sample(
    all_rows: list[dict], yes_n: int, no_n: int, rng: random.Random
) -> list[dict]:
    yes_rows = [r for r in all_rows if r["judge_verdict"] == "yes"]
    no_rows = [r for r in all_rows if r["judge_verdict"] == "no"]
    rng.shuffle(yes_rows)
    rng.shuffle(no_rows)
    picked_yes = yes_rows[:yes_n]
    picked_no = no_rows[:no_n]
    combined = picked_yes + picked_no
    rng.shuffle(combined)
    return combined, len(yes_rows), len(no_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    all_samples = []
    for bench, phases in BENCHMARK_PHASES.items():
        bench_rows = []
        for dataset_dir, phase in phases:
            bench_rows.extend(load_rows(dataset_dir, phase))

        picked, n_yes_avail, n_no_avail = stratified_sample(
            bench_rows, PER_BENCH_YES, PER_BENCH_NO, rng
        )
        print(
            f"{bench}: {len(bench_rows)} rows available "
            f"(yes={n_yes_avail}, no={n_no_avail}) "
            f"-> sampled {len(picked)} ({sum(1 for r in picked if r['judge_verdict'] == 'yes')} yes / "
            f"{sum(1 for r in picked if r['judge_verdict'] == 'no')} no)"
        )
        for r in picked:
            r["benchmark"] = bench
            all_samples.append(r)

    cols = [
        "benchmark",
        "query_id",
        "phase",
        "question",
        "gold_answer",
        "predicted_answer",
        "judge_verdict",
        "your_label",
        "judge_reasoning",
    ]
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in all_samples:
            r.setdefault("your_label", "")
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"\nWrote {len(all_samples)} rows to {OUT_FILE}")
    print("Label each row's `your_label` column with 'yes' (predicted answer is correct) or 'no'.")


if __name__ == "__main__":
    main()
