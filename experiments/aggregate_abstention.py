"""Aggregate per-phase abstention taxonomy using the LLM ensemble majority vote.

Correct predictions inherit the category 'correct'. Incorrect predictions
receive the majority vote across (regex, gpt-4.1-mini, claude-haiku).

Writes: analysis/abstention_taxonomy_final.csv and prints an AC-vs-HGC
comparison table for the main conditions.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_REGEX = CODE_ROOT / "analysis" / "abstention_taxonomy.csv"
TAXONOMY_LLM = CODE_ROOT / "analysis" / "abstention_taxonomy_llm.csv"
OUT_CSV = CODE_ROOT / "analysis" / "abstention_taxonomy_final.csv"

MAIN_EXPERIMENTS = {
    "bcp_gpt41_seed42",
    "bcp_gpt41_seed7",
    "bcp_gpt41_seed100",
    "bcp_gpt4o",
    "base_qasper_oracle",
    "base_qasper_rag",
    "base_finbench_lc",
    "bcp_a1_g2off",
    "bcp_a1_g3off",
    "bcp_a2_5pct",
    "bcp_a2_50pct",
}

# Main-narrative phase pairs (AC vs HGC; clean and contaminated)
MAIN_COMPARISONS = [
    ("bcp_gpt41_seed42", "P3-AC", "P3-Hybrid", "BCP seed 42 (clean)"),
    ("bcp_gpt41_seed42", "P4-AC", "P4-Hybrid", "BCP seed 42 (contam)"),
    ("bcp_gpt41_seed7", "P3-AC", "P3-Hybrid", "BCP seed 7 (clean)"),
    ("bcp_gpt41_seed7", "P4-AC", "P4-Hybrid", "BCP seed 7 (contam)"),
    ("bcp_gpt41_seed100", "P3-AC", "P3-Hybrid", "BCP seed 100 (clean)"),
    ("bcp_gpt41_seed100", "P4-AC", "P4-Hybrid", "BCP seed 100 (contam)"),
    ("bcp_gpt4o", "P3-AC", "P3-Hybrid", "BCP gpt-4o (clean)"),
    ("bcp_gpt4o", "P4-AC", "P4-Hybrid", "BCP gpt-4o (contam)"),
    ("base_qasper_oracle", "AC-LC", "HGC-LC", "QASPER-Oracle (clean)"),
    ("base_qasper_oracle", "C-AC-LC", "C-HGC-LC", "QASPER-Oracle (contam)"),
    ("base_qasper_rag", "AC-RAG", "HGC-RAG", "QASPER-RAG (clean)"),
    ("base_qasper_rag", "C-AC-RAG", "C-HGC-RAG", "QASPER-RAG (contam)"),
    ("base_finbench_lc", "AC-LC", "HGC-LC", "FinanceBench-LC (clean)"),
    ("base_finbench_lc", "C-AC-LC", "C-HGC-LC", "FinanceBench-LC (contam)"),
]


def load_regex() -> dict[tuple[str, str, str], str]:
    """Key: (experiment, phase, query_id) -> regex category."""
    with open(TAXONOMY_REGEX) as f:
        rows = list(csv.DictReader(f))
    return {(r["experiment"], r["phase"], r["query_id"]): r["category"] for r in rows}


def load_llm() -> dict[tuple[str, str, str], dict]:
    with open(TAXONOMY_LLM) as f:
        rows = list(csv.DictReader(f))
    return {(r["experiment"], r["phase"], r["query_id"]): r for r in rows}


def majority3(a: str, b: str, c: str) -> str:
    counts = Counter([x for x in (a, b, c) if not x.startswith("error")])
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def main() -> None:
    regex_all = load_regex()
    llm_rows = load_llm()

    final = {}
    for key, cat in regex_all.items():
        exp, phase, qid = key
        if exp not in MAIN_EXPERIMENTS:
            continue
        if cat == "correct":
            final[key] = "correct"
        elif cat == "empty_or_error":
            final[key] = "empty_or_error"
        else:
            # judge=incorrect; consult LLM labels if available
            llm = llm_rows.get(key)
            if llm is None:
                final[key] = cat  # regex fallback (shouldn't happen for main exps)
            else:
                final[key] = majority3(
                    llm["regex_label"], llm["gpt4mini_label"], llm["claude_label"]
                )

    # Write final per-row taxonomy
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["experiment", "phase", "query_id", "final_category"])
        for (exp, phase, qid), cat in sorted(final.items()):
            w.writerow([exp, phase, qid, cat])

    print(f"Wrote {len(final)} rows to {OUT_CSV}")
    print()

    # Per-phase aggregate
    grp: dict[tuple[str, str], list[str]] = {}
    for (exp, phase, qid), cat in final.items():
        grp.setdefault((exp, phase), []).append(cat)

    # AC vs HGC comparison table
    print(
        f"{'Condition':<28} {'n':>4} "
        f"{'AC correct':>10} {'AC abstain':>10} {'AC wrong':>9} "
        f"{'HGC correct':>11} {'HGC abstain':>11} {'HGC wrong':>10} "
        f"{'Δ wrong':>8}"
    )
    print("-" * 130)
    for exp, ac_phase, hgc_phase, label in MAIN_COMPARISONS:
        ac = grp.get((exp, ac_phase))
        hg = grp.get((exp, hgc_phase))
        if ac is None or hg is None:
            continue
        n = len(ac)
        ac_c = Counter(ac)
        hg_c = Counter(hg)

        def fmt(cnt, key):
            v = cnt.get(key, 0)
            return f"{v} ({100 * v / n:.0f}%)"

        print(
            f"{label:<28} {n:>4} "
            f"{fmt(ac_c, 'correct'):>10} {fmt(ac_c, 'honest_abstention'):>10} {fmt(ac_c, 'confident_wrong'):>9} "
            f"{fmt(hg_c, 'correct'):>11} {fmt(hg_c, 'honest_abstention'):>11} {fmt(hg_c, 'confident_wrong'):>10} "
            f"{hg_c.get('confident_wrong', 0) - ac_c.get('confident_wrong', 0):>+8}"
        )


if __name__ == "__main__":
    main()
