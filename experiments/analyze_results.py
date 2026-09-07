"""Comprehensive analysis of HGC rerun results.

Outputs:
- McNemar's test (AC vs HGC under contamination + clean)
- Error breakdown by path / gate_reject_reason
- Cost table with gate_tokens breakdown
- Markdown report at analysis/final_analysis.md
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

from hgc.paths import results_root

RESULTS_DIR = results_root(Path(__file__).resolve().parent.parent)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis"


EXPERIMENTS = [
    (
        "bcp_gpt41_seed42",
        "BCP seed=42 (gpt-4.1)",
        {
            "clean_ac": "P3-AC",
            "clean_hgc": "P3-Hybrid",
            "contam_ac": "P4-AC",
            "contam_hgc": "P4-Hybrid",
            "extras": [
                ("P4-AC", "E4-Hybrid", "entity-swap"),
                ("P4-AC", "T4-Hybrid", "typo-mutation"),
            ],
        },
    ),
    (
        "bcp_gpt41_seed7",
        "BCP seed=7",
        {
            "clean_ac": "P3-AC",
            "clean_hgc": "P3-Hybrid",
            "contam_ac": "P4-AC",
            "contam_hgc": "P4-Hybrid",
        },
    ),
    (
        "bcp_gpt41_seed100",
        "BCP seed=100",
        {
            "clean_ac": "P3-AC",
            "clean_hgc": "P3-Hybrid",
            "contam_ac": "P4-AC",
            "contam_hgc": "P4-Hybrid",
        },
    ),
    (
        "bcp_gpt4o",
        "BCP gpt-4o",
        {
            "clean_ac": "P3-AC",
            "clean_hgc": "P3-Hybrid",
            "contam_ac": "P4-AC",
            "contam_hgc": "P4-Hybrid",
        },
    ),
    (
        "base_qasper_rag",
        "QASPER-RAG",
        {
            "clean_ac": "AC-RAG",
            "clean_hgc": "HGC-RAG",
            "contam_ac": "C-AC-RAG",
            "contam_hgc": "C-HGC-RAG",
        },
    ),
    (
        "base_qasper_oracle",
        "QASPER-Oracle",
        {
            "clean_ac": "AC-LC",
            "clean_hgc": "HGC-LC",
            "contam_ac": "C-AC-LC",
            "contam_hgc": "C-HGC-LC",
        },
    ),
    (
        "base_finbench_lc",
        "FinanceBench-LC",
        {
            "clean_ac": "AC-LC",
            "clean_hgc": "HGC-LC",
            "contam_ac": "C-AC-LC",
            "contam_hgc": "C-HGC-LC",
        },
    ),
]


def load_phase(exp_dir: Path, phase: str) -> dict[str, dict]:
    """Return {query_id: record} for one phase dir (via trajectory JSONs)."""
    phase_dir = exp_dir / phase
    if not phase_dir.exists():
        return {}
    out = {}
    for tf in phase_dir.glob("trajectory_q*.json"):
        try:
            data = json.loads(tf.read_text())
            qid = str(data.get("query_id", ""))
            if qid:
                out[qid] = data
        except Exception:
            continue
    return out


def mcnemar(ac: dict[str, dict], hgc: dict[str, dict]) -> dict:
    """Return {a, b, c, d, chi2, p, n_pairs}."""
    common = set(ac) & set(hgc)
    a = b = c = d = 0
    for qid in common:
        ac_ok = bool(ac[qid].get("judgment_correct"))
        hgc_ok = bool(hgc[qid].get("judgment_correct"))
        if ac_ok and hgc_ok:
            a += 1
        elif ac_ok and not hgc_ok:
            b += 1
        elif not ac_ok and hgc_ok:
            c += 1
        else:
            d += 1
    if b + c == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = math.erfc(math.sqrt(chi2 / 2))  # 1-sided normal approx of chi2 df=1
    return {"a": a, "b": b, "c": c, "d": d, "chi2": chi2, "p": p, "n_pairs": len(common)}


def phase_accuracy(records: dict[str, dict]) -> float:
    if not records:
        return 0.0
    correct = sum(1 for r in records.values() if r.get("judgment_correct"))
    return 100 * correct / len(records)


def phase_cost(records: dict[str, dict]) -> dict:
    """Return cost-related aggregates from trajectory records."""
    if not records:
        return {"n": 0}
    toks = [int(r.get("tokens", 0)) for r in records.values()]
    gate_toks = [int(r.get("gate_tokens", 0) or 0) for r in records.values()]
    times = [float(r.get("time_s", 0.0)) for r in records.values()]
    paths = [r.get("path") for r in records.values()]
    paths_count = {}
    for p in paths:
        paths_count[p] = paths_count.get(p, 0) + 1
    gate_reject = {}
    for r in records.values():
        reason = r.get("gate_reject_reason")
        if reason:
            gate_reject[reason] = gate_reject.get(reason, 0) + 1
    return {
        "n": len(records),
        "avg_tokens": statistics.mean(toks) if toks else 0,
        "total_tokens": sum(toks),
        "avg_gate_tokens": statistics.mean(gate_toks) if gate_toks else 0,
        "avg_time_s": statistics.mean(times) if times else 0,
        "paths": paths_count,
        "gate_reject": gate_reject,
    }


def format_mcnemar(m: dict, label: str) -> str:
    sig = ""
    if m["p"] < 0.001:
        sig = "***"
    elif m["p"] < 0.01:
        sig = "**"
    elif m["p"] < 0.05:
        sig = "*"
    elif m["p"] < 0.1:
        sig = "."
    return (
        f"| {label} | {m['n_pairs']} | a={m['a']} b={m['b']} c={m['c']} d={m['d']} | "
        f"{m['chi2']:.2f} | {m['p']:.4g}{sig} |"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    lines: list[str] = []
    lines.append("# HGC Final Analysis (post G3 token fix)")
    lines.append("")
    lines.append(
        "Generated from `experiments/analyze_results.py` using released "
        "trajectory JSONs in `trajectories/<experiment>/<phase>/` (or `results/...` in the code repository)."
    )
    lines.append("")

    # ==================================================================
    # Section 1: Accuracy table
    # ==================================================================
    lines.append("## 1. Accuracy table (judge)")
    lines.append("")
    lines.append(
        "Drop columns are computed from raw correct/n counts (so they may differ "
        "in the last digit from the displayed-cell difference: e.g., FinanceBench-LC "
        "HGC drop $128/150 - 124/150 = -2.67$pp $\\to -2.7$pp on a raw-count basis, "
        "while the displayed-percent difference $85.3 - 82.7 = -2.6$pp). The paper "
        "Table 1 caption uses the same raw-count convention."
    )
    lines.append("")
    lines.append(
        "| Experiment | Clean AC | Clean HGC | Contam AC | Contam HGC | AC drop | HGC drop |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for exp_id, label, phases in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_id
        ac_c = load_phase(exp_dir, phases["clean_ac"])
        hgc_c = load_phase(exp_dir, phases["clean_hgc"])
        ac_x = load_phase(exp_dir, phases["contam_ac"])
        hgc_x = load_phase(exp_dir, phases["contam_hgc"])
        ac_clean_acc = phase_accuracy(ac_c)
        hgc_clean_acc = phase_accuracy(hgc_c)
        ac_contam_acc = phase_accuracy(ac_x)
        hgc_contam_acc = phase_accuracy(hgc_x)
        lines.append(
            f"| {label} | {ac_clean_acc:.1f} | {hgc_clean_acc:.1f} | "
            f"{ac_contam_acc:.1f} | **{hgc_contam_acc:.1f}** | "
            f"{ac_contam_acc - ac_clean_acc:+.1f} | {hgc_contam_acc - hgc_clean_acc:+.1f} |"
        )
    lines.append("")

    # ==================================================================
    # Section 2: McNemar's tests (AC vs HGC under contamination)
    # ==================================================================
    lines.append("## 2. McNemar's tests")
    lines.append("")
    lines.append("Pair-level comparison per query, 2x2 contingency on judge correctness.")
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
    lines.append("")
    lines.append("### 2.1 Contaminated: C-AC vs C-HGC")
    lines.append("")
    lines.append("| Test | n | a/b/c/d | χ² | p |")
    lines.append("|---|---|---|---|---|")
    all_pvalues: list[tuple[str, float]] = []
    for exp_id, label, phases in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_id
        ac_x = load_phase(exp_dir, phases["contam_ac"])
        hgc_x = load_phase(exp_dir, phases["contam_hgc"])
        m = mcnemar(ac_x, hgc_x)
        lines.append(format_mcnemar(m, f"{label} (contam)"))
        all_pvalues.append((f"{label} contam AC vs HGC", m["p"]))
    lines.append("")
    lines.append("### 2.2 Clean: AC vs HGC")
    lines.append("")
    lines.append("| Test | n | a/b/c/d | χ² | p |")
    lines.append("|---|---|---|---|---|")
    for exp_id, label, phases in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_id
        ac_c = load_phase(exp_dir, phases["clean_ac"])
        hgc_c = load_phase(exp_dir, phases["clean_hgc"])
        m = mcnemar(ac_c, hgc_c)
        lines.append(format_mcnemar(m, f"{label} (clean)"))
        all_pvalues.append((f"{label} clean AC vs HGC", m["p"]))
    lines.append("")

    # Bonferroni correction
    n_tests = len(all_pvalues)
    lines.append(f"### Bonferroni correction (n={n_tests} tests)")
    lines.append("")
    lines.append("| Test | raw p | Bonferroni-adjusted p | Significant at α=0.05? |")
    lines.append("|---|---|---|---|")
    for name, p in all_pvalues:
        adj = min(1.0, p * n_tests)
        sig = "✓" if adj < 0.05 else "—"
        lines.append(f"| {name} | {p:.4g} | {adj:.4g} | {sig} |")
    lines.append("")

    # ==================================================================
    # Section 3: Cost breakdown
    # ==================================================================
    lines.append("## 3. Cost breakdown")
    lines.append("")
    lines.append("| Experiment | Phase | n | avg_tokens | avg_gate_tokens | gate% | avg_time_s |")
    lines.append("|---|---|---|---|---|---|---|")
    for exp_id, label, phases in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_id
        for ph_key in ("clean_ac", "clean_hgc", "contam_ac", "contam_hgc"):
            phase = phases[ph_key]
            records = load_phase(exp_dir, phase)
            c = phase_cost(records)
            if c["n"] == 0:
                continue
            gate_pct = 100 * c["avg_gate_tokens"] / c["avg_tokens"] if c["avg_tokens"] else 0
            lines.append(
                f"| {label} | {phase} | {c['n']} | {c['avg_tokens']:,.0f} | "
                f"{c['avg_gate_tokens']:,.0f} | {gate_pct:.1f}% | {c['avg_time_s']:.1f} |"
            )
    lines.append("")

    # ==================================================================
    # Section 4: HGC path breakdown
    # ==================================================================
    lines.append("## 4. HGC path distribution (contaminated setting)")
    lines.append("")
    lines.append(
        "Paths: cache_verified (gate passed), cache_fallback (gate rejected), "
        "cache_miss (no AC hit), cache_hit_no_hint (AC hit but no location hints)."
    )
    lines.append("")
    lines.append(
        "| Experiment | cache_verified | cache_fallback | cache_miss | cache_hit_no_hint |"
    )
    lines.append("|---|---|---|---|---|")
    for exp_id, label, phases in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_id
        hgc_x = load_phase(exp_dir, phases["contam_hgc"])
        c = phase_cost(hgc_x)
        paths = c.get("paths", {})
        n = c["n"] or 1
        row = "| " + label
        for p_name in ("cache_verified", "cache_fallback", "cache_miss", "cache_hit_no_hint"):
            count = paths.get(p_name, 0)
            row += f" | {count} ({100 * count / n:.0f}%)"
        row += " |"
        lines.append(row)
    lines.append("")

    # ==================================================================
    # Section 5: Gate reject reasons
    # ==================================================================
    lines.append("## 5. Gate reject reasons (cache_fallback breakdown)")
    lines.append("")
    lines.append("| Experiment | no_location_hint_in_scope | no_hint_passed_all_components |")
    lines.append("|---|---|---|")
    for exp_id, label, phases in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_id
        hgc_x = load_phase(exp_dir, phases["contam_hgc"])
        c = phase_cost(hgc_x)
        reasons = c.get("gate_reject", {})
        lines.append(
            f"| {label} | {reasons.get('no_location_hint_in_scope', 0)} | "
            f"{reasons.get('no_hint_passed_all_components', 0)} |"
        )
    lines.append("")

    # ==================================================================
    # Section 6: Contamination-mode sub-analysis (BCP seed=42 only)
    # ==================================================================
    lines.append("## 6. Contamination-mode comparison (BCP seed=42)")
    lines.append("")
    lines.append("Compare cross-swap / entity-swap / typo-mutation against P4-AC baseline.")
    lines.append("")
    lines.append("| Contamination mode | HGC phase | Accuracy | AC drop baseline | HGC drop |")
    lines.append("|---|---|---|---|---|")
    seed42 = RESULTS_DIR / "bcp_gpt41_seed42"
    ac_clean = phase_accuracy(load_phase(seed42, "P3-AC"))
    ac_contam = phase_accuracy(load_phase(seed42, "P4-AC"))
    ac_drop = ac_contam - ac_clean
    for hgc_phase, mode in [
        ("P4-Hybrid", "cross-swap"),
        ("E4-Hybrid", "entity-swap"),
        ("T4-Hybrid", "typo-mutation"),
    ]:
        hgc_clean = phase_accuracy(load_phase(seed42, "P3-Hybrid"))
        hgc_contam = phase_accuracy(load_phase(seed42, hgc_phase))
        hgc_drop = hgc_contam - hgc_clean
        lines.append(
            f"| {mode} | {hgc_phase} | {hgc_contam:.1f} | "
            f"AC: {ac_drop:+.1f} | HGC: {hgc_drop:+.1f} |"
        )
    lines.append("")

    # ==================================================================
    # Save report
    # ==================================================================
    report_path = OUTPUT_DIR / "final_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    main()
