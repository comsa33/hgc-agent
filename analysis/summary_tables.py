"""Build every table in the paper from the trajectories.

One entry point for every experiment so the paper never quotes a number
that was transcribed by hand:

  E1  LC/RAG baselines re-run on the fixed code
  E2  gate-component ablation extended off BCP
  E3  parameter sensitivity
  E4  gate-aware threat model
  E6-E8  the three repairs: no fast path, reworded predicate, two-stage gate

Accuracy is read from the trajectories rather than the phase summaries, since
the path census and the per-stage gate counts only exist there. Cells that
have not been run yet are skipped with a note — a missing directory must never
be silently reported as a zero.

    uv run python analysis/summary_tables.py
    uv run python analysis/summary_tables.py --json analysis/summary_tables.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from hgc.paths import results_root

_REPO = Path(__file__).resolve().parents[1]
_RESULTS = results_root(_REPO)

# (label, results dir, clean phase, contaminated phase)
BASELINES = [
    ("QASPER-Oracle", "base_qasper_oracle", "HGC-LC", "C-HGC-LC"),
    ("FinanceBench-LC", "base_finbench_lc", "HGC-LC", "C-HGC-LC"),
    ("QASPER-RAG", "base_qasper_rag", "HGC-RAG", "C-HGC-RAG"),
]
AC_PHASES = {
    "QASPER-Oracle": ("AC-LC", "C-AC-LC"),
    "FinanceBench-LC": ("AC-LC", "C-AC-LC"),
    "QASPER-RAG": ("AC-RAG", "C-AC-RAG"),
}
ABLATION_PREFIX = {
    "QASPER-Oracle": "abl_qasper_oracle",
    "FinanceBench-LC": "abl_financebench_lc",
    "QASPER-RAG": "abl_qasper_rag",
}
# (label, base dir, clean phase, random-contaminated phase, attacked dir, attacked
# phase). The BCP rows of tab:main are the April cohort; the gate-aware BCP run
# re-runs the clean and random cells in its own out-dir on current code and is
# compared within it, never against the April rows.
ADAPTIVE = [
    (
        "QASPER-Oracle",
        "base_qasper_oracle",
        "HGC-LC",
        "C-HGC-LC",
        "adaptive_qasper_oracle",
        "A-HGC-LC",
    ),
    (
        "FinanceBench-LC",
        "base_finbench_lc",
        "HGC-LC",
        "C-HGC-LC",
        "adaptive_finbench_lc",
        "A-HGC-LC",
    ),
    (
        "QASPER-RAG",
        "base_qasper_rag",
        "HGC-RAG",
        "C-HGC-RAG",
        "adaptive_qasper_rag",
        "A-HGC-RAG",
    ),
    (
        "BCP s=42 (rerun)",
        "adaptive_bcp_s42",
        "P3-Hybrid",
        "P4-Hybrid",
        "adaptive_bcp_s42",
        "A-Hybrid",
    ),
]
# (repair, cell, attacked dir, clean dir, attacked phase, clean phase). The
# clean directory is the same configuration run without the attack, so a drop
# is never read across two different gates. FinanceBench has no containment-off
# clean run, so that one row is measured against the released gate's clean cell
# and says so.
REPAIRS = [
    (
        "no fast path",
        "QASPER-Oracle",
        "contoff_adaptive_qasper_oracle",
        "contoff_sens_qasper_oracle",
        "A-HGC-LC",
        "HGC-LC",
    ),
    (
        "no fast path",
        "FinanceBench-LC",
        "contoff_adaptive_finbench_lc",
        "base_finbench_lc",
        "A-HGC-LC",
        "HGC-LC",
    ),
    (
        "answerhood predicate",
        "QASPER-Oracle",
        "pred_contoff_qasper_oracle",
        "pred_sens_qasper_oracle",
        "A-HGC-LC",
        "HGC-LC",
    ),
    (
        "answerhood predicate",
        "FinanceBench-LC",
        "pred_contoff_finbench_lc",
        "base_finbench_lc",
        "A-HGC-LC",
        "HGC-LC",
    ),
    (
        "answerhood, fast path on",
        "QASPER-Oracle",
        "pred_adaptive_qasper_oracle",
        "pred_sens_qasper_oracle",
        "A-HGC-LC",
        "HGC-LC",
    ),
    (
        "two-stage gate",
        "QASPER-Oracle",
        "twostage_adaptive_qasper_oracle",
        "twostage_sens_qasper_oracle",
        "A-HGC-LC",
        "HGC-LC",
    ),
    (
        "two-stage gate",
        "FinanceBench-LC",
        "twostage_adaptive_finbench_lc",
        "twostage_sens_finbench_lc",
        "A-HGC-LC",
        "HGC-LC",
    ),
    (
        "two-stage gate",
        "QASPER-RAG",
        "twostage_adaptive_qasper_rag",
        "twostage_sens_qasper_rag",
        "A-HGC-RAG",
        "HGC-RAG",
    ),
    (
        "two-stage gate",
        "BCP s=42 (rerun)",
        "twostage_adaptive_bcp_s42",
        "twostage_sens_bcp_s42",
        "A-Hybrid",
        "P3-Hybrid",
    ),
    (
        "two-stage, judge sees doc",
        "QASPER-RAG",
        "twostagedoc_qasper_rag",
        "twostagedoc_qasper_rag",
        "A-HGC-RAG",
        "HGC-RAG",
    ),
    (
        "two-stage, judge sees doc",
        "BCP s=42 (rerun)",
        "twostagedoc_bcp_s42",
        "twostagedoc_bcp_s42",
        "A-Hybrid",
        "P3-Hybrid",
    ),
]

# Two-stage over-rejection check: on clean and randomly contaminated memory the
# two-stage gate should accept exactly the hits the released gate accepts.
# (cell, released dir, two-stage dir, clean phase, random phase)
TWOSTAGE_ACCEPTANCE = [
    (
        "QASPER-Oracle",
        "base_qasper_oracle",
        "twostage_sens_qasper_oracle",
        "HGC-LC",
        "C-HGC-LC",
    ),
    (
        "FinanceBench-LC",
        "base_finbench_lc",
        "twostage_sens_finbench_lc",
        "HGC-LC",
        "C-HGC-LC",
    ),
    (
        "QASPER-RAG",
        "base_qasper_rag",
        "twostage_sens_qasper_rag",
        "HGC-RAG",
        "C-HGC-RAG",
    ),
    # The two-stage clean cell comes from twostage_sens_bcp_s42, whose
    # P3-Hybrid started from the pristine P1 store; its P4-Hybrid ran after
    # that P3 in the same invocation and inherited the hints P3 added, so the
    # random cell is read from a separate run on a pristine copy.
    (
        "BCP s=42 (rerun)",
        "adaptive_bcp_s42",
        ("twostage_sens_bcp_s42", "twostage_random_bcp_s42"),
        "P3-Hybrid",
        "P4-Hybrid",
    ),
]

# Clean-memory acceptance under the document-aware judge, against the released
# gate and the question-only judge. (cell, released dir, two-stage dir,
# two-stage-doc dir, clean phase)
CLEAN_ACCEPTANCE_DOC = [
    (
        "QASPER-RAG",
        "base_qasper_rag",
        "twostage_sens_qasper_rag",
        "twostagedoc_qasper_rag",
        "HGC-RAG",
    ),
    (
        "BCP s=42 (rerun)",
        "adaptive_bcp_s42",
        "twostage_sens_bcp_s42",
        "twostagedoc_bcp_s42",
        "P3-Hybrid",
    ),
]

SENSITIVITY = [
    ("tau=0.80", "sens_tau080_qasper_oracle", "QASPER-Oracle"),
    ("tau=0.90", "sens_tau090_qasper_oracle", "QASPER-Oracle"),
    ("k_h=1", "sens_kh1_qasper_oracle", "QASPER-Oracle"),
    ("k_h=5", "sens_kh5_qasper_oracle", "QASPER-Oracle"),
    ("containment=1", "sens_cont1_qasper_oracle", "QASPER-Oracle"),
    ("containment=20", "sens_cont20_qasper_oracle", "QASPER-Oracle"),
    ("max_doc=1000", "sens_doc1000_finbench_lc", "FinanceBench-LC"),
    ("max_doc=5000", "sens_doc5000_finbench_lc", "FinanceBench-LC"),
]


def read_phase(results_dir: str, phase: str) -> dict | None:
    paths = glob.glob(str(_RESULTS / results_dir / phase / "trajectory_q*.json"))
    if not paths:
        return None
    n = correct = errors = 0
    paths_census: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    stages: dict[str, int] = defaultdict(int)
    agent_times: dict[str, list[float]] = defaultdict(list)
    gate_tokens = tokens = 0
    for p in paths:
        t = json.loads(Path(p).read_text(encoding="utf-8"))
        n += 1
        correct += bool(t.get("judgment_correct"))
        errors += bool(t.get("error"))
        tokens += int(t.get("tokens") or 0)
        gate_tokens += int(t.get("gate_tokens") or 0)
        path = t.get("path") or "none"
        paths_census[path][0] += 1
        paths_census[path][1] += bool(t.get("judgment_correct"))
        for k, v in (t.get("gate_stage_counts") or {}).items():
            stages[k] += v
        wall = t.get("agent_wall_time")
        if wall is not None:
            agent_times[path].append(float(wall))
    return {
        "n": n,
        "correct": correct,
        "acc": 100 * correct / n,
        "errors": errors,
        "tokens": tokens,
        "gate_tokens": gate_tokens,
        "paths": {k: v for k, v in paths_census.items()},
        "stages": dict(stages),
        "median_agent_time": {k: sorted(v)[len(v) // 2] for k, v in agent_times.items() if v},
    }


def _drop(clean: dict | None, contam: dict | None) -> float | None:
    if not clean or not contam:
        return None
    return contam["acc"] - clean["acc"]


def _verified(cell: dict | None) -> str:
    if not cell:
        return "-"
    n, c = cell["paths"].get("cache_verified", [0, 0])
    return f"{c}/{n} ({100 * c / n:.0f}%)" if n else "-"


def _wrong(cell: dict | None) -> str:
    """Accepted cache hits that were judged wrong, as tab:adaptive reports them."""
    if not cell:
        return "-"
    n, c = cell["paths"].get("cache_verified", [0, 0])
    return f"{n - c}/{n}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {"e1": {}, "e2": {}, "e3": {}, "e4": {}, "repairs": {}}

    print("=" * 78)
    print("E1  contamination drop, released defaults on the fixed code")
    print("=" * 78)
    print(f"{'cell':17s} {'AC drop':>9s} {'HGC drop':>10s} {'HGC verified':>16s} {'errors':>7s}")
    for label, d, ph, cph in BASELINES:
        clean, contam = read_phase(d, ph), read_phase(d, cph)
        ac_ph, ac_cph = AC_PHASES[label]
        ac_clean, ac_contam = read_phase(d, ac_ph), read_phase(d, ac_cph)
        if not contam:
            print(f"{label:17s} (not run)")
            continue
        ac_drop, hgc_drop = _drop(ac_clean, ac_contam), _drop(clean, contam)
        errs = sum(x["errors"] for x in (clean, contam, ac_clean, ac_contam) if x)
        print(
            f"{label:17s} {ac_drop:+8.1f}pp {hgc_drop:+9.1f}pp {_verified(contam):>16s} {errs:>7d}"
        )
        report["e1"][label] = {"ac_drop": ac_drop, "hgc_drop": hgc_drop, "errors": errs}

    print()
    print("=" * 78)
    print("E2  gate-component ablation")
    print("=" * 78)
    print(f"{'cell':17s} {'gate':16s} {'clean':>7s} {'contam':>8s} {'drop':>9s} {'verified':>15s}")
    for label, base, ph, cph in BASELINES:
        pre = ABLATION_PREFIX[label]
        for name, d in [
            ("full G1+G2+G3", base),
            ("G3 off (G1+G2)", f"{pre}_g1g2"),
            ("G2 off (G1+G3)", f"{pre}_g1g3"),
        ]:
            clean, contam = read_phase(d, ph), read_phase(d, cph)
            if not contam:
                print(f"{'':17s} {name:16s} (not run)")
                continue
            print(
                f"{label if name.startswith('full') else '':17s} {name:16s} "
                f"{clean['acc']:6.1f}% {contam['acc']:7.1f}% {_drop(clean, contam):+8.1f}pp "
                f"{_verified(contam):>15s}"
            )
            report["e2"].setdefault(label, {})[name] = {
                "clean": clean["acc"],
                "contam": contam["acc"],
                "drop": _drop(clean, contam),
                "stages": contam["stages"],
            }
        print()

    print("=" * 78)
    print("E4  gate-aware threat model")
    print("=" * 78)
    print(
        f"{'cell':17s} {'random (C-*)':>13s} {'gate-aware':>12s} {'verified under attack':>22s}"
        f" {'wrong: random':>14s} {'wrong: attack':>14s}"
    )
    for label, base, ph, cph, adir, aph in ADAPTIVE:
        clean = read_phase(base, ph)
        random_contam = read_phase(base, cph)
        adaptive = read_phase(adir, aph)
        if not adaptive or not clean or not random_contam:
            print(f"{label:17s} (not run)")
            continue
        print(
            f"{label:17s} {_drop(clean, random_contam):+12.1f}pp "
            f"{_drop(clean, adaptive):+11.1f}pp {_verified(adaptive):>22s}"
            f" {_wrong(random_contam):>14s} {_wrong(adaptive):>14s}"
        )
        report["e4"][label] = {
            "random_drop": _drop(clean, random_contam),
            "adaptive_drop": _drop(clean, adaptive),
            "wrong_random": _wrong(random_contam),
            "wrong_adaptive": _wrong(adaptive),
            "paths": adaptive["paths"],
            "gate_tokens": adaptive["gate_tokens"],
            "gate_tokens_random": random_contam["gate_tokens"],
            "clean_dir": base,
        }

    print()
    print("=" * 78)
    print("E3  parameter sensitivity")
    print("=" * 78)
    print(
        f"{'setting':16s} {'cell':17s} {'clean':>7s} {'contam':>8s} {'drop':>9s} {'gate tok':>10s}"
    )
    for label, base, ph, cph in BASELINES:
        b_clean, b_contam = read_phase(base, ph), read_phase(base, cph)
        if b_contam:
            print(
                f"{'released default':16s} {label:17s} {b_clean['acc']:6.1f}% "
                f"{b_contam['acc']:7.1f}% {_drop(b_clean, b_contam):+8.1f}pp "
                f"{b_contam['gate_tokens']:>10d}"
            )
    for name, d, cell in SENSITIVITY:
        ph, cph = ("HGC-RAG", "C-HGC-RAG") if "RAG" in cell else ("HGC-LC", "C-HGC-LC")
        clean, contam = read_phase(d, ph), read_phase(d, cph)
        if not contam:
            print(f"{name:16s} {cell:17s} (not run)")
            continue
        print(
            f"{name:16s} {cell:17s} {clean['acc']:6.1f}% {contam['acc']:7.1f}% "
            f"{_drop(clean, contam):+8.1f}pp {contam['gate_tokens']:>10d}"
        )
        report["e3"][name] = {
            "cell": cell,
            "clean": clean["acc"],
            "contam": contam["acc"],
            "drop": _drop(clean, contam),
            "gate_tokens": contam["gate_tokens"],
        }

    print()
    print("=" * 78)
    print("E6-E8  the three repairs, each against a clean cell of its own gate")
    print("=" * 78)
    print(f"{'repair':26s} {'cell':17s} {'drop':>9s} {'accepted':>10s} {'wrong':>7s}")
    report["repairs"] = {}
    for repair, cell, attacked_dir, clean_dir, aph, ph in REPAIRS:
        attacked, clean = read_phase(attacked_dir, aph), read_phase(clean_dir, ph)
        if not attacked or not clean:
            print(f"{repair:26s} {cell:17s} (not run)")
            continue
        n, correct = attacked["paths"].get("cache_verified", [0, 0])
        matched = "  (cross-condition)" if clean_dir.startswith("base") else ""
        print(
            f"{repair:26s} {cell:17s} {_drop(clean, attacked):+8.1f}pp {n:>10d} {n - correct:>7d}"
            f"{matched}"
        )
        report["repairs"][f"{repair} / {cell}"] = {
            "drop": _drop(clean, attacked),
            "accepted": n,
            "wrong": n - correct,
            "clean_dir": clean_dir,
        }

    print()
    print("=" * 78)
    print("Two-stage over-rejection: accepted hits (wrong) on clean / random memory")
    print("=" * 78)
    print(
        f"{'cell':17s} {'released clean':>15s} {'two-stage clean':>16s} "
        f"{'released random':>16s} {'two-stage random':>17s}"
    )
    report["twostage_acceptance"] = {}
    for cell, rdir, tdir, ph, cph in TWOSTAGE_ACCEPTANCE:
        tdir_clean, tdir_random = tdir if isinstance(tdir, tuple) else (tdir, tdir)
        cells = [
            read_phase(rdir, ph),
            read_phase(tdir_clean, ph),
            read_phase(rdir, cph),
            read_phase(tdir_random, cph),
        ]
        if not all(cells):
            print(f"{cell:17s} (not run)")
            continue

        def _acc(c: dict) -> str:
            n, k = c["paths"].get("cache_verified", [0, 0])
            return f"{n} ({n - k})"

        print(f"{cell:17s} " + " ".join(f"{_acc(c):>16s}" for c in cells))
        report["twostage_acceptance"][cell] = {
            k: c["paths"].get("cache_verified", [0, 0])
            for k, c in zip(
                ("released_clean", "twostage_clean", "released_random", "twostage_random"),
                cells,
                strict=True,
            )
        }

    print()
    print("=" * 78)
    print("Clean-memory acceptance: released / judge (q,a) / judge (q,a,doc)")
    print("=" * 78)
    report["clean_acceptance_doc"] = {}
    for cell, rdir, tdir, ddir, ph in CLEAN_ACCEPTANCE_DOC:
        cells = [read_phase(rdir, ph), read_phase(tdir, ph), read_phase(ddir, ph)]
        if not all(cells):
            print(f"{cell:17s} (not run)")
            continue
        parts = []
        for c in cells:
            n, k = c["paths"].get("cache_verified", [0, 0])
            parts.append(f"{n} ({n - k}) {c['gate_tokens'] // 1000}k tok, acc {c['acc']:.1f}")
        print(f"{cell:17s} " + " | ".join(parts))
        report["clean_acceptance_doc"][cell] = {
            k: {
                "cache_verified": c["paths"].get("cache_verified", [0, 0]),
                "gate_tokens": c["gate_tokens"],
                "acc": c["acc"],
            }
            for k, c in zip(("released", "two_stage", "two_stage_doc"), cells, strict=True)
        }

    print()
    print("=" * 78)
    print("Latency by path — agent time only, judge excluded (median seconds)")
    print("=" * 78)
    for label, base, ph, cph in BASELINES:
        for tag, phase in (("clean", ph), ("contam", cph)):
            cell = read_phase(base, phase)
            if not cell:
                continue
            spans = "  ".join(f"{k}={v:.2f}s" for k, v in sorted(cell["median_agent_time"].items()))
            print(f"  {label:17s} {tag:7s} {spans}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
