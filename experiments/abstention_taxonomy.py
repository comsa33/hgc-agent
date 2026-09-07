"""Classify each prediction into {correct, honest_abstention, confident_wrong, empty_or_error}.

Honest abstention = the system explicitly says it could not determine the answer
(e.g., "cannot be determined", "unable to", "I don't know", "not specified",
"no information available"). A prediction is confident_wrong only when the
judge said incorrect AND the prediction is not an abstention.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from hgc.paths import results_root

CODE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = results_root(CODE_ROOT)
OUT_DIR = CODE_ROOT / "analysis"
OUT_CSV = OUT_DIR / "abstention_taxonomy.csv"

# Patterns for honest abstention.  Case-insensitive substring match after whitespace-normalisation.
_ABSTAIN_PATTERNS = [
    # English
    r"cannot be (?:determined|found|established|verified|identified)",
    r"unable to (?:determine|find|identify|verify|extract|answer)",
    r"could not (?:determine|find|identify|verify|be determined)",
    r"not (?:determinable|available|provided|specified|stated|mentioned|given|found|identified|listed|known|found in|explicit)",
    r"no (?:information|data|evidence|mention|record|details?) (?:is |was )?(?:available|provided|given|found)",
    r"\bi (?:do not|don['']?t) know\b",
    r"\bnot enough information\b",
    r"\binsufficient information\b",
    r"\bnot (?:clearly )?specified\b",
    r"\bnot (?:explicitly )?(?:stated|mentioned)\b",
    r"\bthe (?:document|documents|source|context|evidence) (?:does not|do not|doesn['']?t|don['']?t) (?:contain|provide|mention|specify|include)\b",
    r"\bthere is no (?:information|mention|reference|data)\b",
    r"\bimpossible to (?:determine|find|identify)\b",
    r"\bno(?:t)? (?:possible|feasible) to (?:determine|identify|find)\b",
    r"\banswer (?:is )?unknown\b",
    r"\bunknown\.",
    r"^unknown$",
    r"\bn/?a\b$",
]
_ABSTAIN_RE = re.compile("|".join(f"({p})" for p in _ABSTAIN_PATTERNS), re.IGNORECASE)


def classify(pred: str, judgment_correct: bool) -> str:
    pred_norm = " ".join(pred.lower().split()) if pred else ""
    if not pred_norm.strip():
        return "empty_or_error"
    if judgment_correct:
        return "correct"
    # Judge said incorrect: is it an honest abstention or a confident wrong?
    if _ABSTAIN_RE.search(pred_norm):
        return "honest_abstention"
    return "confident_wrong"


def process() -> None:
    rows = []
    for exp_dir in sorted(RESULTS.iterdir()):
        if not exp_dir.is_dir():
            continue
        for phase_dir in sorted(exp_dir.iterdir()):
            if not phase_dir.is_dir():
                continue
            for tp in phase_dir.glob("trajectory_q*.json"):
                try:
                    d = json.loads(tp.read_text())
                except Exception:
                    continue
                cat = classify(d.get("pred") or "", bool(d.get("judgment_correct")))
                rows.append(
                    {
                        "experiment": exp_dir.name,
                        "phase": phase_dir.name,
                        "query_id": d.get("query_id", ""),
                        "judgment_correct": d.get("judgment_correct"),
                        "category": cat,
                        "pred_preview": (d.get("pred") or "")[:140].replace("\n", " "),
                    }
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "phase",
                "query_id",
                "judgment_correct",
                "category",
                "pred_preview",
            ],
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print()

    # Per-experiment per-phase aggregation
    def key(r):
        return (r["experiment"], r["phase"])

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    print(
        f"{'experiment':<30} {'phase':<18} {'n':>4} {'correct':>7} {'abstain':>7} {'wrong':>6} {'empty':>6}"
    )
    print("-" * 95)
    for k in sorted(groups):
        grp = groups[k]
        n = len(grp)
        c = Counter(x["category"] for x in grp)
        correct = c.get("correct", 0)
        abstain = c.get("honest_abstention", 0)
        wrong = c.get("confident_wrong", 0)
        empty = c.get("empty_or_error", 0)
        exp, phase = k
        print(
            f"{exp:<30} {phase:<18} {n:>4} "
            f"{correct:>4} ({100 * correct / n:>4.0f}%) "
            f"{abstain:>4} ({100 * abstain / n:>3.0f}%) "
            f"{wrong:>4} ({100 * wrong / n:>3.0f}%) "
            f"{empty:>4}"
        )


if __name__ == "__main__":
    process()
