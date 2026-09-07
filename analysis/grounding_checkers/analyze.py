"""G-1 analysis: discrimination collapse, approval rates, question sensitivity.

AUROC here is the probability that a clean cached answer outscores a poisoned
one under a given checker, computed by rank (ties counted as half), so 1.0 is
perfect separation and 0.5 is a coin flip. Two are reported per checker: one
against random cross-swap poison, one against targeted verbatim poison. The gap
between them is the quantity of interest -- it is the discrimination the checker
loses when the poison stops being foreign text and starts being the victim's own
document.

Point estimates only, with the n they rest on. No confidence intervals: the
sample here is a fixed enumeration of one contaminated cache, not a draw from a
population, and the paper avoids CI language for exactly this reason.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# Threshold above which each checker is treated as having approved the answer.
APPROVE_ABOVE = {
    "hhem": 0.5,
    "nli": 0.5,
    "lettucedetect": 0.5,
    "qa_relevance": 0.0,
}

MIN_WORDS = 7


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(random pos > random neg), ties at half. None when either side is empty."""
    if not pos or not neg:
        return None
    merged = sorted([(v, 0) for v in pos] + [(v, 1) for v in neg])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k, (_, side) in enumerate(merged) if side == 0)
    n_p, n_n = len(pos), len(neg)
    return (rank_sum_pos - n_p * (n_p + 1) / 2) / (n_p * n_n)


def load() -> list[dict]:
    rows = []
    for name in ("hhem", "nli", "msmarco", "lettuce"):
        path = os.path.join(HERE, f"scores_{name}.csv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["rid"] = int(r["rid"])
                r["score"] = float(r["score"])
                r["cand_words"] = int(r["cand_words"])
                r["truncated"] = int(r["truncated"])
                r["cand_in_retained"] = int(r["cand_in_retained"])
                r["doc_tokens"] = int(r["doc_tokens"])
                rows.append(r)
    return rows


def key(r: dict) -> tuple:
    return (r["bench"], r["checker"], r["model"], r["qvariant"])


def block(rows: list[dict], label: str, drop_lost_spans: bool) -> None:
    print(f"\n{'=' * 96}\n{label}\n{'=' * 96}")
    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if drop_lost_spans and not r["cand_in_retained"]:
            continue
        groups[key(r)][r["regime"]].append(r)

    hdr = (
        f"{'bench':<13} {'checker':<14} {'model':<32} {'q':<9} "
        f"{'AUROC rand':>10} {'AUROC targ':>10} {'drop':>7}  "
        f"{'appr clean':>10} {'appr rand':>9} {'appr targ':>9}   n(c/r/t)"
    )
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(groups):
        bench, checker, model, qv = k
        g = groups[k]
        clean = [r["score"] for r in g.get("clean", [])]
        rand = [r["score"] for r in g.get("random", [])]
        targ = [r["score"] for r in g.get("targeted", [])]
        a_r, a_t = auroc(clean, rand), auroc(clean, targ)
        thr = APPROVE_ABOVE[checker]

        def appr(v: list[float]) -> str:
            return f"{100 * sum(1 for x in v if x > thr) / len(v):.1f}%" if v else "-"

        drop = f"{a_r - a_t:+.3f}" if (a_r is not None and a_t is not None) else "-"
        print(
            f"{bench:<13} {checker:<14} {model:<32} {qv:<9} "
            f"{(f'{a_r:.3f}' if a_r is not None else '-'):>10} "
            f"{(f'{a_t:.3f}' if a_t is not None else '-'):>10} {drop:>7}  "
            f"{appr(clean):>10} {appr(rand):>9} {appr(targ):>9}   "
            f"{len(clean)}/{len(rand)}/{len(targ)}"
        )


def mean_scores(rows: list[dict], min_words: int) -> None:
    """Mean support score per regime.

    Where the targeted mean sits above the clean mean, the checker is not merely
    failing to reject the poison -- it rates a sentence that answers nothing as
    better supported than the answer a judge marked correct.
    """
    print(f"\n{'=' * 96}\nMEAN SCORE BY REGIME  (>= {min_words} words, truncation-lost dropped)\n{'=' * 96}")
    g: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["cand_words"] < min_words or not r["cand_in_retained"]:
            continue
        g[key(r)][r["regime"]].append(r["score"])
    hdr = f"{'bench':<13} {'checker':<14} {'model':<32} {'q':<9} {'clean':>9} {'random':>9} {'targeted':>9}   targ>clean?"
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(g):
        m = {reg: (sum(v) / len(v) if v else float("nan")) for reg, v in g[k].items()}
        flag = "YES" if m.get("targeted", 0) > m.get("clean", 0) else ""
        print(
            f"{k[0]:<13} {k[1]:<14} {k[2]:<32} {k[3]:<9} "
            f"{m.get('clean', float('nan')):>9.3f} {m.get('random', float('nan')):>9.3f} "
            f"{m.get('targeted', float('nan')):>9.3f}   {flag}"
        )


def question_sensitivity(rows: list[dict], min_words: int | None) -> None:
    print(f"\n{'=' * 96}\nQUESTION-SHUFFLE SENSITIVITY (Tier B only)"
          f"{'  [>=7 words]' if min_words else '  [all]'}\n{'=' * 96}")
    by: dict[tuple, dict[str, dict[str, float]]] = defaultdict(dict)
    for r in rows:
        if r["qvariant"] not in ("orig", "shuffled"):
            continue
        if min_words and r["cand_words"] < min_words:
            continue
        by[(r["bench"], r["checker"], r["model"], r["regime"])].setdefault(r["qvariant"], {})
        by[(r["bench"], r["checker"], r["model"], r["regime"])][r["qvariant"]][r["rid"]] = r["score"]

    hdr = (
        f"{'bench':<13} {'checker':<14} {'model':<24} {'regime':<9} "
        f"{'mean|d|':>9} {'median|d|':>10} {'flip%':>7} {'appr orig':>10} {'appr shuf':>10}   n"
    )
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(by):
        bench, checker, model, regime = k
        o, s = by[k].get("orig", {}), by[k].get("shuffled", {})
        rids = sorted(set(o) & set(s))
        if not rids:
            continue
        thr = APPROVE_ABOVE[checker]
        deltas = sorted(abs(o[i] - s[i]) for i in rids)
        flips = sum(1 for i in rids if (o[i] > thr) != (s[i] > thr))
        ao = 100 * sum(1 for i in rids if o[i] > thr) / len(rids)
        as_ = 100 * sum(1 for i in rids if s[i] > thr) / len(rids)
        print(
            f"{bench:<13} {checker:<14} {model:<24} {regime:<9} "
            f"{sum(deltas) / len(deltas):>9.4f} {deltas[len(deltas) // 2]:>10.4f} "
            f"{100 * flips / len(rids):>6.1f}% {ao:>9.1f}% {as_:>9.1f}%   {len(rids)}"
        )


def truncation_report(rows: list[dict]) -> None:
    print(f"\n{'=' * 96}\nTRUNCATION\n{'=' * 96}")
    hdr = (
        f"{'bench':<13} {'checker':<14} {'model':<32} "
        f"{'maxdoc tok':>10} {'truncated':>10} {'span lost':>10}   n"
    )
    print(hdr)
    print("-" * len(hdr))
    seen: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["qvariant"] == "shuffled":
            continue
        seen[(r["bench"], r["checker"], r["model"])].append(r)
    for k in sorted(seen):
        g = seen[k]
        t = sum(r["truncated"] for r in g)
        lost = sum(1 for r in g if not r["cand_in_retained"])
        lost_t = sum(1 for r in g if not r["cand_in_retained"] and r["regime"] == "targeted")
        print(
            f"{k[0]:<13} {k[1]:<14} {k[2]:<32} {max(r['doc_tokens'] for r in g):>10} "
            f"{t:>10} {f'{lost} ({lost_t} targ)':>10}   {len(g)}"
        )


def main() -> None:
    rows = load()
    print(f"loaded {len(rows)} score rows")
    with open(os.path.join(HERE, "triplets.jsonl"), encoding="utf-8") as fh:
        trip = [json.loads(x) for x in fh if x.strip()]
    n_by = defaultdict(int)
    for t in trip:
        n_by[(t["bench"], t["regime"])] += 1
    print("triplets:", dict(n_by))

    # The 20% subset the shipped contaminators actually sampled at run time, so
    # the full enumeration can be checked against the entries that were really
    # in play during the reported HGC experiments.
    victim_rid = {t["rid"] for t in trip if t["was_run_victim"]}
    for r in rows:
        r["was_run_victim"] = int(r["rid"] in victim_rid)

    main_rows = [r for r in rows if r["cand_words"] >= MIN_WORDS]
    block(main_rows, f"MAIN RESULT  (candidates >= {MIN_WORDS} words, truncation-lost spans dropped)", True)
    block(rows, "APPENDIX  (all candidates, no word filter, truncation-lost spans dropped)", True)
    block(main_rows, f"APPENDIX  (>= {MIN_WORDS} words, truncation-lost spans KEPT)", False)
    block(
        [r for r in main_rows if r["was_run_victim"]],
        f"ROBUSTNESS  (>= {MIN_WORDS} words, only the 20% entries the shipped run actually poisoned)",
        True,
    )
    mean_scores(rows, MIN_WORDS)
    question_sensitivity(rows, MIN_WORDS)
    truncation_report(rows)
    capped = sum(1 for r in rows if r["extra"] == "cand_capped")
    print(f"\nNLI candidate-cap events (hypothesis > 250 tokens, forced out of scope): {capped}")


if __name__ == "__main__":
    main()
