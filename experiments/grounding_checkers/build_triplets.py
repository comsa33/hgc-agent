"""G-1 triplet builder.

Reconstructs, offline and with zero LLM calls, the two poison regimes that the
HGC contamination experiments inject into the AnswerCache, plus the clean
control, for every seeded cache entry (not just the 20% that were actually
sampled as victims at run time).

Sources (read-only):
  <repo>/results/adaptive_qasper_oracle/P1-LC/*.json
  <repo>/results/adaptive_finbench_lc/P1-LC/*.json

The logic mirrors, verbatim in behaviour:
  src/hgc/contaminators/gate_aware.py :: usable_spans / contaminate_answer_cache_from_p1
  src/hgc/contaminators/cross_swap.py :: contaminate_answer_cache_from_p1

Deviation, deliberate and documented: the shipped contaminators mutate the
cache in place while walking a 20% victim list, so a later victim can draw a
donor that an earlier iteration already poisoned. Generating every index means
that cascade would compound, so donors here are always drawn from the *clean*
answer pool. For the 20% subset the shipped code actually ran, the two agree
except where a donor happened to be an earlier victim.
"""

from __future__ import annotations

import json
import os
import random
import re
from math import floor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "analysis", "grounding_checkers")

SEED = 42
FRACTION = 0.20
MIN_CHARS = 5
SHUFFLE_SEED = 1234

_DOCID_HEADER = re.compile(r"\[docid=[^\]]*\]\s*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

BENCHES = {
    "qasper": f"{REPO}/results/adaptive_qasper_oracle/P1-LC",
    "financebench": f"{REPO}/results/adaptive_finbench_lc/P1-LC",
}


def normalise(text: str) -> str:
    """Match SupportVerifier._normalise so containment is judged identically."""
    return " ".join(text.lower().split())


def usable_spans(context: str, gold: str, min_chars: int = MIN_CHARS) -> list[str]:
    body = _DOCID_HEADER.sub("", context)
    document = normalise(body)
    gold_norm = normalise(gold)
    spans: list[str] = []
    for raw in _SENTENCE_SPLIT.split(body):
        span = raw.strip()
        if len(span) < min_chars:
            continue
        span_norm = normalise(span)
        if gold_norm and gold_norm in span_norm:
            continue
        if span_norm not in document:
            continue
        spans.append(span)
    return spans


def seeded_records(p1_dir: str) -> list[dict]:
    """P1 records in the order seed_answer_cache_from_p1 appends them."""
    records = []
    for path in sorted(
        os.path.join(p1_dir, f) for f in os.listdir(p1_dir) if f.endswith(".json")
    ):
        data = json.loads(open(path, encoding="utf-8").read())
        question = data.get("question", "")
        answer = data.get("pred", data.get("answer", ""))
        if not question or not answer or not data.get("judgment_correct", False):
            continue
        records.append(data)
    return records


def build(bench: str, p1_dir: str) -> list[dict]:
    records = seeded_records(p1_dir)
    n = len(records)
    docs = [_DOCID_HEADER.sub("", (r.get("trajectory") or {}).get("context", "")) for r in records]
    clean = [r["pred"] for r in records]

    # Which indices the shipped 20% run actually corrupted, so the subset that
    # was really served can be flagged in the output.
    victims = set(random.Random(SEED).sample(range(n), floor(FRACTION * n)))

    rows: list[dict] = []
    for i, rec in enumerate(records):
        doc = docs[i]
        if not doc:
            continue

        # -- targeted (gate-aware verbatim span) --------------------------------
        spans = usable_spans((rec.get("trajectory") or {}).get("context", ""), rec.get("gold", ""))
        targeted = random.Random(SEED + i).choice(spans) if spans else None

        # -- random (cross-swap donor from a different entry) -------------------
        candidates = [j for j in range(n) if j != i]
        rng_copy = random.Random(SEED + i)
        rng_copy.shuffle(candidates)
        donor = next((j for j in candidates if clean[j] != clean[i]), candidates[0])

        # -- question shuffle: a question from a different document -------------
        srng = random.Random(SHUFFLE_SEED + i)
        pool = [j for j in range(n) if j != i and docs[j] != doc]
        shuf_j = srng.choice(pool) if pool else srng.choice(candidates)

        common = {
            "bench": bench,
            "idx": i,
            "query_id": rec["query_id"],
            "question": rec["question"],
            "gold": rec.get("gold", ""),
            "doc": doc,
            "doc_chars": len(doc),
            "was_run_victim": i in victims,
            "shuf_question": records[shuf_j]["question"],
            "shuf_from": records[shuf_j]["query_id"],
        }
        for regime, cand, src in (
            ("clean", clean[i], rec["query_id"]),
            ("targeted", targeted, rec["query_id"]),
            ("random", clean[donor], records[donor]["query_id"]),
        ):
            if cand is None:
                continue
            rows.append(
                {
                    **common,
                    "regime": regime,
                    "candidate": cand,
                    "candidate_src": src,
                    "cand_words": len(cand.split()),
                    "cand_chars": len(cand),
                    "verbatim_in_doc": normalise(cand) in normalise(doc),
                }
            )
    return rows


def main() -> None:
    allrows: list[dict] = []
    for bench, p1 in BENCHES.items():
        rows = build(bench, p1)
        allrows.extend(rows)
        by = {}
        for r in rows:
            by[r["regime"]] = by.get(r["regime"], 0) + 1
        kept = {
            k: sum(1 for r in rows if r["regime"] == k and r["cand_words"] >= 7)
            for k in by
        }
        print(f"== {bench}")
        print(f"   all       : {by}")
        print(f"   >=7 words : {kept}")
        vb = {
            k: sum(1 for r in rows if r["regime"] == k and r["verbatim_in_doc"])
            for k in by
        }
        print(f"   verbatim in doc: {vb}")

    path = os.path.join(OUT, "triplets.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for i, r in enumerate(allrows):
            r["rid"] = i
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(allrows)} rows -> {path}")


if __name__ == "__main__":
    main()
