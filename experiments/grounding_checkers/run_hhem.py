"""Tier A: Vectara HHEM-2.1-open. Input is (premise, hypothesis) only.

Needs transformers<5 (the model's remote code predates the 5.x tied-weight
bookkeeping). Run inside venv_hhem.

HHEM's own predict() tokenises with padding but no truncation and no
max_length, so nothing is cut; token counts are recorded anyway so the
long-document behaviour can be inspected alongside the 512-token checkers.
"""

from __future__ import annotations

import os
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, load_rows, writer  # noqa: E402

MODEL = "vectara/hallucination_evaluation_model"
BATCH = 8

rows = load_rows()
model = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=True)
model.eval()
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")

# HHEM feeds the whole premise to a T5 encoder with no truncation, so a
# FinanceBench filing at ~3.5k tokens costs quadratic attention. Batch by a
# token budget rather than a fixed count, longest documents alone.
for r in rows:
    r["_ntok"] = len(tok(r["doc"], add_special_tokens=False)["input_ids"]) + len(
        tok(r["candidate"], add_special_tokens=False)["input_ids"]
    )
order = sorted(rows, key=lambda r: r["_ntok"])
batches: list[list[dict]] = []
for r in order:
    size = max(1, min(BATCH, 12000 // max(r["_ntok"], 1)))
    if batches and len(batches[-1]) < size and batches[-1][0]["_ntok"] >= r["_ntok"] // 2:
        batches[-1].append(r)
    else:
        batches.append([r])

fh, w = writer(os.path.join(HERE, "scores_hhem.csv"))
t0 = time.time()
for start, chunk in enumerate(batches):
    with torch.no_grad():
        scores = model.predict([(r["doc"], r["candidate"]) for r in chunk]).tolist()
    for r, s in zip(chunk, scores):
        w.writerow(
            {
                "rid": r["rid"],
                "bench": r["bench"],
                "regime": r["regime"],
                "checker": "hhem",
                "model": "HHEM-2.1-open",
                "qvariant": "none",
                "score": s,
                "cand_words": r["cand_words"],
                "doc_tokens": len(tok(r["doc"], add_special_tokens=False)["input_ids"]),
                "cand_tokens": len(tok(r["candidate"], add_special_tokens=False)["input_ids"]),
                "truncated": 0,
                "cand_in_retained": 1,
                "extra": "",
            }
        )
    if start % 40 == 0:
        print(f"  batch {start}/{len(batches)}  {time.time() - t0:.0f}s", flush=True)
fh.close()
print(f"hhem done: {len(rows)} rows in {time.time() - t0:.0f}s")
