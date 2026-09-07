"""Tier A standard NLI (DeBERTa-v3-base MNLI-FEVER-ANLI) and Tier B QA
relevance (ms-marco MiniLM cross-encoder).

Both are 512-token models, so both record what the truncation actually cost:
how many documents were cut, and in how many of those the candidate span no
longer survives inside the retained evidence.

The NLI model never sees the question. The cross-encoder sees only the question
and the candidate — never the document — which is exactly the axis the
grounding checkers leave uncovered, so it runs on both the real question and a
question lifted from a different document.

Run inside .venv (transformers 5.x is fine for both).
"""

from __future__ import annotations

import os
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, load_rows, pair_truncation, writer  # noqa: E402

NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
XENC = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAXLEN = 512
BATCH = 8

rows = load_rows()

# ---------------------------------------------------------------- Tier A: NLI
tok = AutoTokenizer.from_pretrained(NLI)
model = AutoModelForSequenceClassification.from_pretrained(NLI)
model.eval()
entail_idx = [i for i, lab in model.config.id2label.items() if lab.lower().startswith("entail")][0]

CAND_CAP = 250  # tokens; a hypothesis longer than this cannot leave room for a premise


def fit_pair(doc: str, cand: str) -> tuple[str, str, bool]:
    """Cut the pair down to something that fits, premise first.

    `truncation="only_first"` throws outright when the hypothesis alone exceeds
    max_length, which some FinanceBench spans do — they are whole table blocks.
    Capping the hypothesis and giving the premise whatever is left keeps every
    item scoreable; the cap is recorded so those rows can be inspected apart.
    """
    cand_ids = tok(cand, add_special_tokens=False)["input_ids"]
    capped = len(cand_ids) > CAND_CAP
    cand_ids = cand_ids[:CAND_CAP]
    budget = MAXLEN - len(cand_ids) - 3
    doc_ids = tok(doc, add_special_tokens=False)["input_ids"][: max(budget, 1)]
    return tok.decode(doc_ids), tok.decode(cand_ids), capped


fh, w = writer(os.path.join(HERE, "scores_nli.csv"))
t0 = time.time()
for start in range(0, len(rows), BATCH):
    chunk = rows[start : start + BATCH]
    fitted = [fit_pair(r["doc"], r["candidate"]) for r in chunk]
    with torch.no_grad():
        enc = tok(
            [f[0] for f in fitted],
            [f[1] for f in fitted],
            return_tensors="pt",
            padding=True,
            truncation="longest_first",
            max_length=MAXLEN,
        )
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, entail_idx].tolist()
    for r, p, (_, _, capped) in zip(chunk, probs, fitted):
        trunc = pair_truncation(tok, r["doc"], r["candidate"], MAXLEN)
        if capped:
            # The candidate itself was cut, so the containment question is moot.
            trunc["cand_in_retained"] = 0
        w.writerow(
            {
                "rid": r["rid"],
                "bench": r["bench"],
                "regime": r["regime"],
                "checker": "nli",
                "model": "DeBERTa-v3-base-mnli-fever-anli",
                "qvariant": "none",
                "score": p,
                "cand_words": r["cand_words"],
                **trunc,
                "extra": "cand_capped" if capped else "",
            }
        )
    if start % 160 == 0:
        print(f"  nli {start}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
fh.close()
print(f"nli done in {time.time() - t0:.0f}s")

# -------------------------------------------------- Tier B: QA relevance rank
tok2 = AutoTokenizer.from_pretrained(XENC)
xenc = AutoModelForSequenceClassification.from_pretrained(XENC)
xenc.eval()

fh, w = writer(os.path.join(HERE, "scores_msmarco.csv"))
t0 = time.time()
for qvariant, qkey in (("orig", "question"), ("shuffled", "shuf_question")):
    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        with torch.no_grad():
            enc = tok2(
                [r[qkey] for r in chunk],
                [r["candidate"] for r in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAXLEN,
            )
            logits = xenc(**enc).logits.squeeze(-1).tolist()
        if not isinstance(logits, list):
            logits = [logits]
        for r, s in zip(chunk, logits):
            w.writerow(
                {
                    "rid": r["rid"],
                    "bench": r["bench"],
                    "regime": r["regime"],
                    "checker": "qa_relevance",
                    "model": "ms-marco-MiniLM-L6-v2",
                    "qvariant": qvariant,
                    "score": s,
                    "cand_words": r["cand_words"],
                    "doc_tokens": 0,
                    "cand_tokens": 0,
                    "truncated": 0,
                    "cand_in_retained": 1,
                    "extra": r["shuf_from"] if qvariant == "shuffled" else "",
                }
            )
    print(f"  msmarco {qvariant} done {time.time() - t0:.0f}s", flush=True)
fh.close()
print(f"msmarco done in {time.time() - t0:.0f}s")
