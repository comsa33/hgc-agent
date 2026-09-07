"""Tier B: LettuceDetect, flagship and tiny, on the real question and on a
question taken from a different document.

LettuceDetect is the one open checker that takes the question as a first-class
input and still runs locally for free, which makes it the load-bearing case: if
a question-aware detector waves the poison through, "your prompt was bad" stops
being an available explanation.

Score is an approval score in [0, 1]: 1 - max over answer tokens of
P(hallucinated). The span API thresholds the same quantity at argmax, so this
is the continuous form of the decision the library actually makes.

Both backbones are ModernBERT with an 8k window; max_length is set above every
document in the set so nothing is chunked or truncated. Run inside venv_lettuce.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, load_rows, writer  # noqa: E402

from lettucedetect.models.inference import HallucinationDetector  # noqa: E402

MODELS = [
    ("flagship", "KRLabsOrg/lettucedect-base-modernbert-en-v1", 7999),
    ("tiny", "KRLabsOrg/tinylettuce-ettin-68m-en", 7999),
]

rows = load_rows()
fh, w = writer(os.path.join(HERE, "scores_lettuce.csv"))

for tag, path, maxlen in MODELS:
    det = HallucinationDetector(method="transformer", model_path=path, max_length=maxlen)
    tok = det.detector.tokenizer
    for qvariant, qkey in (("orig", "question"), ("shuffled", "shuf_question")):
        t0 = time.time()
        for i, r in enumerate(rows):
            toks = det.predict(
                context=[r["doc"]],
                question=r[qkey],
                answer=r["candidate"],
                output_format="tokens",
            )
            probs = [t["prob"] for t in toks] or [0.0]
            n_flagged = sum(1 for t in toks if t["pred"] == 1)
            w.writerow(
                {
                    "rid": r["rid"],
                    "bench": r["bench"],
                    "regime": r["regime"],
                    "checker": "lettucedetect",
                    "model": tag,
                    "qvariant": qvariant,
                    "score": 1.0 - max(probs),
                    "cand_words": r["cand_words"],
                    "doc_tokens": len(tok(r["doc"], add_special_tokens=False)["input_ids"]),
                    "cand_tokens": len(tok(r["candidate"], add_special_tokens=False)["input_ids"]),
                    "truncated": 0,
                    "cand_in_retained": 1,
                    "extra": f"meanprob={sum(probs) / len(probs):.4f};flagged={n_flagged}/{len(toks)}",
                }
            )
            if i % 200 == 0:
                print(f"  {tag}/{qvariant} {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        print(f"  {tag}/{qvariant} done in {time.time() - t0:.0f}s", flush=True)

fh.close()
print("lettuce done")
