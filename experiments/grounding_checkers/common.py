"""Shared loading / truncation bookkeeping for the G-1 checker runs."""

from __future__ import annotations

import csv
import json
import os

# Scripts live under experiments/, the data they read and write under analysis/.
HERE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "analysis",
    "grounding_checkers",
)
TRIPLETS = os.path.join(HERE, "triplets.jsonl")


def load_rows() -> list[dict]:
    with open(TRIPLETS, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    limit = os.environ.get("G1_LIMIT")  # smoke runs only
    return rows[: int(limit)] if limit else rows


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def pair_truncation(tok, doc: str, cand: str, max_length: int, n_special: int = 3) -> dict:
    """What survives `truncation='only_first'` on a (doc, cand) pair.

    Reports whether the candidate string is still present in the retained
    portion of the document. When it is not, any verdict the checker returns is
    uninterpretable for our purpose: the span it is being asked about is no
    longer in the evidence it was shown.
    """
    doc_ids = tok(doc, add_special_tokens=False)["input_ids"]
    cand_ids = tok(cand, add_special_tokens=False)["input_ids"]
    budget = max_length - len(cand_ids) - n_special
    truncated = len(doc_ids) > budget
    retained = tok.decode(doc_ids[: max(budget, 0)]) if truncated else doc
    in_full = normalise(cand) in normalise(doc)
    # 1 means "nothing was lost that mattered": either the candidate was never a
    # verbatim span of the document (clean / cross-swap, where there is nothing
    # for truncation to cut away) or it is still inside the retained evidence.
    survives = (not in_full) or (normalise(cand) in normalise(retained))
    return {
        "doc_tokens": len(doc_ids),
        "cand_tokens": len(cand_ids),
        "truncated": int(truncated),
        "cand_in_retained": int(survives),
    }


FIELDS = [
    "rid",
    "bench",
    "regime",
    "checker",
    "model",
    "qvariant",
    "score",
    "cand_words",
    "doc_tokens",
    "cand_tokens",
    "truncated",
    "cand_in_retained",
    "extra",
]


def writer(path: str):
    fh = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    return fh, w
