"""FinanceBench dataset loader for HGC experiment pipeline.

PatronusAI/financebench (Islam et al., 2023) — CC-BY-NC-4.0.
License: Non-commercial use only. Academic research publication is permitted
under the non-commercial academic use interpretation. Commercial deployment
requires a separate license from Patronus AI (contact@patronus.ai).

Open-source subset: 150 questions across financial filings (10-K, 10-Q, etc.)
for approximately 50–60 unique company filings, 2020–2022.

Each question yields one query record shaped like QASPERDataset / BCPDataset:
    {
        "query_id": str,                  # "financebench_<row_idx>"
        "question": str,
        "answer": str,                    # gold (human-annotated)
        "docs": list[{"docid": str,       # f"{doc_name}_page_{i}"
                       "text": str}],     # evidence_text_full_page pages
        "doc_name": str,                  # e.g. "JNJ_2021_10K"
        "doc_period": str,                # e.g. "2021"
        "question_type": str,             # metrics-generated | domain-relevant | novel-generated
    }

PDF parsing is NOT required: evidence_text_full_page provides pre-extracted
page text for every evidence item. For a full-document pilot, download PDFs
from doc_link and parse with pdfminer.six or pypdf.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PARAPHRASE_PROMPT = """\
Rephrase the following question so it asks for the same information but uses \
different wording. Preserve the specific details and constraints. Return only \
the rephrased question, no preamble.

Question: {query}
"""


def _make_default_lm(deployment: str | None = None) -> Any:
    import dspy  # deferred so tests can import without dspy installed

    dep = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    return dspy.LM(
        model=f"azure/{dep}",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_base=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0.7,
        max_tokens=512,
    )


def _hash_key(text: str) -> str:
    """Return a 16-char SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _build_docs_from_evidence(doc_name: str, evidence_list: list) -> list[dict]:
    """Convert evidence items to a doc list using evidence_text_full_page.

    Each evidence item is a dict with keys including:
      - evidence_text_full_page: str (full page text)
      - evidence_page_num: int (optional)

    If evidence_list is empty or all pages are blank, returns an empty list
    (caller should fall back to justification).
    """
    docs: list[dict] = []
    seen_pages: set = set()
    for i, ev in enumerate(evidence_list):
        if not isinstance(ev, dict):
            continue
        page_text = ev.get("evidence_text_full_page", "")
        if not page_text or not page_text.strip():
            continue
        page_num = ev.get("evidence_page_num", i)
        page_key = (doc_name, page_num)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        docs.append(
            {
                "docid": f"{doc_name}_page_{page_num}",
                "text": page_text,
            }
        )
    return docs


def _row_to_record(row: dict, row_idx: int) -> dict:
    """Convert one FinanceBench row to a query record.

    Fallback chain for docs:
      1. Build docs from evidence[*].evidence_text_full_page.
      2. If no usable evidence pages, use justification as a single-doc entry.
      3. If justification is also absent/empty, use the question itself as
         minimal placeholder text to keep docs non-empty.
    """
    doc_name: str = row.get("doc_name") or f"doc_{row_idx}"
    doc_period: str = str(row.get("doc_period", ""))
    question: str = row.get("question", "")
    answer: str = row.get("answer", "")
    justification: str = row.get("justification") or ""
    question_type: str = row.get("question_type", "")

    # Build docs from evidence list
    evidence_list = row.get("evidence", []) or []
    docs = _build_docs_from_evidence(doc_name, evidence_list)

    # Fallback: justification as single doc
    if not docs:
        fallback_text = justification.strip() if justification.strip() else question.strip()
        docs = [
            {
                "docid": f"{doc_name}_fallback",
                "text": fallback_text,
            }
        ]

    return {
        "query_id": f"financebench_{row_idx}",
        "query": question,  # runner.py reads qr["query"]
        "question": question,  # kept for API symmetry
        "answer": answer,
        "docs": docs,
        "doc_name": doc_name,
        "doc_period": doc_period,
        "question_type": question_type,
    }


class FinanceBenchDataset:
    """Loader for PatronusAI/financebench (Islam et al., 2023).

    License: CC-BY-NC-4.0. Non-commercial use only.

    Lazy-loads the HuggingFace dataset inside __init__ so tests can mock
    datasets.load_dataset before instantiation.

    Parameters
    ----------
    cache_dir:
        Directory for HF dataset cache and derived JSON files.
    """

    def __init__(self, cache_dir: str = "data/financebench") -> None:
        self.cache_dir = Path(cache_dir)
        self._cache_path = self.cache_dir / "paraphrase_cache.json"
        self._lm: Any = None
        self._hf_data: Any = None  # lazy-loaded

    def _snapshot_path(self, n: int, seed: int) -> Path:
        return self.cache_dir / f"dataset_n{n}_seed{seed}.json"

    # ------------------------------------------------------------------
    # Core loader
    # ------------------------------------------------------------------

    def _load_hf(self) -> Any:
        """Load HuggingFace dataset (lazy, cached in self._hf_data)."""
        if self._hf_data is None:
            from datasets import load_dataset  # deferred — not always installed

            self._hf_data = load_dataset(
                "PatronusAI/financebench",
                split="train",
                cache_dir=str(self.cache_dir),
            )
        return self._hf_data

    def load_all(self) -> list[dict]:
        """Return all 150 query records from the open-source subset."""
        hf_ds = self._load_hf()
        records: list[dict] = []
        for row_idx, row in enumerate(hf_ds):
            records.append(_row_to_record(row, row_idx))
        logger.info("Loaded %d FinanceBench query records", len(records))
        return records

    def select_N(self, seed: int = 42, n: int = 150) -> list[dict]:
        """Return a deterministic list of *n* FinanceBench records.

        Selection rule:
          1. Load all records.
          2. Sort by query_id (lexicographic).
          3. Seed a random.Random(seed) and shuffle in place.
          4. Return first n.

        The n parameter exists so the runner can shrink to a smoke-test
        subset (e.g. n=5) without changing the seed contract.
        """
        records = self.load_all()
        records.sort(key=lambda r: r["query_id"])
        rng = random.Random(seed)
        rng.shuffle(records)
        return records[:n]

    # ------------------------------------------------------------------
    # Paraphrase
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            with self._cache_path.open() as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: dict) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._cache_path.parent), suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(self._cache_path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def paraphrase(self, query_text: str, lm: Any = None) -> str:
        """Return a paraphrased version of *query_text*.

        Cache is keyed by sha256(query_text)[:16]. On cache hit no LM call
        is made. On miss the LM is called, the cache is updated atomically.
        """
        key = _hash_key(query_text)
        cache = self._load_cache()

        if key in cache:
            return cache[key]

        effective_lm = lm or self._lm or _make_default_lm()
        prompt = _PARAPHRASE_PROMPT.format(query=query_text)
        raw = effective_lm(prompt)
        if isinstance(raw, list):
            raw = raw[0] if raw else query_text
        result = str(raw).strip()

        cache[key] = result
        self._save_cache(cache)
        return result

    def paraphrases(self, seed: int = 42, lm: Any = None) -> dict[str, str]:
        """Return {query_id: paraphrased_question} for the deterministic n=150 subset.

        Calls self.paraphrase() for each question; results are disk-cached.
        """
        records = self.select_N(seed=seed, n=150)
        result: dict[str, str] = {}
        for rec in records:
            result[rec["query_id"]] = self.paraphrase(rec["question"], lm=lm)
        return result

    # ------------------------------------------------------------------
    # load_or_build
    # ------------------------------------------------------------------

    def load_or_build(
        self,
        n: int = 150,
        seed: int = 42,
        paraphrase: bool = False,
        lm: Any = None,
    ) -> dict:
        """Return ``{queries: [...], paraphrased: [...]}`` aligned by index.

        Cached to ``dataset_n{n}_seed{seed}.json`` so different n values do not
        collide (mirrors BCPDataset / QASPERDataset load_or_build).
        """
        snapshot_path = self._snapshot_path(n, seed)
        if snapshot_path.exists():
            with snapshot_path.open() as f:
                cached = json.load(f)
            if len(cached.get("queries", [])) == n:
                return cached

        queries = self.select_N(seed=seed, n=n)

        if paraphrase:
            paraphrased = [self.paraphrase(q["question"], lm=lm) for q in queries]
        else:
            paraphrased = [q["question"] for q in queries]

        dataset = {"queries": queries, "paraphrased": paraphrased}
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        return dataset
