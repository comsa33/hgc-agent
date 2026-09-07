"""QASPER dataset loader for HGC experiment pipeline.

allenai/qasper (Dasigi et al., NAACL 2021) — CC-BY-4.0.
Validation split: 281 papers / 1,005 questions.

Each question yields one query record shaped like BCPDataset:
    {
        "query_id": str,          # f"{paper_id}_q{qas_idx}"
        "question": str,
        "answer": str,            # gold: extractive | abstractive | yes/no | UNANSWERABLE
        "docs": [{"docid": str, "text": str}],  # paper sections
        "paper_id": str,
        "answer_type": str,       # 'extractive', 'abstractive', 'yesno', 'unanswerable'
    }
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

# Sections with this prefix are figure/table floats — excluded from docs.
_FLOAT_PREFIX = "FLOAT SELECTED"


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


def _extract_answer(answers_list: list[dict]) -> tuple[str, str]:
    """Return (gold_answer_text, answer_type) from a list of annotator answers.

    Priority: extractive_spans > free_form_answer > yes_no > unanswerable.
    Uses the first annotator answer object.
    """
    if not answers_list:
        return "UNANSWERABLE", "unanswerable"

    ann = answers_list[0]  # first annotator

    # Each answer object has sub-lists keyed by type
    extractive = ann.get("extractive_spans", [])
    if extractive:
        return " ".join(extractive), "extractive"

    free_form = ann.get("free_form_answer", "")
    if free_form and free_form.strip():
        return free_form.strip(), "abstractive"

    yes_no = ann.get("yes_no")
    if yes_no is not None:
        return "yes" if yes_no else "no", "yesno"

    return "UNANSWERABLE", "unanswerable"


def _build_docs(paper_id: str, full_text: dict) -> list[dict]:
    """Convert full_text (section_name + paragraphs) to doc list.

    Skips sections whose name starts with FLOAT SELECTED.
    Each section becomes one doc:  docid=f"qasper_{paper_id}_sec_{i}",
    text = "\\n\\n".join(paragraphs[i]).
    """
    section_names: list[str] = full_text.get("section_name", [])
    paragraphs_list: list[list[str]] = full_text.get("paragraphs", [])
    docs: list[dict] = []
    for i, (sec_name, paras) in enumerate(zip(section_names, paragraphs_list, strict=False)):
        if sec_name is None:
            # Some QASPER papers have null section names (e.g. the introductory
            # paragraph with no heading); keep them as valid docs.
            sec_name = ""
        if sec_name.startswith(_FLOAT_PREFIX):
            continue
        text = "\n\n".join(paras) if paras else ""
        if not text.strip():
            continue
        docs.append(
            {
                "docid": f"qasper_{paper_id}_sec_{i}",
                "text": text,
            }
        )
    return docs


def _build_oracle_docs(paper_id: str, q_idx: int, ann_list: list[dict]) -> list[dict]:
    """Build Oracle-setting docs: joined evidence sentences annotator marked.

    Each question's answer contains ``evidence`` (list of strings marked by the
    annotator). Joining them produces a single-section "Oracle doc" containing
    (most likely) the information needed to answer — modelling Patronus-style
    Oracle retrieval where the evidence pages are pre-identified.
    """
    evidence_texts: list[str] = []
    for ann in ann_list:
        ev = ann.get("evidence", []) if isinstance(ann, dict) else []
        for e in ev:
            if isinstance(e, str) and e.strip():
                evidence_texts.append(e.strip())
    # Deduplicate preserving order
    seen: set[str] = set()
    unique_evidence = []
    for e in evidence_texts:
        if e not in seen:
            seen.add(e)
            unique_evidence.append(e)
    text = "\n\n".join(unique_evidence)
    if not text:
        text = "(no evidence annotated for this question)"
    return [{"docid": f"qasper_{paper_id}_q{q_idx}_oracle", "text": text}]


def _paper_to_records(paper: dict, oracle: bool = False) -> list[dict]:
    """Convert one QASPER paper row to a list of query records (one per question).

    Parameters
    ----------
    oracle:
        If True, each query's ``docs`` is a single Oracle doc built from that
        question's annotator-marked evidence (no full-paper retrieval). If
        False (default), every query in the paper shares the full set of
        paper sections as docs (RAG setting).
    """
    paper_id = paper.get("id", "unknown")
    full_text = paper.get("full_text", {})
    paper_docs = None if oracle else _build_docs(paper_id, full_text)

    qas_list = paper.get("qas", {})
    # HuggingFace column format: qas is a dict with list-valued fields
    questions: list[str] = qas_list.get("question", [])
    answers_outer: list = qas_list.get("answers", [])

    records: list[dict] = []
    for q_idx, question in enumerate(questions):
        # answers_outer[q_idx] is a dict with key "answer" -> list of annotator dicts
        if q_idx < len(answers_outer):
            ann_container = answers_outer[q_idx]
            ann_list = ann_container.get("answer", []) if isinstance(ann_container, dict) else []
        else:
            ann_list = []

        gold, ans_type = _extract_answer(ann_list)
        query_id = f"{paper_id}_q{q_idx}"
        docs = _build_oracle_docs(paper_id, q_idx, ann_list) if oracle else paper_docs
        records.append(
            {
                "query_id": query_id,
                "query": question,  # runner.py reads qr["query"] (BCP-compatible)
                "question": question,  # kept for API symmetry and tests
                "answer": gold,
                "docs": docs,
                "paper_id": paper_id,
                "answer_type": ans_type,
            }
        )
    return records


class QASPERDataset:
    """Loader for allenai/qasper (NAACL 2021).

    Lazy-loads the HuggingFace dataset inside __init__ so tests can mock
    datasets.load_dataset before instantiation.

    Parameters
    ----------
    cache_dir:
        Directory for HF dataset cache and derived JSON files.
    split:
        HuggingFace split to use (default: "validation").
    """

    def __init__(
        self,
        cache_dir: str = "data/qasper",
        split: str = "validation",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
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
                "allenai/qasper",
                split=self.split,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True,
            )
        return self._hf_data

    def load_all(self, oracle: bool = False) -> list[dict]:
        """Return all query records across all papers in the split.

        When ``oracle=True`` each record's ``docs`` contains only the
        annotator-marked evidence sentences instead of the full paper.
        """
        hf_ds = self._load_hf()
        records: list[dict] = []
        for paper in hf_ds:
            records.extend(_paper_to_records(paper, oracle=oracle))
        logger.info(
            "Loaded %d QASPER query records from split=%s (oracle=%s)",
            len(records), self.split, oracle,
        )
        return records

    def select_N(self, n: int = 100, seed: int = 42, oracle: bool = False) -> list[dict]:
        """Return a deterministic list of *n* QASPER records."""
        records = self.load_all(oracle=oracle)
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
        """Return {query_id: paraphrased_question} for the deterministic n=100 subset.

        Calls self.paraphrase() for each question; results are disk-cached.
        """
        records = self.select_N(n=100, seed=seed)
        result: dict[str, str] = {}
        for rec in records:
            result[rec["query_id"]] = self.paraphrase(rec["question"], lm=lm)
        return result

    # ------------------------------------------------------------------
    # load_or_build
    # ------------------------------------------------------------------

    def load_or_build(
        self,
        n: int = 100,
        seed: int = 42,
        paraphrase: bool = False,
        lm: Any = None,
        oracle: bool = False,
    ) -> dict:
        """Return ``{queries: [...], paraphrased: [...]}`` aligned by index.

        Cached to ``dataset_n{n}_seed{seed}{_oracle}.json`` so oracle and
        RAG snapshots don't collide.
        """
        suffix = "_oracle" if oracle else ""
        snapshot_path = self.cache_dir / f"dataset_n{n}_seed{seed}{suffix}.json"
        if snapshot_path.exists():
            with snapshot_path.open() as f:
                cached = json.load(f)
            if len(cached.get("queries", [])) == n:
                return cached

        queries = self.select_N(n=n, seed=seed, oracle=oracle)

        if paraphrase:
            paraphrased = [self.paraphrase(q["question"], lm=lm) for q in queries]
        else:
            paraphrased = [q["question"] for q in queries]

        dataset = {"queries": queries, "paraphrased": paraphrased}
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        return dataset
