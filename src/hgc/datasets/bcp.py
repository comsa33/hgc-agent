"""Deterministic BrowseComp-Plus (BCP) dataset loader with paraphrase cache.

Selects a fixed subset of queries by ``(sort by query_id, seeded shuffle)``
and materialises paraphrased versions with an on-disk cache.
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
    import dspy

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
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class BCPDataset:
    """Loader for BrowseComp-Plus with deterministic query selection and paraphrase cache.

    ``load_or_build(n, seed)`` chooses its on-disk cache file from
    ``data_dir/dataset_n{n}_seed{seed}.json`` so that subsequent runs at a
    different *n* do not silently re-read a stale snapshot. The paraphrase
    cache is keyed by sha256(query_text)[:16], so growing *n* only pays the
    paraphrase cost for the newly added queries.
    """

    def __init__(
        self,
        queries_path: str = "data/bcp/queries.jsonl",
        cache_path: str = "data/bcp/paraphrase_cache.json",
        data_dir: str | None = None,
    ) -> None:
        self.queries_path = Path(queries_path)
        self.cache_path = Path(cache_path)
        self.data_dir = Path(data_dir) if data_dir else self.queries_path.parent
        self._lm: Any = None

    def _dataset_path(self, n: int, seed: int) -> Path:
        return self.data_dir / f"dataset_n{n}_seed{seed}.json"

    def load_all(self) -> list[dict]:
        records = []
        with self.queries_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def select_n(self, n: int = 100, seed: int = 42) -> list[dict]:
        """Deterministic subset: sort by int(query_id), seeded shuffle, take first n."""
        records = self.load_all()
        records.sort(key=lambda r: int(r["query_id"]))
        rng = random.Random(seed)
        rng.shuffle(records)
        return records[:n]

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.cache_path.parent), suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(self.cache_path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def paraphrase(self, query_text: str, lm: Any = None) -> str:
        """Return a paraphrased version of *query_text* (disk-cached by sha256[:16])."""
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

    def load_or_build(
        self,
        n: int = 100,
        seed: int = 42,
        paraphrase: bool = True,
        lm: Any = None,
    ) -> dict:
        """Return ``{"queries": [...], "paraphrased": [...]}`` aligned by index.

        The on-disk snapshot is ``dataset_n{n}_seed{seed}.json`` so that a call
        at ``n=200`` does not silently reuse a cached ``n=100`` snapshot from
        a prior session. The paraphrase cache is global and key-by-hash, so
        growing *n* only pays the paraphrase cost for the new queries.
        """
        snapshot_path = self._dataset_path(n, seed)
        if snapshot_path.exists() and self.cache_path.exists():
            with snapshot_path.open() as f:
                cached = json.load(f)
            if len(cached.get("queries", [])) == n:
                return cached

        queries = self.select_n(n=n, seed=seed)
        paraphrased = (
            [self.paraphrase(q["query"], lm=lm) for q in queries]
            if paraphrase
            else [q["query"] for q in queries]
        )

        dataset = {"queries": queries, "paraphrased": paraphrased}
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        return dataset
