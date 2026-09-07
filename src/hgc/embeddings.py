"""Azure OpenAI text-embedding-3-small wrapper with LRU cache."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from openai import AzureOpenAI


class Embedder:
    """Embedding client backed by Azure OpenAI text-embedding-3-small.

    Parameters
    ----------
    deployment:
        Azure deployment name.  Falls back to env var
        ``AZURE_OPENAI_EMBEDDING_DEPLOYMENT``, then to ``"text-embedding-3-small"``.
    api_key:
        Azure OpenAI API key.  Falls back to ``AZURE_OPENAI_API_KEY``.
    endpoint:
        Azure OpenAI endpoint URL.  Falls back to ``AZURE_OPENAI_ENDPOINT``.
    api_version:
        API version string.  Falls back to ``AZURE_OPENAI_API_VERSION``.
    """

    DIMS = 1536

    def __init__(
        self,
        deployment: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self.deployment = (
            deployment
            or os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or "text-embedding-3-small"
        )
        resolved_api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_api_version = api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        )

        self._client = AzureOpenAI(
            api_key=resolved_api_key,
            azure_endpoint=resolved_endpoint or "",
            api_version=resolved_api_version,
        )

        # lru_cache requires a hashable key; _embed_uncached returns a tuple.
        self._embed_cached = lru_cache(maxsize=1024)(self._embed_uncached)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_uncached(self, text: str) -> tuple:
        """Call the API and return the embedding as a plain tuple (hashable)."""
        import time

        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                response = self._client.embeddings.create(input=text, model=self.deployment)
                return tuple(response.data[0].embedding)
            except Exception as exc:
                last_exc = exc
                time.sleep(min(2**attempt, 8))
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Return a float32 array of shape (1536,) for *text*, cached up to 1024 entries."""
        vec = self._embed_cached(text)
        return np.array(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str], chunk_size: int = 32) -> np.ndarray:
        """Return a float32 array of shape (N, 1536) for *texts*.

        Splits into sub-batches of *chunk_size* to keep request payload small and
        retries each sub-batch up to 5 times with exponential backoff on transient
        errors (connection resets, DNS blips, timeouts).
        """
        import time

        if not texts:
            return np.empty((0, self.DIMS), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), chunk_size):
            sub = texts[start : start + chunk_size]
            last_exc: Exception | None = None
            for attempt in range(5):
                try:
                    response = self._client.embeddings.create(input=sub, model=self.deployment)
                    items = sorted(response.data, key=lambda d: d.index)
                    all_embeddings.extend(item.embedding for item in items)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    time.sleep(min(2**attempt, 8))
            if last_exc is not None:
                raise last_exc

        return np.array(all_embeddings, dtype=np.float32)
