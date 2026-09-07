"""Tests for hgc.embeddings.Embedder."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hgc.embeddings import Embedder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embedding_data(index: int = 0, dims: int = 1536) -> MagicMock:
    """Return a mock embedding data object matching the OpenAI response shape."""
    item = MagicMock()
    item.index = index
    item.embedding = [float(i % 256) / 255.0 for i in range(dims)]
    return item


def _fake_response(texts: list[str], dims: int = 1536) -> MagicMock:
    """Return a mock CreateEmbeddingResponse for *texts*."""
    response = MagicMock()
    response.data = [_fake_embedding_data(i, dims) for i in range(len(texts))]
    return response


# ---------------------------------------------------------------------------
# Unit tests (no real API calls)
# ---------------------------------------------------------------------------


class TestEmbedderUnit:
    """All tests here patch AzureOpenAI so no network traffic is made."""

    def _make_embedder(self) -> tuple[Embedder, MagicMock]:
        """Return (Embedder instance, mock embeddings.create callable)."""
        with patch("hgc.embeddings.AzureOpenAI") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            embedder = Embedder(
                deployment="text-embedding-3-small",
                api_key="fake-key",
                endpoint="https://fake.openai.azure.com/",
                api_version="2024-12-01-preview",
            )
            return embedder, mock_client.embeddings.create

    def test_embed_returns_float32_array_shape_1536(self) -> None:
        embedder, mock_create = self._make_embedder()
        mock_create.return_value = _fake_response(["hello"])

        result = embedder.embed("hello")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1536,)

    def test_embed_batch_returns_correct_shape(self) -> None:
        embedder, mock_create = self._make_embedder()
        texts = ["foo", "bar", "baz"]
        mock_create.return_value = _fake_response(texts)

        result = embedder.embed_batch(texts)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (3, 1536)
        mock_create.assert_called_once()  # single API call

    def test_embed_batch_empty_list(self) -> None:
        embedder, mock_create = self._make_embedder()

        result = embedder.embed_batch([])

        assert result.shape == (0, 1536)
        mock_create.assert_not_called()

    def test_embed_cache_hit_calls_api_once(self) -> None:
        """Calling embed() twice with the same text must result in one API call."""
        embedder, mock_create = self._make_embedder()
        mock_create.return_value = _fake_response(["cached text"])

        r1 = embedder.embed("cached text")
        r2 = embedder.embed("cached text")

        assert mock_create.call_count == 1
        np.testing.assert_array_equal(r1, r2)

    def test_embed_cache_miss_calls_api_each_time(self) -> None:
        """Different texts should each trigger an API call."""
        embedder, mock_create = self._make_embedder()
        mock_create.side_effect = [
            _fake_response(["text A"]),
            _fake_response(["text B"]),
        ]

        embedder.embed("text A")
        embedder.embed("text B")

        assert mock_create.call_count == 2

    def test_deployment_fallback_order(self) -> None:
        """Deployment falls back to env var then to 'text-embedding-3-small'."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("hgc.embeddings.AzureOpenAI"):
                e = Embedder(api_key="k", endpoint="https://ep/", api_version="v")
                assert e.deployment == "text-embedding-3-small"

        with patch.dict(
            os.environ,
            {"AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "my-custom-deployment"},
            clear=True,
        ):
            with patch("hgc.embeddings.AzureOpenAI"):
                e2 = Embedder(api_key="k", endpoint="https://ep/", api_version="v")
                assert e2.deployment == "my-custom-deployment"


# ---------------------------------------------------------------------------
# Integration test (skipped without real credentials)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("AZURE_OPENAI_API_KEY"),
    reason="AZURE_OPENAI_API_KEY not set — skipping integration test",
)
class TestEmbedderIntegration:
    def test_real_embed_shape_and_dtype(self) -> None:
        embedder = Embedder()
        result = embedder.embed("What is the capital of France?")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1536,)
        # Sanity check: embeddings should be roughly unit-normalised
        norm = float(np.linalg.norm(result))
        assert 0.9 < norm < 1.1, f"Unexpected norm: {norm}"

    def test_real_embed_batch(self) -> None:
        embedder = Embedder()
        texts = ["Hello world", "Goodbye world"]
        result = embedder.embed_batch(texts)

        assert result.shape == (2, 1536)
        assert result.dtype == np.float32
