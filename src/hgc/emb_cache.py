"""Persistent disk-backed document embedding cache.

Keyed by docid (str); backed by a numpy .npz file.  A single shared instance
can be passed across multiple NaiveRAGAgent calls to eliminate redundant
embed_batch API calls when the same BCP docs appear across queries/phases.

File format
-----------
A compressed .npz with exactly two arrays:
  ``docids``     — object array (strings), shape (N,)
  ``embeddings`` — float32 2-D array, shape (N, dim)

The file is written atomically via a sibling temp file + os.replace() so a
crash mid-write never leaves a corrupt cache on disk.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


class DocEmbeddingCache:
    """In-memory dict of {docid: embedding} with optional disk persistence.

    Parameters
    ----------
    path:
        Path to the ``.npz`` cache file.  If *None* the cache is in-memory only
        and ``save()`` is a no-op.  If the file exists it is loaded at
        construction time; a missing file starts an empty cache silently.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path: Path | None = Path(path) if path is not None else None
        self._store: dict[str, np.ndarray] = {}
        self._dim: int | None = None

        if self._path is not None and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load cache from ``self._path``.  Silently skips on any error."""
        if self._path is None:
            return
        try:
            data = np.load(str(self._path), allow_pickle=True)
            docids: np.ndarray = data["docids"]
            embeddings: np.ndarray = data["embeddings"].astype(np.float32)
            for docid, vec in zip(docids, embeddings, strict=False):
                self._store[str(docid)] = vec
            if embeddings.shape[0] > 0:
                self._dim = embeddings.shape[1]
        except Exception:
            # Corrupt/empty file — start fresh
            self._store = {}
            self._dim = None

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get(self, docid: str) -> np.ndarray | None:
        """Return the cached embedding for *docid*, or None on a miss."""
        return self._store.get(docid)

    def get_many(self, docids: list[str]) -> dict[str, np.ndarray]:
        """Return a dict of {docid: embedding} for every *docid* that hits.

        Docs absent from the cache are simply omitted (no KeyError).
        """
        result: dict[str, np.ndarray] = {}
        for docid in docids:
            vec = self._store.get(docid)
            if vec is not None:
                result[docid] = vec
        return result

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def put(self, docid: str, vec: np.ndarray) -> None:
        """Store *vec* under *docid*.

        Parameters
        ----------
        docid:
            String key.
        vec:
            1-D float32 numpy array.  The first call establishes the expected
            dimension; subsequent calls with a different shape raise ValueError.

        Raises
        ------
        ValueError
            If *vec*.shape[0] does not match the dimension inferred from the
            first stored vector.
        """
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"Expected 1-D vector, got shape {vec.shape}")
        if self._dim is None:
            self._dim = vec.shape[0]
        elif vec.shape[0] != self._dim:
            raise ValueError(f"Dimension mismatch: cache dim={self._dim}, got {vec.shape[0]}")
        self._store[docid] = vec

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Atomically write the cache to ``self._path``.

        Uses a sibling temp file + ``os.replace()`` so a crash mid-write
        never leaves a corrupt or partial file at the target path.

        If ``path`` was not provided at construction this is a no-op.
        """
        if self._path is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)

        if not self._store:
            # Nothing to save yet; don't overwrite an existing file with empty data
            return

        docids = np.array(list(self._store.keys()), dtype=object)
        embeddings = np.stack(list(self._store.values())).astype(np.float32)

        # np.savez_compressed always appends ".npz" to the filename it receives.
        # Use a stem without the extension so the real output file is stem+".npz",
        # then rename that to the final target path.
        dir_ = str(self._path.parent)
        fd, tmp_stem = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        tmp_npz = tmp_stem + ".npz"
        try:
            os.close(fd)
            os.unlink(tmp_stem)  # remove the placeholder; numpy will create tmp_npz
            np.savez_compressed(tmp_stem, docids=docids, embeddings=embeddings)
            os.replace(tmp_npz, str(self._path))
        except Exception:
            # Clean up any leftover temp files
            for p in (tmp_stem, tmp_npz):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            raise

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int | None:
        """Embedding dimension inferred from the first stored vector, or None."""
        return self._dim

    def size(self) -> int:
        """Return the number of cached embeddings."""
        return len(self._store)
