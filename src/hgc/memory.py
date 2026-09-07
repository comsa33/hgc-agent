"""
HintRecord dataclass + pluggable HintStore (SQLite default, Qdrant optional).
Implements DESIGN.md §1 (schema), §2 (scoring), §4 (update rules + pruning), §5 (storage).
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float32 vectors. Returns 0 if either is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _confidence(success_count: int, failure_count: int) -> float:
    """success / max(1, success + failure)  ∈ [0, 1]."""
    return success_count / max(1, success_count + failure_count)


def _recency(created_at: float, t_half_days: float = 30.0) -> float:
    """exp(-λ · Δt)  where λ = ln2 / T_half,  Δt in seconds."""
    lam = math.log(2) / (t_half_days * 86400)
    delta = time.time() - created_at
    return math.exp(-lam * delta)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class HintRecord:
    # Identity
    hint_id: str
    hint_type: Literal["location", "entity", "strategy"]
    polarity: Literal["positive", "negative"]

    # Content
    content: str
    content_meta: dict  # stored as JSON text in SQLite

    # Context
    query_ctx: str
    query_ctx_embedding: bytes  # float32 bytes (np array serialised)
    trajectory_step: int

    # Temporal
    created_at: float
    last_validated_at: float

    # Dynamics
    success_count: int
    failure_count: int
    retrieval_count: int

    # Scope
    scope_id: str = "default"


# ---------------------------------------------------------------------------
# SQLite schema helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS hints (
    hint_id            TEXT PRIMARY KEY,
    hint_type          TEXT NOT NULL,
    polarity           TEXT NOT NULL,
    content            TEXT NOT NULL,
    content_meta       TEXT NOT NULL,
    query_ctx          TEXT NOT NULL,
    query_ctx_embedding BLOB NOT NULL,
    trajectory_step    INTEGER NOT NULL,
    created_at         REAL NOT NULL,
    last_validated_at  REAL NOT NULL,
    success_count      INTEGER NOT NULL DEFAULT 0,
    failure_count      INTEGER NOT NULL DEFAULT 0,
    retrieval_count    INTEGER NOT NULL DEFAULT 0,
    scope_id           TEXT NOT NULL DEFAULT 'default'
)
"""


def _migrate_sqlite(conn: sqlite3.Connection) -> None:
    """Add scope_id column to existing databases that pre-date US-019."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(hints)").fetchall()}
    if "scope_id" not in existing:
        conn.execute("ALTER TABLE hints ADD COLUMN scope_id TEXT NOT NULL DEFAULT 'default'")
        conn.commit()


def _row_to_record(row: tuple) -> HintRecord:
    (
        hint_id,
        hint_type,
        polarity,
        content,
        content_meta_json,
        query_ctx,
        query_ctx_embedding_blob,
        trajectory_step,
        created_at,
        last_validated_at,
        success_count,
        failure_count,
        retrieval_count,
        scope_id,
    ) = row
    return HintRecord(
        hint_id=hint_id,
        hint_type=hint_type,
        polarity=polarity,
        content=content,
        content_meta=json.loads(content_meta_json),
        query_ctx=query_ctx,
        query_ctx_embedding=query_ctx_embedding_blob,
        trajectory_step=trajectory_step,
        created_at=created_at,
        last_validated_at=last_validated_at,
        success_count=success_count,
        failure_count=failure_count,
        retrieval_count=retrieval_count,
        scope_id=scope_id,
    )


# ---------------------------------------------------------------------------
# Backend protocol (private)
# ---------------------------------------------------------------------------


class _Backend:
    def add(self, record: HintRecord) -> str:
        raise NotImplementedError

    def get(self, hint_id: str) -> HintRecord | None:
        raise NotImplementedError

    def all(self) -> list[HintRecord]:
        raise NotImplementedError

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        alpha: float,
        beta: float,
        gamma: float,
        t_half_days: float,
        theta_pos: float,
        theta_neg: float,
        scope_id: str | None = None,
    ) -> list[HintRecord]:
        raise NotImplementedError

    def update_on_outcome(self, hint_ids: list[str], correct: bool) -> None:
        raise NotImplementedError

    def update_content(self, hint_id: str, new_content: str) -> None:
        raise NotImplementedError

    def delete(self, hint_id: str) -> None:
        raise NotImplementedError

    def prune(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class _SQLiteBackend(_Backend):
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        _migrate_sqlite(self._conn)

    def add(self, record: HintRecord) -> str:
        if not record.hint_id:
            record.hint_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT OR REPLACE INTO hints
            (hint_id, hint_type, polarity, content, content_meta,
             query_ctx, query_ctx_embedding, trajectory_step,
             created_at, last_validated_at,
             success_count, failure_count, retrieval_count, scope_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.hint_id,
                record.hint_type,
                record.polarity,
                record.content,
                json.dumps(record.content_meta),
                record.query_ctx,
                record.query_ctx_embedding,
                record.trajectory_step,
                record.created_at,
                record.last_validated_at,
                record.success_count,
                record.failure_count,
                record.retrieval_count,
                record.scope_id,
            ),
        )
        self._conn.commit()
        return record.hint_id

    def get(self, hint_id: str) -> HintRecord | None:
        cur = self._conn.execute("SELECT * FROM hints WHERE hint_id = ?", (hint_id,))
        row = cur.fetchone()
        return _row_to_record(row) if row else None

    def all(self) -> list[HintRecord]:
        cur = self._conn.execute("SELECT * FROM hints")
        return [_row_to_record(r) for r in cur.fetchall()]

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        alpha: float,
        beta: float,
        gamma: float,
        t_half_days: float,
        theta_pos: float,
        theta_neg: float,
        scope_id: str | None = None,
    ) -> list[HintRecord]:
        q_emb = query_embedding.astype(np.float32)
        candidates: list[tuple[float, HintRecord]] = []

        for h in self.all():
            # Scope filter: location hints are corpus-local (strict); entity/strategy
            # hints are cross-scope (loose).
            if scope_id is not None and h.hint_type == "location":
                if h.scope_id != scope_id:
                    continue

            h_emb = np.frombuffer(h.query_ctx_embedding, dtype=np.float32)
            sim = _cosine_sim(q_emb, h_emb)

            if h.polarity == "positive" and sim < theta_pos:
                continue
            if h.polarity == "negative" and sim < theta_neg:
                continue

            conf = _confidence(h.success_count, h.failure_count)
            rec = _recency(h.created_at, t_half_days)
            score = alpha * sim + beta * conf + gamma * rec
            candidates.append((score, h))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = [h for _, h in candidates[:k]]

        if top:
            placeholders = ",".join("?" * len(top))
            ids = [h.hint_id for h in top]
            sql = (
                "UPDATE hints SET retrieval_count = retrieval_count + 1"
                f" WHERE hint_id IN ({placeholders})"
            )
            self._conn.execute(sql, ids)
            self._conn.commit()
            for h in top:
                h.retrieval_count += 1

        return top

    def update_on_outcome(self, hint_ids: list[str], correct: bool) -> None:
        now = time.time()
        for hint_id in hint_ids:
            row = self._conn.execute(
                "SELECT success_count, failure_count FROM hints WHERE hint_id = ?",
                (hint_id,),
            ).fetchone()
            if row is None:
                continue
            success_count, failure_count = row

            if correct:
                self._conn.execute(
                    """
                    UPDATE hints
                    SET success_count = success_count + 1,
                        last_validated_at = ?
                    WHERE hint_id = ?
                    """,
                    (now, hint_id),
                )
            else:
                new_failure = failure_count + 1
                new_conf = _confidence(success_count, new_failure)
                if new_conf < 0.25:
                    self._conn.execute(
                        """
                        UPDATE hints
                        SET failure_count = ?,
                            polarity = 'negative'
                        WHERE hint_id = ?
                        """,
                        (new_failure, hint_id),
                    )
                else:
                    self._conn.execute(
                        "UPDATE hints SET failure_count = ? WHERE hint_id = ?",
                        (new_failure, hint_id),
                    )
        self._conn.commit()

    def update_content(self, hint_id: str, new_content: str) -> None:
        self._conn.execute(
            "UPDATE hints SET content = ? WHERE hint_id = ?",
            (new_content, hint_id),
        )
        self._conn.commit()

    def delete(self, hint_id: str) -> None:
        self._conn.execute("DELETE FROM hints WHERE hint_id = ?", (hint_id,))
        self._conn.commit()

    def prune(self) -> int:
        now = time.time()
        age_180 = now - 180 * 86400
        age_90 = now - 90 * 86400

        cur = self._conn.execute(
            """
            DELETE FROM hints
            WHERE (failure_count > 5 AND success_count = 0)
               OR (created_at < ? AND retrieval_count = 0)
               OR (polarity = 'negative' AND created_at < ?)
            """,
            (age_180, age_90),
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Qdrant backend
# ---------------------------------------------------------------------------

_QDRANT_VECTOR_SIZE = 1536


def _payload_to_record(point_id: str, payload: dict) -> HintRecord:
    embedding_list: list[float] = payload["query_ctx_embedding"]
    embedding_bytes = np.array(embedding_list, dtype=np.float32).tobytes()
    return HintRecord(
        hint_id=point_id,
        hint_type=payload["hint_type"],
        polarity=payload["polarity"],
        content=payload["content"],
        content_meta=json.loads(payload["content_meta"]),
        query_ctx=payload["query_ctx"],
        query_ctx_embedding=embedding_bytes,
        trajectory_step=int(payload["trajectory_step"]),
        created_at=float(payload["created_at"]),
        last_validated_at=float(payload["last_validated_at"]),
        success_count=int(payload["success_count"]),
        failure_count=int(payload["failure_count"]),
        retrieval_count=int(payload["retrieval_count"]),
        scope_id=str(payload.get("scope_id", "default")),
    )


def _record_to_payload(record: HintRecord) -> dict:
    embedding_array = np.frombuffer(record.query_ctx_embedding, dtype=np.float32)
    return {
        "hint_type": record.hint_type,
        "polarity": record.polarity,
        "content": record.content,
        "content_meta": json.dumps(record.content_meta),
        "query_ctx": record.query_ctx,
        "query_ctx_embedding": embedding_array.tolist(),
        "trajectory_step": record.trajectory_step,
        "created_at": record.created_at,
        "last_validated_at": record.last_validated_at,
        "success_count": record.success_count,
        "failure_count": record.failure_count,
        "retrieval_count": record.retrieval_count,
        "scope_id": record.scope_id,
    }


class _QdrantBackend(_Backend):
    """
    Qdrant-backed HintStore using local disk-persistent mode (no server required).

    Requires `qdrant-client` (optional dep). Import is lazy so users without
    the package are not blocked when using the default SQLite backend.

    Qdrant returns cosine *distance* in the range [0, 2] when using Distance.Cosine.
    We normalise: cosine_sim = 1 - distance / 2  → [0, 1].
    All post-filtering and scoring use the same formula as the SQLite backend.
    """

    def __init__(
        self,
        qdrant_path: str = "./qdrant_storage",
        collection_name: str = "hints",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for the Qdrant backend. "
                "Install it with: pip install 'hgc[qdrant]'"
            ) from exc

        self._collection = collection_name
        self._client = QdrantClient(path=qdrant_path)

        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=_QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_or_trim(self, vec: list[float]) -> list[float]:
        """Ensure vector is exactly _QDRANT_VECTOR_SIZE dimensions."""
        n = len(vec)
        if n == _QDRANT_VECTOR_SIZE:
            return vec
        if n < _QDRANT_VECTOR_SIZE:
            return vec + [0.0] * (_QDRANT_VECTOR_SIZE - n)
        return vec[:_QDRANT_VECTOR_SIZE]

    def _get_point(self, hint_id: str):
        """Retrieve a single point by id; returns None if not found."""
        results = self._client.retrieve(
            collection_name=self._collection,
            ids=[hint_id],
            with_payload=True,
            with_vectors=False,
        )
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def add(self, record: HintRecord) -> str:
        from qdrant_client.models import PointStruct

        if not record.hint_id:
            record.hint_id = str(uuid.uuid4())

        embedding_array = np.frombuffer(record.query_ctx_embedding, dtype=np.float32)
        vector = self._pad_or_trim(embedding_array.tolist())
        payload = _record_to_payload(record)

        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=record.hint_id, vector=vector, payload=payload)],
        )
        return record.hint_id

    def get(self, hint_id: str) -> HintRecord | None:
        point = self._get_point(hint_id)
        if point is None:
            return None
        return _payload_to_record(point.id, point.payload)

    def all(self) -> list[HintRecord]:
        records: list[HintRecord] = []
        offset = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=self._collection,
                with_payload=True,
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            for point in batch:
                records.append(_payload_to_record(point.id, point.payload))
            if next_offset is None:
                break
            offset = next_offset
        return records

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        alpha: float,
        beta: float,
        gamma: float,
        t_half_days: float,
        theta_pos: float,
        theta_neg: float,
        scope_id: str | None = None,
    ) -> list[HintRecord]:
        q_emb = query_embedding.astype(np.float32)
        vector = self._pad_or_trim(q_emb.tolist())

        # Request k*4 candidates from Qdrant for post-filtering headroom.
        fetch_limit = max(k * 4, 20)
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
        )
        hits = response.points

        candidates: list[tuple[float, HintRecord]] = []
        for hit in hits:
            # Qdrant Distance.Cosine returns score = 1 - cosine_distance.
            # score ∈ [-1, 1] (same as cosine similarity).
            # Clamp to [0, 1] to match our formula's expectation.
            sim = max(0.0, float(hit.score))

            rec = _payload_to_record(hit.id, hit.payload)

            # Scope filter: location hints are corpus-local (strict); entity/strategy
            # hints are cross-scope (loose).
            if scope_id is not None and rec.hint_type == "location":
                if rec.scope_id != scope_id:
                    continue

            if rec.polarity == "positive" and sim < theta_pos:
                continue
            if rec.polarity == "negative" and sim < theta_neg:
                continue

            conf = _confidence(rec.success_count, rec.failure_count)
            recency = _recency(rec.created_at, t_half_days)
            score = alpha * sim + beta * conf + gamma * recency
            candidates.append((score, rec))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = [h for _, h in candidates[:k]]

        if top:
            for h in top:
                point = self._get_point(h.hint_id)
                if point is None:
                    continue
                new_count = int(point.payload.get("retrieval_count", 0)) + 1
                self._client.set_payload(
                    collection_name=self._collection,
                    payload={"retrieval_count": new_count},
                    points=[h.hint_id],
                )
                h.retrieval_count = new_count

        return top

    def update_on_outcome(self, hint_ids: list[str], correct: bool) -> None:
        now = time.time()
        for hint_id in hint_ids:
            point = self._get_point(hint_id)
            if point is None:
                continue
            payload = point.payload
            success_count = int(payload.get("success_count", 0))
            failure_count = int(payload.get("failure_count", 0))

            if correct:
                self._client.set_payload(
                    collection_name=self._collection,
                    payload={
                        "success_count": success_count + 1,
                        "last_validated_at": now,
                    },
                    points=[hint_id],
                )
            else:
                new_failure = failure_count + 1
                new_conf = _confidence(success_count, new_failure)
                update: dict = {"failure_count": new_failure}
                if new_conf < 0.25:
                    update["polarity"] = "negative"
                self._client.set_payload(
                    collection_name=self._collection,
                    payload=update,
                    points=[hint_id],
                )

    def update_content(self, hint_id: str, new_content: str) -> None:
        self._client.set_payload(
            collection_name=self._collection,
            payload={"content": new_content},
            points=[hint_id],
        )

    def delete(self, hint_id: str) -> None:
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[hint_id]),
        )

    def prune(self) -> int:
        from qdrant_client.models import PointIdsList

        now = time.time()
        age_180 = now - 180 * 86400
        age_90 = now - 90 * 86400

        all_records = self.all()
        to_delete: list[str] = []
        for rec in all_records:
            rule1 = rec.failure_count > 5 and rec.success_count == 0
            rule2 = rec.created_at < age_180 and rec.retrieval_count == 0
            rule3 = rec.polarity == "negative" and rec.created_at < age_90
            if rule1 or rule2 or rule3:
                to_delete.append(rec.hint_id)

        if to_delete:
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=to_delete),
            )
        return len(to_delete)

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()


# ---------------------------------------------------------------------------
# HintStore — public facade
# ---------------------------------------------------------------------------


class HintStore:
    """
    Persistent store for HintRecord objects.

    backend='sqlite' (default): portable, zero extra deps, file at db_path.
    backend='qdrant': disk-persistent local Qdrant (requires qdrant-client).
                      Uses qdrant_path for storage, collection_name for the
                      collection. Vector size fixed at 1536 (Cosine distance).
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        backend: Literal["sqlite", "qdrant"] = "sqlite",
        qdrant_path: str = "./qdrant_storage",
        collection_name: str = "hints",
    ) -> None:
        if backend == "qdrant":
            self._backend: _Backend = _QdrantBackend(
                qdrant_path=qdrant_path,
                collection_name=collection_name,
            )
        else:
            self._backend = _SQLiteBackend(db_path=db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: HintRecord) -> str:
        """Persist a HintRecord; assigns a UUID if hint_id is empty."""
        return self._backend.add(record)

    def get(self, hint_id: str) -> HintRecord | None:
        return self._backend.get(hint_id)

    def all(self) -> list[HintRecord]:
        return self._backend.all()

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        t_half_days: float = 30.0,
        theta_pos: float = 0.3,
        theta_neg: float = 0.6,
        scope_id: str | None = None,
    ) -> list[HintRecord]:
        """Return top-K hints scored by α·sim + β·conf + γ·recency.

        scope_id filtering: location hints are STRICT (same scope only);
        entity/strategy hints are LOOSE (cross-scope). None disables filtering.
        """
        return self._backend.search(
            query_embedding=query_embedding,
            k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            t_half_days=t_half_days,
            theta_pos=theta_pos,
            theta_neg=theta_neg,
            scope_id=scope_id,
        )

    def update_on_outcome(self, hint_ids: list[str], correct: bool) -> None:
        return self._backend.update_on_outcome(hint_ids, correct)

    def update_content(self, hint_id: str, new_content: str) -> None:
        self._backend.update_content(hint_id, new_content)

    def delete(self, hint_id: str) -> None:
        self._backend.delete(hint_id)

    def prune(self) -> int:
        """Remove hints matching DESIGN.md §4 rules. Returns count deleted."""
        return self._backend.prune()

    def close(self) -> None:
        self._backend.close()
