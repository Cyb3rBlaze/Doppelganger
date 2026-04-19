"""Unified memory graph schema and backfill helpers."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.services import message_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DOTENV_PATH = PROJECT_ROOT / ".env"
FALLBACK_DOTENV_PATH = PROJECT_ROOT.parent / "internal_documents_core" / ".env"
UNIFIED_MEMORY_DSN_ENV = "POSTGRES_DSN"
INTERNAL_DOCUMENTS_SOURCE_DSN_ENV = "INTERNAL_DOCUMENTS_POSTGRES_DSN"
EMBEDDING_MODEL_ENV = "INTERNAL_DOCUMENTS_EMBEDDING_MODEL"
EMBEDDING_DIMENSION_ENV = "INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_MESSAGE_DOCUMENT_EDGE_LIMIT = 5
DEFAULT_MESSAGE_DOCUMENT_SIMILARITY_THRESHOLD = 0.45

load_dotenv(LOCAL_DOTENV_PATH)
if FALLBACK_DOTENV_PATH.exists():
    load_dotenv(FALLBACK_DOTENV_PATH, override=False)

ENABLE_PGVECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

CREATE_MEMORY_NODES_TABLE_SQL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS memory_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    user_id TEXT NULL,
    conversation_id TEXT NULL,
    session_date DATE NULL,
    source_system TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding VECTOR({embedding_dimension}) NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_system, source_id)
)
"""

CREATE_MEMORY_EDGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_edges (
    id BIGSERIAL PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    edge_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    signals JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_node_id, target_node_id)
)
"""

CREATE_MEMORY_NODE_TYPE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS memory_nodes_node_type_idx
ON memory_nodes (node_type)
"""

CREATE_MEMORY_NODE_SESSION_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS memory_nodes_session_idx
ON memory_nodes (user_id, conversation_id, session_date)
"""

CREATE_MEMORY_EDGE_SOURCE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS memory_edges_source_idx
ON memory_edges (source_node_id)
"""

CREATE_MEMORY_EDGE_TARGET_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS memory_edges_target_idx
ON memory_edges (target_node_id)
"""

UPSERT_MEMORY_NODE_SQL = """
INSERT INTO memory_nodes (
    node_id,
    node_type,
    user_id,
    conversation_id,
    session_date,
    source_system,
    source_id,
    title,
    content,
    metadata,
    embedding
) VALUES (
    %(node_id)s,
    %(node_type)s,
    %(user_id)s,
    %(conversation_id)s,
    %(session_date)s,
    %(source_system)s,
    %(source_id)s,
    %(title)s,
    %(content)s,
    %(metadata)s::jsonb,
    %(embedding)s::vector
)
ON CONFLICT (node_id) DO UPDATE SET
    node_type = EXCLUDED.node_type,
    user_id = EXCLUDED.user_id,
    conversation_id = EXCLUDED.conversation_id,
    session_date = EXCLUDED.session_date,
    source_system = EXCLUDED.source_system,
    source_id = EXCLUDED.source_id,
    title = EXCLUDED.title,
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding,
    updated_at = NOW()
"""

UPSERT_MEMORY_EDGE_SQL = """
INSERT INTO memory_edges (
    source_node_id,
    target_node_id,
    score,
    edge_types,
    signals
) VALUES (
    %(source_node_id)s,
    %(target_node_id)s,
    %(score)s,
    %(edge_types)s::jsonb,
    %(signals)s::jsonb
)
ON CONFLICT (source_node_id, target_node_id) DO UPDATE SET
    score = EXCLUDED.score,
    edge_types = EXCLUDED.edge_types,
    signals = EXCLUDED.signals,
    updated_at = NOW()
"""

SELECT_MESSAGE_SESSIONS_FOR_BACKFILL_SQL = """
SELECT
    session_id,
    session_date,
    channel,
    user_id,
    conversation_id,
    session_summary,
    message_history
FROM message_sessions
ORDER BY session_date ASC, session_id ASC
"""

SELECT_DOCUMENT_CHUNKS_FOR_BACKFILL_SQL = """
SELECT
    document_id,
    chunk_id,
    source_path,
    source_kind,
    title,
    content,
    metadata,
    connected_nodes,
    chunk_index,
    window_start_chunk_index,
    window_end_chunk_index,
    embedding
FROM document_chunks
ORDER BY document_id ASC, chunk_index ASC
"""

DELETE_MESSAGE_DOCUMENT_EDGES_SQL = """
DELETE FROM memory_edges
WHERE edge_types @> '["message_document"]'::jsonb
"""

SELECT_MESSAGE_DOCUMENT_SEMANTIC_EDGE_CANDIDATES_SQL = """
SELECT
    message.node_id AS message_node_id,
    document.node_id AS document_node_id,
    document.score AS semantic_score
FROM memory_nodes AS message
JOIN LATERAL (
    SELECT
        candidate.node_id,
        1 - (candidate.embedding <=> message.embedding) AS score
    FROM memory_nodes AS candidate
    WHERE candidate.node_type = 'document_chunk'
      AND candidate.embedding IS NOT NULL
    ORDER BY candidate.embedding <=> message.embedding
    LIMIT %(limit)s
) AS document ON TRUE
WHERE message.node_type = 'message'
  AND message.embedding IS NOT NULL
  AND document.score >= %(threshold)s
ORDER BY message.node_id ASC, document.score DESC, document.node_id ASC
"""


@dataclass(frozen=True)
class MemoryNodeRecord:
    """One node in the unified memory graph."""

    node_id: str
    node_type: str
    user_id: str | None
    conversation_id: str | None
    session_date: object | None
    source_system: str
    source_id: str
    title: str | None
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None


@dataclass(frozen=True)
class MemoryEdgeRecord:
    """One edge in the unified memory graph."""

    source_node_id: str
    target_node_id: str
    score: float
    edge_types: list[str]
    signals: dict[str, float]


def _load_psycopg() -> Any:
    """Import psycopg lazily."""
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Postgres dependencies are not installed. Run `pip install -e .` first."
        ) from exc
    return psycopg


def _load_openai_sdk() -> Any:
    """Import the OpenAI SDK lazily."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `pip install -e .` first."
        ) from exc
    return OpenAI


def get_unified_memory_dsn() -> str | None:
    """Return the target doppelganger Postgres DSN for unified memory."""
    return os.getenv(UNIFIED_MEMORY_DSN_ENV)


def get_internal_documents_source_dsn() -> str | None:
    """Return the source DSN for the legacy document chunk store."""
    return os.getenv(INTERNAL_DOCUMENTS_SOURCE_DSN_ENV)


def get_embedding_model() -> str:
    """Return the embedding model to use for unified-memory nodes."""
    return os.getenv(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)


def get_embedding_dimension() -> int:
    """Return the embedding dimension to use for the unified memory schema."""
    raw_value = os.getenv(EMBEDDING_DIMENSION_ENV)
    if not raw_value:
        return DEFAULT_EMBEDDING_DIMENSION
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{EMBEDDING_DIMENSION_ENV} must be an integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{EMBEDDING_DIMENSION_ENV} must be greater than zero.")
    return value


def build_create_memory_nodes_table_sql(embedding_dimension: int) -> str:
    """Render the unified memory node DDL with a validated vector dimension."""
    if embedding_dimension <= 0:
        raise RuntimeError("Embedding dimension must be greater than zero.")
    return CREATE_MEMORY_NODES_TABLE_SQL_TEMPLATE.format(
        embedding_dimension=embedding_dimension
    )


def build_openai_client() -> Any:
    """Build an OpenAI client from environment variables."""
    OpenAI = _load_openai_sdk()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_memory_texts(
    texts: Sequence[str],
    *,
    client: Any | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Create embeddings for unified-memory node content in stable input order."""
    if not texts:
        return []

    resolved_client = client or build_openai_client()
    resolved_model = model or get_embedding_model()
    resolved_dimensions = dimensions or get_embedding_dimension()
    results: list[list[float]] = []

    for start_index in range(0, len(texts), batch_size):
        batch = list(texts[start_index : start_index + batch_size])
        response = resolved_client.embeddings.create(
            model=resolved_model,
            input=batch,
            dimensions=resolved_dimensions,
        )
        results.extend(list(item.embedding) for item in response.data)

    return results


@lru_cache(maxsize=None)
def ensure_unified_memory_schema(dsn: str, embedding_dimension: int) -> None:
    """Create the unified memory graph schema once per process."""
    psycopg = _load_psycopg()
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(ENABLE_PGVECTOR_EXTENSION_SQL)
            cursor.execute(build_create_memory_nodes_table_sql(embedding_dimension))
            cursor.execute(CREATE_MEMORY_EDGES_TABLE_SQL)
            cursor.execute(CREATE_MEMORY_NODE_TYPE_INDEX_SQL)
            cursor.execute(CREATE_MEMORY_NODE_SESSION_INDEX_SQL)
            cursor.execute(CREATE_MEMORY_EDGE_SOURCE_INDEX_SQL)
            cursor.execute(CREATE_MEMORY_EDGE_TARGET_INDEX_SQL)
        connection.commit()


def format_embedding_literal(embedding: list[float]) -> str:
    """Format an embedding as a pgvector literal."""
    return "[" + ",".join(str(value) for value in embedding) + "]"


def _normalize_json_list(value: Any) -> list[Any]:
    """Normalize a JSON-like list payload."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return list(value)


def _normalize_json_dict(value: Any) -> dict[str, Any]:
    """Normalize a JSON-like dict payload."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return dict(value)


def _normalize_embedding(value: Any) -> list[float] | None:
    """Normalize an embedding payload from Postgres."""
    if value is None:
        return None
    if isinstance(value, list):
        numbers = [float(item) for item in value]
        return numbers if numbers else None
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        if trimmed.startswith("[") and trimmed.endswith("]"):
            parts = [part.strip() for part in trimmed[1:-1].split(",") if part.strip()]
            return [float(part) for part in parts] if parts else None
    return [float(item) for item in value]


def _message_node_id(session_id: str, event_index: int) -> str:
    return f"message:{session_id}:{event_index}"


def _summary_node_id(session_id: str) -> str:
    return f"session_summary:{session_id}"


def _document_chunk_node_id(chunk_id: str) -> str:
    return f"document_chunk:{chunk_id}"


def build_message_embedding_text(
    *,
    channel: str,
    conversation_id: str | None,
    user_id: str | None,
    session_id: str,
    event_index: int,
    event: dict[str, Any],
) -> str:
    """Build stable embedding text for a message node."""
    message_text = str(event.get("text") or "").strip()
    direction = str(event.get("direction") or "unknown")
    message_id = str(event.get("message_id") or "")
    created_at = str(event.get("created_at") or "")

    lines = [
        "Node type: message",
        f"Channel: {channel}",
        f"Conversation ID: {conversation_id or ''}",
        f"User ID: {user_id or ''}",
        f"Session ID: {session_id}",
        f"Event index: {event_index}",
        f"Direction: {direction}",
    ]
    if message_id:
        lines.append(f"Message ID: {message_id}")
    if created_at:
        lines.append(f"Created at: {created_at}")
    lines.extend(
        [
            "",
            "Message content:",
            message_text or "(empty message)",
        ]
    )
    return "\n".join(lines)


def build_session_summary_embedding_text(
    *,
    channel: str,
    conversation_id: str | None,
    user_id: str | None,
    session_id: str,
    session_summary: str,
) -> str:
    """Build stable embedding text for a session-summary node."""
    lines = [
        "Node type: session_summary",
        f"Channel: {channel}",
        f"Conversation ID: {conversation_id or ''}",
        f"User ID: {user_id or ''}",
        f"Session ID: {session_id}",
        "",
        "Session summary:",
        session_summary.strip() or "(empty summary)",
    ]
    return "\n".join(lines)


def _upsert_memory_node(cursor: Any, node: MemoryNodeRecord) -> None:
    cursor.execute(
        UPSERT_MEMORY_NODE_SQL,
        {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "user_id": node.user_id,
            "conversation_id": node.conversation_id,
            "session_date": node.session_date,
            "source_system": node.source_system,
            "source_id": node.source_id,
            "title": node.title,
            "content": node.content,
            "metadata": json.dumps(node.metadata, default=str),
            "embedding": None if node.embedding is None else format_embedding_literal(node.embedding),
        },
    )


def _upsert_memory_edge(cursor: Any, edge: MemoryEdgeRecord) -> None:
    cursor.execute(
        UPSERT_MEMORY_EDGE_SQL,
        {
            "source_node_id": edge.source_node_id,
            "target_node_id": edge.target_node_id,
            "score": edge.score,
            "edge_types": json.dumps(edge.edge_types, default=str),
            "signals": json.dumps(edge.signals, default=str),
        },
    )


def backfill_message_document_semantic_edges(
    *,
    target_dsn: str | None = None,
    similarity_threshold: float = DEFAULT_MESSAGE_DOCUMENT_SIMILARITY_THRESHOLD,
    limit_per_message: int = DEFAULT_MESSAGE_DOCUMENT_EDGE_LIMIT,
) -> dict[str, int]:
    """Backfill semantic cross-type edges between message nodes and document chunks."""
    resolved_dsn = target_dsn or get_unified_memory_dsn()
    if not resolved_dsn:
        raise RuntimeError(f"{UNIFIED_MEMORY_DSN_ENV} is not set.")
    if limit_per_message <= 0:
        raise RuntimeError("limit_per_message must be greater than zero.")

    ensure_unified_memory_schema(resolved_dsn, get_embedding_dimension())
    psycopg = _load_psycopg()
    with psycopg.connect(resolved_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(DELETE_MESSAGE_DOCUMENT_EDGES_SQL)
            cursor.execute(
                SELECT_MESSAGE_DOCUMENT_SEMANTIC_EDGE_CANDIDATES_SQL,
                {
                    "limit": limit_per_message,
                    "threshold": similarity_threshold,
                },
            )
            rows = cursor.fetchall()

            edge_count = 0
            unique_message_ids: set[str] = set()
            unique_document_ids: set[str] = set()

            for message_node_id, document_node_id, semantic_score in rows:
                score = float(semantic_score)
                unique_message_ids.add(message_node_id)
                unique_document_ids.add(document_node_id)
                for source_node_id, target_node_id in (
                    (message_node_id, document_node_id),
                    (document_node_id, message_node_id),
                ):
                    _upsert_memory_edge(
                        cursor,
                        MemoryEdgeRecord(
                            source_node_id=source_node_id,
                            target_node_id=target_node_id,
                            score=score,
                            edge_types=["semantic", "message_document"],
                            signals={"semantic": score, "message_document": 1.0},
                        ),
                    )
                    edge_count += 1

        connection.commit()

    return {
        "message_count": len(unique_message_ids),
        "document_chunk_count": len(unique_document_ids),
        "edge_count": edge_count,
    }


def backfill_message_sessions_to_unified_memory(
    *,
    target_dsn: str | None = None,
) -> dict[str, int]:
    """Backfill message session events and summaries into unified memory nodes and edges."""
    resolved_dsn = target_dsn or get_unified_memory_dsn()
    if not resolved_dsn:
        raise RuntimeError(f"{UNIFIED_MEMORY_DSN_ENV} is not set.")

    ensure_unified_memory_schema(resolved_dsn, get_embedding_dimension())
    psycopg = _load_psycopg()
    with psycopg.connect(resolved_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_MESSAGE_SESSIONS_FOR_BACKFILL_SQL)
            rows = cursor.fetchall()

            pending_nodes: list[tuple[MemoryNodeRecord, str]] = []
            pending_edges: list[MemoryEdgeRecord] = []

            for row in rows:
                session_id, session_date, channel, user_id, conversation_id, session_summary, raw_history = row
                message_events = _normalize_json_list(raw_history)
                message_node_ids: list[str] = []

                for event_index, event in enumerate(message_events):
                    event_dict = _normalize_json_dict(event)
                    node_id = _message_node_id(session_id, event_index)
                    message_node_ids.append(node_id)
                    metadata = {
                        "channel": channel,
                        "session_id": session_id,
                        "event_index": event_index,
                        "direction": event_dict.get("direction"),
                        "message_id": event_dict.get("message_id"),
                        "created_at": event_dict.get("created_at"),
                        "message_metadata": _normalize_json_dict(event_dict.get("metadata")),
                    }
                    pending_nodes.append(
                        (
                            MemoryNodeRecord(
                                node_id=node_id,
                                node_type="message",
                                user_id=user_id,
                                conversation_id=conversation_id,
                                session_date=session_date,
                                source_system="message_sessions",
                                source_id=f"{session_id}:{event_index}",
                                title=f"{channel} {event_dict.get('direction') or 'message'}",
                                content=str(event_dict.get("text") or ""),
                                metadata=metadata,
                            ),
                            build_message_embedding_text(
                                channel=channel,
                                conversation_id=conversation_id,
                                user_id=user_id,
                                session_id=session_id,
                                event_index=event_index,
                                event=event_dict,
                            ),
                        )
                    )

                    if event_index > 0:
                        previous_node_id = message_node_ids[event_index - 1]
                        for source_node_id, target_node_id in (
                            (previous_node_id, node_id),
                            (node_id, previous_node_id),
                        ):
                            pending_edges.append(
                                MemoryEdgeRecord(
                                    source_node_id=source_node_id,
                                    target_node_id=target_node_id,
                                    score=1.0,
                                    edge_types=["same_session", "session_sequence"],
                                    signals={"same_session": 0.5, "session_sequence": 1.0},
                                )
                            )

                if session_summary:
                    summary_node_id = _summary_node_id(session_id)
                    pending_nodes.append(
                        (
                            MemoryNodeRecord(
                                node_id=summary_node_id,
                                node_type="session_summary",
                                user_id=user_id,
                                conversation_id=conversation_id,
                                session_date=session_date,
                                source_system="message_sessions",
                                source_id=f"{session_id}:summary",
                                title=f"{channel} session summary",
                                content=session_summary,
                                metadata={"channel": channel, "session_id": session_id},
                            ),
                            build_session_summary_embedding_text(
                                channel=channel,
                                conversation_id=conversation_id,
                                user_id=user_id,
                                session_id=session_id,
                                session_summary=session_summary,
                            ),
                        )
                    )

                    for message_node_id in message_node_ids:
                        for source_node_id, target_node_id in (
                            (summary_node_id, message_node_id),
                            (message_node_id, summary_node_id),
                        ):
                            pending_edges.append(
                                MemoryEdgeRecord(
                                    source_node_id=source_node_id,
                                    target_node_id=target_node_id,
                                    score=1.0,
                                    edge_types=["summarizes_session"],
                                    signals={"summarizes_session": 1.0},
                                )
                            )

            embeddings = embed_memory_texts([text for _, text in pending_nodes])

            node_count = 0
            for (node, _), embedding in zip(pending_nodes, embeddings, strict=True):
                _upsert_memory_node(
                    cursor,
                    MemoryNodeRecord(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        user_id=node.user_id,
                        conversation_id=node.conversation_id,
                        session_date=node.session_date,
                        source_system=node.source_system,
                        source_id=node.source_id,
                        title=node.title,
                        content=node.content,
                        metadata=node.metadata,
                        embedding=embedding,
                    ),
                )
                node_count += 1

            edge_count = 0
            for edge in pending_edges:
                _upsert_memory_edge(cursor, edge)
                edge_count += 1

        connection.commit()

    return {
        "session_count": len(rows),
        "node_count": node_count,
        "edge_count": edge_count,
    }


def backfill_document_chunks_to_unified_memory(
    *,
    source_dsn: str | None = None,
    target_dsn: str | None = None,
) -> dict[str, int]:
    """Backfill legacy internal document chunks into unified memory nodes and edges."""
    resolved_source_dsn = source_dsn or get_internal_documents_source_dsn()
    resolved_target_dsn = target_dsn or get_unified_memory_dsn()
    if not resolved_source_dsn:
        raise RuntimeError(f"{INTERNAL_DOCUMENTS_SOURCE_DSN_ENV} is not set.")
    if not resolved_target_dsn:
        raise RuntimeError(f"{UNIFIED_MEMORY_DSN_ENV} is not set.")

    ensure_unified_memory_schema(resolved_target_dsn, get_embedding_dimension())
    psycopg = _load_psycopg()

    with psycopg.connect(resolved_source_dsn) as source_connection:
        with source_connection.cursor() as source_cursor:
            source_cursor.execute(SELECT_DOCUMENT_CHUNKS_FOR_BACKFILL_SQL)
            rows = source_cursor.fetchall()

    with psycopg.connect(resolved_target_dsn) as target_connection:
        with target_connection.cursor() as target_cursor:
            node_count = 0
            edge_count = 0
            for row in rows:
                (
                    document_id,
                    chunk_id,
                    source_path,
                    source_kind,
                    title,
                    content,
                    metadata,
                    connected_nodes,
                    chunk_index,
                    window_start_chunk_index,
                    window_end_chunk_index,
                    embedding,
                ) = row

                _upsert_memory_node(
                    target_cursor,
                    MemoryNodeRecord(
                        node_id=_document_chunk_node_id(chunk_id),
                        node_type="document_chunk",
                        user_id=None,
                        conversation_id=None,
                        session_date=None,
                        source_system="document_chunks",
                        source_id=chunk_id,
                        title=title,
                        content=content,
                        metadata={
                            **_normalize_json_dict(metadata),
                            "document_id": document_id,
                            "source_path": source_path,
                            "source_kind": source_kind,
                            "chunk_index": chunk_index,
                            "window_start_chunk_index": window_start_chunk_index,
                            "window_end_chunk_index": window_end_chunk_index,
                        },
                        embedding=_normalize_embedding(embedding),
                    ),
                )
                node_count += 1

                for connected_node in _normalize_json_list(connected_nodes):
                    connected_node_dict = _normalize_json_dict(connected_node)
                    target_chunk_id = connected_node_dict.get("chunk_id")
                    if not isinstance(target_chunk_id, str) or not target_chunk_id:
                        continue
                    _upsert_memory_edge(
                        target_cursor,
                        MemoryEdgeRecord(
                            source_node_id=_document_chunk_node_id(chunk_id),
                            target_node_id=_document_chunk_node_id(target_chunk_id),
                            score=float(connected_node_dict.get("score") or 0.0),
                            edge_types=list(connected_node_dict.get("edge_types") or []),
                            signals={
                                key: float(value)
                                for key, value in _normalize_json_dict(
                                    connected_node_dict.get("signals")
                                ).items()
                            },
                        ),
                    )
                    edge_count += 1

        target_connection.commit()

    return {
        "chunk_count": len(rows),
        "node_count": node_count,
        "edge_count": edge_count,
    }


def backfill_all_to_unified_memory() -> dict[str, Any]:
    """Create unified memory schema and backfill both sessions and document chunks."""
    target_dsn = get_unified_memory_dsn()
    if not target_dsn:
        raise RuntimeError(f"{UNIFIED_MEMORY_DSN_ENV} is not set.")

    ensure_unified_memory_schema(target_dsn, get_embedding_dimension())
    message_result = backfill_message_sessions_to_unified_memory(target_dsn=target_dsn)
    document_result = backfill_document_chunks_to_unified_memory(target_dsn=target_dsn)
    cross_type_result = backfill_message_document_semantic_edges(target_dsn=target_dsn)
    return {
        "status": "ok",
        "target_dsn_env": UNIFIED_MEMORY_DSN_ENV,
        "document_source_dsn_env": INTERNAL_DOCUMENTS_SOURCE_DSN_ENV,
        "message_sessions": message_result,
        "document_chunks": document_result,
        "message_document_edges": cross_type_result,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the unified memory migration CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create unified memory graph tables in the doppelganger database and "
            "backfill message_sessions plus legacy document_chunks into them."
        )
    )
    parser.add_argument(
        "--messages-only",
        action="store_true",
        help="Backfill only message_sessions into unified memory.",
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="Backfill only legacy document_chunks into unified memory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the unified memory migration CLI."""
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.messages_only and args.documents_only:
        raise SystemExit("Choose only one of --messages-only or --documents-only.")

    target_dsn = get_unified_memory_dsn()
    if not target_dsn:
        raise RuntimeError(f"{UNIFIED_MEMORY_DSN_ENV} is not set.")

    ensure_unified_memory_schema(target_dsn, get_embedding_dimension())

    if args.messages_only:
        result = {
            "status": "ok",
            "target_dsn_env": UNIFIED_MEMORY_DSN_ENV,
            "message_sessions": backfill_message_sessions_to_unified_memory(target_dsn=target_dsn),
            "message_document_edges": backfill_message_document_semantic_edges(
                target_dsn=target_dsn
            ),
        }
    elif args.documents_only:
        result = {
            "status": "ok",
            "target_dsn_env": UNIFIED_MEMORY_DSN_ENV,
            "document_source_dsn_env": INTERNAL_DOCUMENTS_SOURCE_DSN_ENV,
            "document_chunks": backfill_document_chunks_to_unified_memory(target_dsn=target_dsn),
            "message_document_edges": backfill_message_document_semantic_edges(
                target_dsn=target_dsn
            ),
        }
    else:
        result = backfill_all_to_unified_memory()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
