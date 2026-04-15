"""Postgres/pgvector storage and search helpers for internal documents."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import os
from urllib.parse import unquote, urlsplit, urlunsplit

from dotenv import load_dotenv

from core.document_sources import InternalDocument
from core.embeddings import get_embedding_dimension as get_default_embedding_dimension

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DOTENV_PATH = PROJECT_ROOT / ".env"
FALLBACK_DOTENV_PATH = PROJECT_ROOT.parent / "doppelganger_core" / ".env"
POSTGRES_DSN_ENV = "INTERNAL_DOCUMENTS_POSTGRES_DSN"
EMBEDDING_DIMENSION_ENV = "INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION"

load_dotenv(LOCAL_DOTENV_PATH)
if FALLBACK_DOTENV_PATH.exists():
    load_dotenv(FALLBACK_DOTENV_PATH, override=False)

ENABLE_PGVECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

CREATE_DOCUMENTS_TABLE_SQL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    title TEXT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    chunk_index INTEGER NOT NULL,
    window_start_chunk_index INTEGER NOT NULL,
    window_end_chunk_index INTEGER NOT NULL,
    embedding VECTOR({embedding_dimension}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (chunk_id)
)
"""

CREATE_DOCUMENTS_LOOKUP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
ON document_chunks (document_id)
"""

CREATE_CHUNKS_ORDER_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS document_chunks_chunk_index_idx
ON document_chunks (document_id, chunk_index)
"""

DELETE_DOCUMENT_CHUNKS_SQL = """
DELETE FROM document_chunks
WHERE document_id = %(document_id)s
"""

UPSERT_DOCUMENT_SQL = """
INSERT INTO document_chunks (
    document_id,
    chunk_id,
    source_path,
    source_kind,
    title,
    content,
    metadata,
    chunk_index,
    window_start_chunk_index,
    window_end_chunk_index,
    embedding
) VALUES (
    %(document_id)s,
    %(chunk_id)s,
    %(source_path)s,
    %(source_kind)s,
    %(title)s,
    %(content)s,
    %(metadata)s::jsonb,
    %(chunk_index)s,
    %(window_start_chunk_index)s,
    %(window_end_chunk_index)s,
    %(embedding)s::vector
)
ON CONFLICT (chunk_id) DO UPDATE SET
    source_path = EXCLUDED.source_path,
    source_kind = EXCLUDED.source_kind,
    title = EXCLUDED.title,
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    chunk_index = EXCLUDED.chunk_index,
    window_start_chunk_index = EXCLUDED.window_start_chunk_index,
    window_end_chunk_index = EXCLUDED.window_end_chunk_index,
    embedding = EXCLUDED.embedding,
    updated_at = NOW()
"""

SEARCH_DOCUMENTS_SQL = """
SELECT
    document_id,
    chunk_id,
    source_path,
    source_kind,
    title,
    content,
    metadata,
    chunk_index,
    window_start_chunk_index,
    window_end_chunk_index,
    1 - (embedding <=> %(embedding)s::vector) AS score
FROM document_chunks
ORDER BY embedding <=> %(embedding)s::vector
LIMIT %(limit)s
"""


@dataclass(frozen=True)
class VectorStoreConfig:
    """Resolved configuration for the pgvector-backed document store."""

    postgres_dsn: str
    embedding_dimension: int


@dataclass(frozen=True)
class DocumentChunkRecord:
    """One stored chunk/window row for the internal document vector store."""

    document_id: str
    chunk_id: str
    source_path: str
    source_kind: str
    title: str
    content: str
    metadata: dict[str, Any]
    chunk_index: int
    window_start_chunk_index: int
    window_end_chunk_index: int


@dataclass(frozen=True)
class EmbeddedChunkRecord:
    """One stored chunk/window row plus its embedding."""

    record: DocumentChunkRecord
    embedding: list[float]


def _load_psycopg() -> Any:
    """Import psycopg lazily."""
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed for internal_documents_core. "
            "Run `python -m pip install -e .` inside internal_documents_core first."
        ) from exc
    return psycopg


def get_postgres_dsn() -> str | None:
    """Return the configured Postgres DSN for the internal documents store."""
    return os.getenv(POSTGRES_DSN_ENV)


def get_embedding_dimension() -> int:
    """Return the configured embedding dimension for pgvector storage."""
    raw_value = os.getenv(EMBEDDING_DIMENSION_ENV)
    if not raw_value:
        return get_default_embedding_dimension()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{EMBEDDING_DIMENSION_ENV} must be an integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{EMBEDDING_DIMENSION_ENV} must be greater than zero.")
    return value


def get_vector_store_config() -> VectorStoreConfig:
    """Return validated pgvector storage configuration."""
    postgres_dsn = get_postgres_dsn()
    if not postgres_dsn:
        raise RuntimeError(f"{POSTGRES_DSN_ENV} is not set.")
    return VectorStoreConfig(
        postgres_dsn=postgres_dsn,
        embedding_dimension=get_embedding_dimension(),
    )


def get_database_name_from_dsn(postgres_dsn: str) -> str:
    """Extract the target database name from a PostgreSQL URI DSN."""
    parsed = urlsplit(postgres_dsn)
    database_name = unquote(parsed.path.lstrip("/"))
    if not database_name:
        raise RuntimeError(f"Could not determine database name from DSN: {postgres_dsn!r}")
    return database_name


def build_maintenance_dsn(postgres_dsn: str, *, database_name: str = "postgres") -> str:
    """Return a DSN pointed at a maintenance database on the same server."""
    parsed = urlsplit(postgres_dsn)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, parsed.fragment))


def _quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier defensively for simple CREATE DATABASE usage."""
    return '"' + identifier.replace('"', '""') + '"'


def ensure_database_exists(*, config: VectorStoreConfig | None = None) -> bool:
    """Create the configured database if it does not exist yet."""
    resolved_config = config or get_vector_store_config()
    psycopg = _load_psycopg()
    operational_error = getattr(psycopg, "OperationalError", Exception)

    try:
        with psycopg.connect(resolved_config.postgres_dsn):
            return False
    except operational_error as exc:
        if "does not exist" not in str(exc):
            raise

    database_name = get_database_name_from_dsn(resolved_config.postgres_dsn)
    maintenance_dsn = build_maintenance_dsn(resolved_config.postgres_dsn)
    create_database_sql = f"CREATE DATABASE {_quote_identifier(database_name)}"

    with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(create_database_sql)
    return True


def build_create_documents_table_sql(embedding_dimension: int) -> str:
    """Render the chunk table DDL with a validated vector dimension."""
    if embedding_dimension <= 0:
        raise RuntimeError("Embedding dimension must be greater than zero.")
    return CREATE_DOCUMENTS_TABLE_SQL_TEMPLATE.format(embedding_dimension=embedding_dimension)


def ensure_pgvector_schema(*, config: VectorStoreConfig | None = None) -> bool:
    """Create the pgvector extension and base document chunk table."""
    resolved_config = config or get_vector_store_config()
    ensure_database_exists(config=resolved_config)
    psycopg = _load_psycopg()
    with psycopg.connect(resolved_config.postgres_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(ENABLE_PGVECTOR_EXTENSION_SQL)
            cursor.execute(build_create_documents_table_sql(resolved_config.embedding_dimension))
            cursor.execute(CREATE_DOCUMENTS_LOOKUP_INDEX_SQL)
            cursor.execute(CREATE_CHUNKS_ORDER_INDEX_SQL)
        connection.commit()
    return True


def format_embedding_literal(embedding: list[float]) -> str:
    """Format an embedding as a pgvector literal."""
    return "[" + ",".join(str(value) for value in embedding) + "]"


def build_document_chunk_record(
    document: InternalDocument,
    *,
    chunk_id: str,
    chunk_index: int,
    window_start_chunk_index: int,
    window_end_chunk_index: int,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentChunkRecord:
    """Build one chunk/window record from a normalized source document."""
    return DocumentChunkRecord(
        document_id=document.document_id,
        chunk_id=chunk_id,
        source_path=document.source_path,
        source_kind=document.source_kind,
        title=document.title,
        content=document.content if content is None else content,
        metadata=document.metadata if metadata is None else metadata,
        chunk_index=chunk_index,
        window_start_chunk_index=window_start_chunk_index,
        window_end_chunk_index=window_end_chunk_index,
    )


def build_default_document_chunk(document: InternalDocument) -> DocumentChunkRecord:
    """Build the temporary one-chunk wrapper used before adaptive chunking lands."""
    return build_document_chunk_record(
        document,
        chunk_id=f"{document.document_id}:chunk:0",
        chunk_index=0,
        window_start_chunk_index=0,
        window_end_chunk_index=0,
    )


def upsert_document_chunk(
    chunk: DocumentChunkRecord,
    embedding: list[float],
    *,
    config: VectorStoreConfig | None = None,
) -> bool:
    """Insert or update one embedded document chunk in Postgres."""
    resolved_config = config or get_vector_store_config()
    psycopg = _load_psycopg()
    with psycopg.connect(resolved_config.postgres_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                UPSERT_DOCUMENT_SQL,
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "source_kind": chunk.source_kind,
                    "title": chunk.title,
                    "content": chunk.content,
                    "metadata": json.dumps(chunk.metadata, default=str),
                    "chunk_index": chunk.chunk_index,
                    "window_start_chunk_index": chunk.window_start_chunk_index,
                    "window_end_chunk_index": chunk.window_end_chunk_index,
                    "embedding": format_embedding_literal(embedding),
                },
            )
        connection.commit()
    return True


def replace_document_chunks(
    document_id: str,
    chunk_records: list[EmbeddedChunkRecord],
    *,
    config: VectorStoreConfig | None = None,
) -> bool:
    """Replace all stored chunks for one document in a single transaction."""
    resolved_config = config or get_vector_store_config()
    psycopg = _load_psycopg()
    with psycopg.connect(resolved_config.postgres_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(DELETE_DOCUMENT_CHUNKS_SQL, {"document_id": document_id})
            for chunk_record in chunk_records:
                chunk = chunk_record.record
                cursor.execute(
                    UPSERT_DOCUMENT_SQL,
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "source_path": chunk.source_path,
                        "source_kind": chunk.source_kind,
                        "title": chunk.title,
                        "content": chunk.content,
                        "metadata": json.dumps(chunk.metadata, default=str),
                        "chunk_index": chunk.chunk_index,
                        "window_start_chunk_index": chunk.window_start_chunk_index,
                        "window_end_chunk_index": chunk.window_end_chunk_index,
                        "embedding": format_embedding_literal(chunk_record.embedding),
                    },
                )
        connection.commit()
    return True


def upsert_document(
    document: InternalDocument,
    embedding: list[float],
    *,
    config: VectorStoreConfig | None = None,
) -> bool:
    """Compatibility wrapper that stores a whole document as chunk 0 for now."""
    return upsert_document_chunk(build_default_document_chunk(document), embedding, config=config)


def search_documents(
    query_embedding: list[float],
    *,
    limit: int = 5,
    config: VectorStoreConfig | None = None,
) -> list[dict[str, Any]]:
    """Run a simple pgvector similarity search over stored document chunks."""
    resolved_config = config or get_vector_store_config()
    psycopg = _load_psycopg()
    with psycopg.connect(resolved_config.postgres_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                SEARCH_DOCUMENTS_SQL,
                {
                    "embedding": format_embedding_literal(query_embedding),
                    "limit": limit,
                },
            )
            rows = cursor.fetchall()
    return [
        {
            "document_id": row[0],
            "chunk_id": row[1],
            "source_path": row[2],
            "source_kind": row[3],
            "title": row[4],
            "content": row[5],
            "metadata": row[6],
            "chunk_index": row[7],
            "window_start_chunk_index": row[8],
            "window_end_chunk_index": row[9],
            "score": row[10],
        }
        for row in rows
    ]
