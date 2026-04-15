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
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    title TEXT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding VECTOR({embedding_dimension}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id)
)
"""

CREATE_DOCUMENTS_LOOKUP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS documents_document_id_idx
ON documents (document_id)
"""

UPSERT_DOCUMENT_SQL = """
INSERT INTO documents (
    document_id,
    source_path,
    source_kind,
    title,
    content,
    metadata,
    embedding
) VALUES (
    %(document_id)s,
    %(source_path)s,
    %(source_kind)s,
    %(title)s,
    %(content)s,
    %(metadata)s::jsonb,
    %(embedding)s::vector
)
ON CONFLICT (document_id) DO UPDATE SET
    source_path = EXCLUDED.source_path,
    source_kind = EXCLUDED.source_kind,
    title = EXCLUDED.title,
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding,
    updated_at = NOW()
"""

SEARCH_DOCUMENTS_SQL = """
SELECT
    document_id,
    source_path,
    source_kind,
    title,
    metadata,
    1 - (embedding <=> %(embedding)s::vector) AS score
FROM documents
ORDER BY embedding <=> %(embedding)s::vector
LIMIT %(limit)s
"""


@dataclass(frozen=True)
class VectorStoreConfig:
    """Resolved configuration for the pgvector-backed document store."""

    postgres_dsn: str
    embedding_dimension: int


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
    """Render the documents table DDL with a validated vector dimension."""
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
        connection.commit()
    return True


def format_embedding_literal(embedding: list[float]) -> str:
    """Format an embedding as a pgvector literal."""
    return "[" + ",".join(str(value) for value in embedding) + "]"


def upsert_document(
    document: InternalDocument,
    embedding: list[float],
    *,
    config: VectorStoreConfig | None = None,
) -> bool:
    """Insert or update one embedded document in Postgres."""
    resolved_config = config or get_vector_store_config()
    psycopg = _load_psycopg()
    with psycopg.connect(resolved_config.postgres_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                UPSERT_DOCUMENT_SQL,
                {
                    "document_id": document.document_id,
                    "source_path": document.source_path,
                    "source_kind": document.source_kind,
                    "title": document.title,
                    "content": document.content,
                    "metadata": json.dumps(document.metadata, default=str),
                    "embedding": format_embedding_literal(embedding),
                },
            )
        connection.commit()
    return True


def search_documents(
    query_embedding: list[float],
    *,
    limit: int = 5,
    config: VectorStoreConfig | None = None,
) -> list[dict[str, Any]]:
    """Run a simple pgvector similarity search over stored documents."""
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
            "source_path": row[1],
            "source_kind": row[2],
            "title": row[3],
            "metadata": row[4],
            "score": row[5],
        }
        for row in rows
    ]
