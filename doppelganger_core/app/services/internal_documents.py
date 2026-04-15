"""Retrieval helpers for pgvector-backed internal documents."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.core.models import Message

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DOTENV_PATH = PROJECT_ROOT / ".env"
FALLBACK_DOTENV_PATH = PROJECT_ROOT.parent / "internal_documents_core" / ".env"
POSTGRES_DSN_ENV = "INTERNAL_DOCUMENTS_POSTGRES_DSN"
EMBEDDING_MODEL_ENV = "INTERNAL_DOCUMENTS_EMBEDDING_MODEL"
EMBEDDING_DIMENSION_ENV = "INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536
DEFAULT_RETRIEVAL_LIMIT = 3
DEFAULT_DOCUMENT_CONTEXT_CHAR_LIMIT = 1600
KNOWLEDGE_SEEKING_PATTERNS = (
    r"\?$",
    r"\b(what|why|how|when|where|who|which)\b",
    r"\b(remember|recall|find|search|look up|lookup)\b",
    r"\b(notes?|documents?|docs?|wrote|written|said before|mentioned before)\b",
    r"\b(summarize|summary|context|background|history)\b",
)

load_dotenv(LOCAL_DOTENV_PATH)
if FALLBACK_DOTENV_PATH.exists():
    load_dotenv(FALLBACK_DOTENV_PATH, override=False)


def _load_openai_sdk() -> Any:
    """Import the OpenAI SDK lazily."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `pip install -e .` first."
        ) from exc
    return OpenAI


def _load_psycopg() -> Any:
    """Import psycopg lazily."""
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed. Run `pip install -e .` first."
        ) from exc
    return psycopg


def get_internal_documents_dsn() -> str | None:
    """Return the configured Postgres DSN for internal documents retrieval."""
    return os.getenv(POSTGRES_DSN_ENV)


def looks_like_knowledge_seeking_query(message: Message) -> bool:
    """Return whether a message looks like it wants lookup-style background context."""
    text = message.text.strip().lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in KNOWLEDGE_SEEKING_PATTERNS)


def get_internal_documents_embedding_model() -> str:
    """Return the embedding model used for internal documents retrieval."""
    return os.getenv(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)


def get_internal_documents_embedding_dimension() -> int:
    """Return the embedding dimension used for internal documents retrieval."""
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


def build_openai_client() -> Any:
    """Build an OpenAI client from environment variables."""
    OpenAI = _load_openai_sdk()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_embedding_literal(embedding: list[float]) -> str:
    """Format an embedding as a pgvector literal."""
    return "[" + ",".join(str(value) for value in embedding) + "]"


def embed_query_text(
    query_text: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:
    """Create an embedding for a retrieval query."""
    resolved_client = client or build_openai_client()
    resolved_model = model or get_internal_documents_embedding_model()
    resolved_dimensions = dimensions or get_internal_documents_embedding_dimension()
    response = resolved_client.embeddings.create(
        model=resolved_model,
        input=query_text,
        dimensions=resolved_dimensions,
    )
    return list(response.data[0].embedding)


def search_internal_documents(
    query_embedding: list[float],
    *,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
    postgres_dsn: str | None = None,
) -> list[dict[str, Any]]:
    """Search the internal documents table by cosine similarity."""
    resolved_dsn = postgres_dsn or get_internal_documents_dsn()
    if not resolved_dsn:
        return []

    psycopg = _load_psycopg()
    with psycopg.connect(resolved_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    document_id,
                    source_path,
                    source_kind,
                    title,
                    content,
                    metadata,
                    1 - (embedding <=> %(embedding)s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %(embedding)s::vector
                LIMIT %(limit)s
                """,
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
            "content": row[4],
            "metadata": row[5],
            "score": row[6],
        }
        for row in rows
    ]


def _truncate_document_content(
    content: str,
    limit: int = DEFAULT_DOCUMENT_CONTEXT_CHAR_LIMIT,
) -> str:
    """Truncate retrieved document content before injecting it into the agent prompt."""
    if len(content) <= limit:
        return content
    return f"{content[: limit - 1]}…"


def _truncate_retrieved_documents(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Truncate retrieved document content before returning results to callers."""
    for result in results:
        content = result.get("content")
        if isinstance(content, str):
            result["content"] = _truncate_document_content(content)
    return results


def search_internal_documents_for_query(
    query_text: str,
    *,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> list[dict[str, Any]]:
    """Embed a raw query string and return top internal document matches."""
    if not get_internal_documents_dsn():
        return []

    query_embedding = embed_query_text(query_text)
    results = search_internal_documents(query_embedding, limit=limit)
    return _truncate_retrieved_documents(results)


def retrieve_internal_document_context_sync(
    message: Message,
    *,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> list[dict[str, Any]]:
    """Retrieve top internal documents for the current user message."""
    return search_internal_documents_for_query(message.text, limit=limit)


async def retrieve_internal_document_context(
    message: Message,
    *,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> list[dict[str, Any]]:
    """Retrieve top internal documents without blocking the async server loop."""
    return await asyncio.to_thread(retrieve_internal_document_context_sync, message, limit=limit)
