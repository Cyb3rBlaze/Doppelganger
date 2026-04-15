"""Embedding helpers for internal_documents_core."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DOTENV_PATH = PROJECT_ROOT / ".env"
FALLBACK_DOTENV_PATH = PROJECT_ROOT.parent / "doppelganger_core" / ".env"
EMBEDDING_MODEL_ENV = "INTERNAL_DOCUMENTS_EMBEDDING_MODEL"
EMBEDDING_DIMENSION_ENV = "INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

load_dotenv(LOCAL_DOTENV_PATH)
if FALLBACK_DOTENV_PATH.exists():
    load_dotenv(FALLBACK_DOTENV_PATH, override=False)


def _load_openai_sdk() -> Any:
    """Import the OpenAI SDK lazily."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK is not installed for internal_documents_core. "
            "Run `python -m pip install -e .` inside internal_documents_core first."
        ) from exc
    return OpenAI


def build_openai_client() -> Any:
    """Build an OpenAI client from environment variables."""
    OpenAI = _load_openai_sdk()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding_model() -> str:
    """Return the embedding model used for document and query vectors."""
    return os.getenv(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)


def get_embedding_dimension(model: str | None = None) -> int:
    """Return the embedding dimension, preferring explicit env configuration."""
    raw_value = os.getenv(EMBEDDING_DIMENSION_ENV)
    if raw_value:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(f"{EMBEDDING_DIMENSION_ENV} must be an integer.") from exc
        if value <= 0:
            raise RuntimeError(f"{EMBEDDING_DIMENSION_ENV} must be greater than zero.")
        return value

    resolved_model = model or get_embedding_model()
    if resolved_model in DEFAULT_MODEL_DIMENSIONS:
        return DEFAULT_MODEL_DIMENSIONS[resolved_model]
    raise RuntimeError(
        f"{EMBEDDING_DIMENSION_ENV} is not set and there is no default dimension mapping "
        f"for model {resolved_model!r}."
    )


def embed_text(
    text: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:
    """Create one embedding vector for a document or query string."""
    resolved_model = model or get_embedding_model()
    resolved_dimensions = dimensions or get_embedding_dimension(resolved_model)
    resolved_client = client or build_openai_client()
    response = resolved_client.embeddings.create(
        model=resolved_model,
        input=text,
        dimensions=resolved_dimensions,
    )
    return list(response.data[0].embedding)


def is_context_length_error(exc: Exception) -> bool:
    """Return whether an embedding failure looks like an input-too-large error."""
    message = str(exc).lower()
    return "maximum context length" in message or (
        "invalid 'input'" in message and "tokens" in message
    )
