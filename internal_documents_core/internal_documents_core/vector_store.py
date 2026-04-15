"""OpenAI Vector Store ingestion helpers for internal documents."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DOTENV_PATH = PROJECT_ROOT / ".env"
FALLBACK_DOTENV_PATH = PROJECT_ROOT.parent / "doppelganger_core" / ".env"
SUPPORTED_DOCUMENT_EXTENSIONS = (".md", ".txt", ".pdf", ".docx")

load_dotenv(LOCAL_DOTENV_PATH)
if FALLBACK_DOTENV_PATH.exists():
    load_dotenv(FALLBACK_DOTENV_PATH, override=False)


@dataclass(frozen=True)
class VectorStoreIngestionResult:
    """Compact result for one ingestion run."""

    vector_store_id: str
    file_count: int
    file_batch_status: str
    completed_count: int | None = None
    failed_count: int | None = None


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


def resolve_source_dir(source_dir: str | Path | None = None) -> Path:
    """Resolve the source directory for internal documents."""
    raw_value = str(source_dir) if source_dir is not None else os.getenv(
        "INTERNAL_DOCUMENTS_SOURCE_DIR",
        "documents",
    )
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def collect_document_paths(
    source_dir: str | Path | None = None,
    *,
    extensions: Iterable[str] = SUPPORTED_DOCUMENT_EXTENSIONS,
) -> list[Path]:
    """Collect supported documents recursively from a source directory."""
    resolved_source_dir = resolve_source_dir(source_dir)
    normalized_extensions = {extension.lower() for extension in extensions}
    if not resolved_source_dir.exists():
        raise RuntimeError(f"Document source directory does not exist: {resolved_source_dir}")
    if not resolved_source_dir.is_dir():
        raise RuntimeError(f"Document source path is not a directory: {resolved_source_dir}")

    return sorted(
        path
        for path in resolved_source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def get_vector_store_name() -> str:
    """Return the configured default vector store name."""
    return os.getenv("INTERNAL_DOCUMENTS_VECTOR_STORE_NAME", "Internal Documents")


def ensure_vector_store(
    *,
    client: Any | None = None,
    vector_store_id: str | None = None,
    vector_store_name: str | None = None,
) -> Any:
    """Create or retrieve the target vector store."""
    resolved_client = client or build_openai_client()
    if vector_store_id:
        return resolved_client.vector_stores.retrieve(vector_store_id)
    return resolved_client.vector_stores.create(name=vector_store_name or get_vector_store_name())


def ingest_documents_to_vector_store(
    *,
    source_dir: str | Path | None = None,
    vector_store_id: str | None = None,
    vector_store_name: str | None = None,
    client: Any | None = None,
) -> VectorStoreIngestionResult:
    """Upload supported documents into an OpenAI vector store and poll until ready."""
    document_paths = collect_document_paths(source_dir)
    if not document_paths:
        raise RuntimeError("No supported documents were found to ingest.")

    resolved_client = client or build_openai_client()
    vector_store = ensure_vector_store(
        client=resolved_client,
        vector_store_id=vector_store_id,
        vector_store_name=vector_store_name,
    )

    with ExitStack() as exit_stack:
        file_streams = [exit_stack.enter_context(path.open("rb")) for path in document_paths]
        file_batch = resolved_client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams,
        )

    file_counts = getattr(file_batch, "file_counts", None)
    completed_count = getattr(file_counts, "completed", None) if file_counts is not None else None
    failed_count = getattr(file_counts, "failed", None) if file_counts is not None else None
    return VectorStoreIngestionResult(
        vector_store_id=vector_store.id,
        file_count=len(document_paths),
        file_batch_status=getattr(file_batch, "status", "unknown"),
        completed_count=completed_count,
        failed_count=failed_count,
    )
