"""Adaptive chunk/window embedding logic for internal documents."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable

from core.document_sources import InternalDocument
from core.vector_store import EmbeddedChunkRecord, build_document_chunk_record

DEFAULT_BASE_CHUNK_CHAR_LIMIT = 1_800
DEFAULT_RUNNING_WINDOW_CHAR_LIMIT = 6_000
DEFAULT_MERGE_SIMILARITY_THRESHOLD = 0.72
DEFAULT_METADATA_KEYS = ("doc_id", "email", "resource_key")


@dataclass(frozen=True)
class BaseChunk:
    """One sequential source chunk before adaptive window merging."""

    index: int
    content: str


@dataclass(frozen=True)
class ChunkDecision:
    """One adaptive chunk/window decision recorded during ingest."""

    output_index: int
    base_chunk_index: int
    window_start_chunk_index: int
    window_end_chunk_index: int
    merged: bool
    similarity_to_previous_window: float | None


@dataclass(frozen=True)
class AdaptiveChunkBuildResult:
    """Adaptive chunk/window build output plus merge-decision trace."""

    embedded_chunks: list[EmbeddedChunkRecord]
    decisions: list[ChunkDecision]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity between two equal-length vectors."""
    if len(left) != len(right):
        raise RuntimeError("Vectors must have the same dimension for cosine similarity.")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise RuntimeError("Cosine similarity is undefined for zero vectors.")

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def format_document_metadata_for_embedding(
    metadata: dict[str, Any],
    *,
    preferred_keys: tuple[str, ...] = DEFAULT_METADATA_KEYS,
) -> str:
    """Format compact, stable document metadata for embedding input."""
    ordered_items: list[tuple[str, Any]] = []
    seen_keys: set[str] = set()

    for key in preferred_keys:
        if key in metadata and metadata[key] not in ("", None):
            ordered_items.append((key, metadata[key]))
            seen_keys.add(key)

    for key in sorted(metadata):
        if key in seen_keys:
            continue
        value = metadata[key]
        if value in ("", None):
            continue
        ordered_items.append((key, value))

    if not ordered_items:
        return ""
    return "\n".join(f"{key}: {value}" for key, value in ordered_items)


def build_chunk_embedding_text(document: InternalDocument, chunk_content: str) -> str:
    """Build the text payload sent to the embedding model for one chunk/window."""
    metadata_text = format_document_metadata_for_embedding(document.metadata)
    sections = [
        f"Title: {document.title}",
        f"Source kind: {document.source_kind}",
    ]
    if metadata_text:
        sections.append(f"Metadata:\n{metadata_text}")
    sections.append(f"Content:\n{chunk_content.strip()}")
    return "\n\n".join(section for section in sections if section).strip()


def _find_chunk_split_index(text: str, start: int, max_chars: int) -> int:
    """Find a clean split index near the character budget."""
    hard_end = min(len(text), start + max_chars)
    if hard_end >= len(text):
        return len(text)

    window = text[start:hard_end]
    newline_index = window.rfind("\n")
    if newline_index > 0:
        return start + newline_index + 1

    punctuation_match = re.search(r"[.!?]\s+(?!.*[.!?]\s)", window)
    if punctuation_match is not None and punctuation_match.end() > 0:
        return start + punctuation_match.end()

    whitespace_index = window.rfind(" ")
    if whitespace_index > 0:
        return start + whitespace_index + 1

    return hard_end


def split_text_by_char_budget(text: str, *, max_chars: int) -> list[str]:
    """Split text by character count, preferring newline and sentence boundaries."""
    normalized = text.strip()
    if not normalized:
        return []
    if max_chars <= 0:
        raise RuntimeError("max_chars must be greater than zero.")

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        split_index = _find_chunk_split_index(normalized, start, max_chars)
        if split_index <= start:
            split_index = min(len(normalized), start + max_chars)
        chunk = normalized[start:split_index].strip()
        if chunk:
            chunks.append(chunk)
        start = split_index
        while start < len(normalized) and normalized[start].isspace():
            start += 1
    return chunks


def split_document_into_base_chunks(
    document: InternalDocument,
    *,
    max_chars: int = DEFAULT_BASE_CHUNK_CHAR_LIMIT,
) -> list[BaseChunk]:
    """Split a document into character-budgeted base chunks."""
    if max_chars <= 0:
        raise RuntimeError("max_chars must be greater than zero.")

    chunks = split_text_by_char_budget(document.content, max_chars=max_chars)
    return [BaseChunk(index=index, content=content) for index, content in enumerate(chunks)]


def _join_window_contents(window_chunks: list[BaseChunk]) -> str:
    """Join base chunks into the current adaptive window body."""
    return "\n\n".join(chunk.content for chunk in window_chunks)


def _shrink_window_to_limit(
    document: InternalDocument,
    window_chunks: list[BaseChunk],
    *,
    running_window_char_limit: int,
) -> list[BaseChunk]:
    """Pop oldest chunks until the framed embedding input fits the running-window limit."""
    trimmed = list(window_chunks)
    while trimmed:
        candidate_text = build_chunk_embedding_text(document, _join_window_contents(trimmed))
        if len(candidate_text) <= running_window_char_limit:
            return trimmed
        trimmed.pop(0)
    return []


def build_adaptive_document_chunks(
    document: InternalDocument,
    *,
    embed_fn: Callable[[str], list[float]],
    base_chunk_char_limit: int = DEFAULT_BASE_CHUNK_CHAR_LIMIT,
    running_window_char_limit: int = DEFAULT_RUNNING_WINDOW_CHAR_LIMIT,
    merge_similarity_threshold: float = DEFAULT_MERGE_SIMILARITY_THRESHOLD,
) -> list[EmbeddedChunkRecord]:
    """Build adaptive chunk/window embeddings for one document."""
    return build_adaptive_document_chunk_result(
        document,
        embed_fn=embed_fn,
        base_chunk_char_limit=base_chunk_char_limit,
        running_window_char_limit=running_window_char_limit,
        merge_similarity_threshold=merge_similarity_threshold,
    ).embedded_chunks


def build_adaptive_document_chunk_result(
    document: InternalDocument,
    *,
    embed_fn: Callable[[str], list[float]],
    base_chunk_char_limit: int = DEFAULT_BASE_CHUNK_CHAR_LIMIT,
    running_window_char_limit: int = DEFAULT_RUNNING_WINDOW_CHAR_LIMIT,
    merge_similarity_threshold: float = DEFAULT_MERGE_SIMILARITY_THRESHOLD,
) -> AdaptiveChunkBuildResult:
    """Build adaptive chunk/window embeddings and decision trace for one document."""
    if running_window_char_limit <= 0:
        raise RuntimeError("running_window_char_limit must be greater than zero.")
    if not 0 <= merge_similarity_threshold <= 1:
        raise RuntimeError("merge_similarity_threshold must be between 0 and 1.")

    base_chunks = split_document_into_base_chunks(document, max_chars=base_chunk_char_limit)
    if not base_chunks:
        return AdaptiveChunkBuildResult(embedded_chunks=[], decisions=[])

    results: list[EmbeddedChunkRecord] = []
    decisions: list[ChunkDecision] = []
    current_window_chunks = [base_chunks[0]]
    current_window_embedding = embed_fn(build_chunk_embedding_text(document, base_chunks[0].content))

    def record_decision(
        output_index: int,
        *,
        base_chunk_index: int,
        merged: bool,
        similarity_to_previous_window: float | None,
        window_start_chunk_index: int,
        window_end_chunk_index: int,
    ) -> None:
        decisions.append(
            ChunkDecision(
                output_index=output_index,
                base_chunk_index=base_chunk_index,
                window_start_chunk_index=window_start_chunk_index,
                window_end_chunk_index=window_end_chunk_index,
                merged=merged,
                similarity_to_previous_window=similarity_to_previous_window,
            )
        )

    def finalize_current_window() -> None:
        start_index = current_window_chunks[0].index
        end_index = current_window_chunks[-1].index
        results.append(
            EmbeddedChunkRecord(
                record=build_document_chunk_record(
                    document,
                    chunk_id=f"{document.document_id}:chunk:{len(results)}",
                    chunk_index=len(results),
                    window_start_chunk_index=start_index,
                    window_end_chunk_index=end_index,
                    content=_join_window_contents(current_window_chunks),
                ),
                embedding=current_window_embedding,
            )
        )

    record_decision(
        0,
        base_chunk_index=base_chunks[0].index,
        merged=False,
        similarity_to_previous_window=None,
        window_start_chunk_index=base_chunks[0].index,
        window_end_chunk_index=base_chunks[0].index,
    )

    for output_index, base_chunk in enumerate(base_chunks[1:], start=1):
        current_chunk_text = build_chunk_embedding_text(document, base_chunk.content)
        current_chunk_embedding = embed_fn(current_chunk_text)

        similarity = cosine_similarity(current_window_embedding, current_chunk_embedding)
        merged = False
        if similarity < merge_similarity_threshold:
            finalize_current_window()
            current_window_chunks = [base_chunk]
            current_window_embedding = current_chunk_embedding
            decision_start_index = base_chunk.index
            decision_end_index = base_chunk.index
        else:
            candidate_window_chunks = _shrink_window_to_limit(
                document,
                current_window_chunks + [base_chunk],
                running_window_char_limit=running_window_char_limit,
            )
            if not candidate_window_chunks:
                finalize_current_window()
                current_window_chunks = [base_chunk]
                current_window_embedding = current_chunk_embedding
                decision_start_index = base_chunk.index
                decision_end_index = base_chunk.index
            else:
                if candidate_window_chunks[0].index > current_window_chunks[0].index:
                    finalize_current_window()
                combined_text = build_chunk_embedding_text(
                    document,
                    _join_window_contents(candidate_window_chunks),
                )
                current_window_chunks = candidate_window_chunks
                current_window_embedding = embed_fn(combined_text)
                merged = True
                decision_start_index = current_window_chunks[0].index
                decision_end_index = current_window_chunks[-1].index

        record_decision(
            output_index,
            base_chunk_index=base_chunk.index,
            merged=merged,
            similarity_to_previous_window=similarity,
            window_start_chunk_index=decision_start_index,
            window_end_chunk_index=decision_end_index,
        )
    finalize_current_window()
    return AdaptiveChunkBuildResult(embedded_chunks=results, decisions=decisions)
