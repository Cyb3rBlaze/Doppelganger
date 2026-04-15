"""Tests for adaptive document chunking and sliding-window embeddings."""

from __future__ import annotations

from core import chunking
from core.document_sources import InternalDocument


def test_cosine_similarity_returns_expected_value() -> None:
    assert chunking.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_split_document_into_base_chunks_prefers_newline_boundaries_for_notes() -> None:
    document = InternalDocument(
        document_id="file:test",
        source_path="/tmp/test.md",
        source_kind="local_text",
        title="Test",
        content="- alpha idea\n- beta idea\n- gamma idea",
        metadata={},
    )

    chunks = chunking.split_document_into_base_chunks(document, max_chars=20)

    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.content for chunk in chunks] == [
        "- alpha idea",
        "- beta idea",
        "- gamma idea",
    ]


def test_split_text_by_char_budget_falls_back_to_whitespace_when_no_newline() -> None:
    chunks = chunking.split_text_by_char_budget(
        "alpha beta gamma delta epsilon",
        max_chars=12,
    )

    assert chunks == ["alpha beta", "gamma delta", "epsilon"]


def test_build_adaptive_document_chunks_merges_similar_neighbors() -> None:
    document = InternalDocument(
        document_id="file:test",
        source_path="/tmp/test.md",
        source_kind="local_text",
        title="Test",
        content="Alpha one.\n\nAlpha two.\n\nBeta topic.",
        metadata={"doc_id": "abc123"},
    )

    def fake_embed(text: str) -> list[float]:
        if "Alpha one.\n\nAlpha two." in text:
            return [0.98, 0.02]
        if "Beta topic." in text:
            return [0.0, 1.0]
        if "Alpha one." in text:
            return [1.0, 0.0]
        if "Alpha two." in text:
            return [0.95, 0.05]
        raise AssertionError(f"Unexpected embedding text: {text!r}")

    records = chunking.build_adaptive_document_chunks(
        document,
        embed_fn=fake_embed,
        base_chunk_char_limit=12,
        running_window_char_limit=400,
        merge_similarity_threshold=0.72,
    )

    assert len(records) == 2
    assert records[0].record.content == "Alpha one.\n\nAlpha two."
    assert records[0].record.window_start_chunk_index == 0
    assert records[0].record.window_end_chunk_index == 1
    assert records[1].record.content == "Beta topic."
    assert records[1].record.window_start_chunk_index == 2
    assert records[1].record.window_end_chunk_index == 2


def test_build_adaptive_document_chunks_slides_window_when_limit_is_exceeded() -> None:
    document = InternalDocument(
        document_id="file:test",
        source_path="/tmp/test.md",
        source_kind="local_text",
        title="Test",
        content="Alpha one.\n\nAlpha two.\n\nAlpha three.",
        metadata={},
    )

    def fake_embed(text: str) -> list[float]:
        if "Alpha one.\n\nAlpha two.\n\nAlpha three." in text:
            return [0.99, 0.01]
        if "Alpha two.\n\nAlpha three." in text:
            return [0.97, 0.03]
        if "Alpha one.\n\nAlpha two." in text:
            return [0.98, 0.02]
        if "Alpha three." in text:
            return [0.95, 0.05]
        if "Alpha two." in text:
            return [0.95, 0.05]
        if "Alpha one." in text:
            return [0.95, 0.05]
        raise AssertionError(f"Unexpected embedding text: {text!r}")

    records = chunking.build_adaptive_document_chunks(
        document,
        embed_fn=fake_embed,
        base_chunk_char_limit=15,
        running_window_char_limit=75,
        merge_similarity_threshold=0.72,
    )

    assert len(records) == 2
    assert records[0].record.content == "Alpha one.\n\nAlpha two."
    assert records[1].record.content == "Alpha two.\n\nAlpha three."
    assert records[1].record.window_start_chunk_index == 1
    assert records[1].record.window_end_chunk_index == 2


def test_build_adaptive_document_chunk_result_keeps_decision_trace() -> None:
    document = InternalDocument(
        document_id="file:test",
        source_path="/tmp/test.md",
        source_kind="local_text",
        title="Test",
        content="Alpha one.\n\nAlpha two.\n\nBeta topic.",
        metadata={},
    )

    def fake_embed(text: str) -> list[float]:
        if "Alpha one.\n\nAlpha two." in text:
            return [0.98, 0.02]
        if "Beta topic." in text:
            return [0.0, 1.0]
        if "Alpha one." in text:
            return [1.0, 0.0]
        if "Alpha two." in text:
            return [0.95, 0.05]
        raise AssertionError(f"Unexpected embedding text: {text!r}")

    result = chunking.build_adaptive_document_chunk_result(
        document,
        embed_fn=fake_embed,
        base_chunk_char_limit=12,
        running_window_char_limit=400,
        merge_similarity_threshold=0.72,
    )

    assert len(result.embedded_chunks) == 2
    assert len(result.decisions) == 3
    assert result.decisions[0].merged is False
    assert result.decisions[0].similarity_to_previous_window is None
    assert result.decisions[1].merged is True
    assert result.decisions[1].window_start_chunk_index == 0
    assert result.decisions[1].window_end_chunk_index == 1
    assert result.decisions[2].merged is False
