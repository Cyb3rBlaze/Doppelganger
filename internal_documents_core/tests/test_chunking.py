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


def test_attach_connected_nodes_adds_typed_graph_edges() -> None:
    document = InternalDocument(
        document_id="file:test",
        source_path="/tmp/test.md",
        source_kind="local_text",
        title="Acme Strategy",
        content="unused",
        metadata={},
    )

    chunks = [
        chunking.EmbeddedChunkRecord(
            record=chunking.build_document_chunk_record(
                document,
                chunk_id="file:test:chunk:0",
                chunk_index=0,
                window_start_chunk_index=0,
                window_end_chunk_index=0,
                content="# Project Acme\nAcme market strategy plan",
            ),
            embedding=[1.0, 0.0],
        ),
        chunking.EmbeddedChunkRecord(
            record=chunking.build_document_chunk_record(
                document,
                chunk_id="file:test:chunk:1",
                chunk_index=1,
                window_start_chunk_index=1,
                window_end_chunk_index=1,
                content="# Project Acme\nAcme market roadmap strategy",
            ),
            embedding=[0.95, 0.05],
        ),
        chunking.EmbeddedChunkRecord(
            record=chunking.build_document_chunk_record(
                document,
                chunk_id="file:test:chunk:2",
                chunk_index=2,
                window_start_chunk_index=2,
                window_end_chunk_index=2,
                content="Completely different topic",
            ),
            embedding=[0.0, 1.0],
        ),
    ]

    connected = chunking.attach_connected_nodes(chunks)

    first_connections = connected[0].record.connected_nodes
    second_connections = connected[1].record.connected_nodes
    assert first_connections[0]["chunk_id"] == "file:test:chunk:1"
    assert "adjacent" in first_connections[0]["edge_types"]
    assert "same_document" in first_connections[0]["edge_types"]
    assert "semantic" in first_connections[0]["edge_types"]
    assert "entity_overlap" in first_connections[0]["edge_types"]
    assert "same_heading" in first_connections[0]["edge_types"]
    assert first_connections[0]["signals"]["adjacent"] == 1.0
    assert second_connections[0]["chunk_id"] == "file:test:chunk:0"


def test_attach_connected_nodes_keeps_edges_bidirectional() -> None:
    document = InternalDocument(
        document_id="file:test",
        source_path="/tmp/test.md",
        source_kind="local_text",
        title="Acme Strategy",
        content="unused",
        metadata={},
    )

    chunks = [
        chunking.EmbeddedChunkRecord(
            record=chunking.build_document_chunk_record(
                document,
                chunk_id="file:test:chunk:0",
                chunk_index=0,
                window_start_chunk_index=0,
                window_end_chunk_index=0,
                content="# Project Acme\nAcme market strategy plan",
            ),
            embedding=[1.0, 0.0],
        ),
        chunking.EmbeddedChunkRecord(
            record=chunking.build_document_chunk_record(
                document,
                chunk_id="file:test:chunk:1",
                chunk_index=1,
                window_start_chunk_index=1,
                window_end_chunk_index=1,
                content="# Project Acme\nAcme market roadmap strategy",
            ),
            embedding=[0.95, 0.05],
        ),
        chunking.EmbeddedChunkRecord(
            record=chunking.build_document_chunk_record(
                document,
                chunk_id="file:test:chunk:2",
                chunk_index=2,
                window_start_chunk_index=2,
                window_end_chunk_index=2,
                content="Completely different topic",
            ),
            embedding=[0.0, 1.0],
        ),
    ]

    connected = chunking.attach_connected_nodes(chunks)
    connection_maps = {
        chunk.record.chunk_id: {
            node["chunk_id"]: node for node in chunk.record.connected_nodes
        }
        for chunk in connected
    }

    for source_id, targets in connection_maps.items():
        for target_id, edge in targets.items():
            reverse_edge = connection_maps[target_id][source_id]
            assert set(reverse_edge["edge_types"]) == set(edge["edge_types"])
            assert reverse_edge["signals"] == edge["signals"]
            assert reverse_edge["score"] == edge["score"]
