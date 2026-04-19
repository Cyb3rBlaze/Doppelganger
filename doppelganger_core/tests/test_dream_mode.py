"""Tests for one-shot dream-mode graph passes."""

from __future__ import annotations

import json

from app.services import dream_mode


class FakeCursor:
    def __init__(self, *, fetchall_results=None) -> None:
        self.executed: list[tuple[str, object | None]] = []
        self.fetchall_results = list(fetchall_results or [])

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))

    def fetchall(self):
        if self.fetchall_results:
            return self.fetchall_results.pop(0)
        return []


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.commits = 0

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1


class FakePsycopg:
    def __init__(self, default_connection: FakeConnection) -> None:
        self.default_connection = default_connection
        self.calls: list[str] = []
        self.side_effects: list[FakeConnection] = []

    def connect(self, dsn: str, **kwargs) -> FakeConnection:
        _ = kwargs
        self.calls.append(dsn)
        if self.side_effects:
            return self.side_effects.pop(0)
        return self.default_connection


def test_run_dream_iteration_creates_bidirectional_message_document_edges(
    monkeypatch,
) -> None:
    node_cursor = FakeCursor(
        fetchall_results=[
            [
                (
                    "message:telegram:1",
                    "message",
                    "telegram inbound",
                    "Should I keep investing in index funds?",
                    {"channel": "telegram"},
                    [1.0, 0.0],
                ),
                (
                    "document_chunk:gdoc:1:chunk:0",
                    "document_chunk",
                    "Investing notes",
                    "Index funds and long term investing notes",
                    {"document_id": "gdoc:1", "source_kind": "gdoc"},
                    [0.96, 0.04],
                ),
                (
                    "message:telegram:2",
                    "message",
                    "telegram inbound",
                    "What should I cook tonight?",
                    {"channel": "telegram"},
                    [0.0, 1.0],
                ),
            ]
        ]
    )
    edge_cursor = FakeCursor(fetchall_results=[[("message:telegram:2", "message:telegram:1")]])
    write_cursor = FakeCursor()

    node_connection = FakeConnection(node_cursor)
    edge_connection = FakeConnection(edge_cursor)
    write_connection = FakeConnection(write_cursor)
    psycopg = FakePsycopg(write_connection)
    psycopg.side_effects = [node_connection, edge_connection, write_connection]

    monkeypatch.setattr(dream_mode.unified_memory, "_load_psycopg", lambda: psycopg)
    monkeypatch.setattr(
        dream_mode.unified_memory,
        "ensure_unified_memory_schema",
        lambda dsn, embedding_dimension: None,
    )
    monkeypatch.setattr(
        dream_mode.unified_memory,
        "get_unified_memory_dsn",
        lambda: "postgresql://localhost/doppelganger",
    )
    monkeypatch.setattr(dream_mode.unified_memory, "get_embedding_dimension", lambda: 1536)

    result = dream_mode.run_dream_iteration(
        semantic_threshold=0.7,
        combined_threshold=0.6,
        max_new_edges_per_node=4,
    )

    assert result["embedded_node_count"] == 3
    assert result["compared_pair_count"] == 2
    assert result["created_pair_count"] == 1
    assert result["created_edge_count"] == 2

    delete_calls = [
        entry for entry in write_cursor.executed if entry[0].strip().startswith("DELETE FROM memory_edges")
    ]
    assert len(delete_calls) == 1
    inserted_edge_calls = [
        entry for entry in write_cursor.executed if entry[0].strip().startswith("INSERT INTO memory_edges")
    ]
    assert len(inserted_edge_calls) == 2
    first_edge_params = inserted_edge_calls[0][1]
    assert first_edge_params["source_node_id"] == "message:telegram:1"
    assert first_edge_params["target_node_id"] == "document_chunk:gdoc:1:chunk:0"
    assert json.loads(first_edge_params["edge_types"]) == [
        "dream",
        "dream_semantic",
        "dream_relevance",
        "dream_cross_type",
        "message_document",
    ]
    signals = json.loads(first_edge_params["signals"])
    assert signals["semantic"] > 0.9
    assert signals["relevance"] > 0.2
    assert signals["dream"] > 0.7


def test_build_dream_keywords_ignores_stopwords() -> None:
    keywords = dream_mode.build_dream_keywords(
        node_type="message",
        title="The note",
        content="This is a note about investing and index funds",
        metadata={"channel": "telegram"},
    )

    assert "the" not in keywords
    assert "investing" in keywords
    assert "index" in keywords
