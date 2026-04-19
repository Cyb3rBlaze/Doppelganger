"""Tests for unified memory graph schema and backfill helpers."""

from __future__ import annotations

import json

from app.services import unified_memory


class FakeCursor:
    def __init__(self, *, fetchall_result=None) -> None:
        self.executed: list[tuple[str, object | None]] = []
        self.fetchall_result = fetchall_result or []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))

    def fetchall(self):
        return self.fetchall_result


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


def test_ensure_unified_memory_schema_executes_expected_sql(monkeypatch) -> None:
    unified_memory.ensure_unified_memory_schema.cache_clear()
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    psycopg = FakePsycopg(connection)

    monkeypatch.setattr(unified_memory, "_load_psycopg", lambda: psycopg)

    unified_memory.ensure_unified_memory_schema(
        "postgresql://localhost/doppelganger",
        1536,
    )

    assert psycopg.calls == ["postgresql://localhost/doppelganger"]
    assert cursor.executed[0][0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert cursor.executed[1][0].strip().startswith("CREATE TABLE IF NOT EXISTS memory_nodes")
    assert "embedding VECTOR(1536) NULL" in cursor.executed[1][0]
    assert cursor.executed[2][0].strip().startswith("CREATE TABLE IF NOT EXISTS memory_edges")
    assert cursor.executed[3][0].strip().startswith("CREATE INDEX IF NOT EXISTS memory_nodes_node_type_idx")
    assert connection.commits == 1


def test_backfill_message_sessions_to_unified_memory_inserts_nodes_and_edges(monkeypatch) -> None:
    unified_memory.ensure_unified_memory_schema.cache_clear()
    cursor = FakeCursor(
        fetchall_result=[
            (
                "api:anshul:thread-1:2026-04-14",
                "2026-04-14",
                "api",
                "anshul",
                "thread-1",
                "Summary text",
                [
                    {
                        "direction": "inbound",
                        "text": "hello",
                        "message_id": "msg-1",
                        "metadata": {"source": "api"},
                        "created_at": "2026-04-14T12:00:00+00:00",
                    },
                    {
                        "direction": "outbound",
                        "text": "hi there",
                        "message_id": "msg-2",
                        "metadata": {"source": "agent"},
                        "created_at": "2026-04-14T12:01:00+00:00",
                    },
                ],
            )
        ]
    )
    connection = FakeConnection(cursor)
    psycopg = FakePsycopg(connection)

    monkeypatch.setattr(unified_memory, "_load_psycopg", lambda: psycopg)
    monkeypatch.setattr(
        unified_memory,
        "embed_memory_texts",
        lambda texts, **kwargs: [[float(index + 1), float(index + 2)] for index, _ in enumerate(texts)],
    )
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://localhost/doppelganger")

    result = unified_memory.backfill_message_sessions_to_unified_memory()

    assert result == {"session_count": 1, "node_count": 3, "edge_count": 6}
    assert psycopg.calls == [
        "postgresql://localhost/doppelganger",
        "postgresql://localhost/doppelganger",
    ]
    assert any(
        "FROM message_sessions" in sql
        for sql, _ in cursor.executed
    )
    inserted_node_calls = [
        entry for entry in cursor.executed if entry[0].strip().startswith("INSERT INTO memory_nodes")
    ]
    inserted_edge_calls = [
        entry for entry in cursor.executed if entry[0].strip().startswith("INSERT INTO memory_edges")
    ]
    assert len(inserted_node_calls) == 3
    assert len(inserted_edge_calls) == 6
    first_message_params = inserted_node_calls[0][1]
    assert first_message_params["node_id"] == "message:api:anshul:thread-1:2026-04-14:0"
    assert first_message_params["node_type"] == "message"
    assert first_message_params["content"] == "hello"
    assert first_message_params["embedding"] == "[1.0,2.0]"
    summary_params = inserted_node_calls[2][1]
    assert summary_params["node_id"] == "session_summary:api:anshul:thread-1:2026-04-14"
    assert summary_params["node_type"] == "session_summary"
    assert summary_params["embedding"] == "[3.0,4.0]"


def test_backfill_document_chunks_to_unified_memory_inserts_nodes_and_edges(monkeypatch) -> None:
    unified_memory.ensure_unified_memory_schema.cache_clear()
    source_cursor = FakeCursor(
        fetchall_result=[
            (
                "gdoc:abc123",
                "gdoc:abc123:chunk:0",
                "/docs/investing.gdoc",
                "gdoc",
                "Investing",
                "Chunk text",
                {"doc_id": "abc123"},
                [
                    {
                        "chunk_id": "gdoc:abc123:chunk:1",
                        "score": 1.0,
                        "edge_types": ["adjacent"],
                        "signals": {"adjacent": 1.0},
                    }
                ],
                0,
                0,
                0,
                [0.1, 0.2],
            )
        ]
    )
    source_connection = FakeConnection(source_cursor)
    target_cursor = FakeCursor()
    target_connection = FakeConnection(target_cursor)
    psycopg = FakePsycopg(target_connection)
    psycopg.side_effects = [target_connection, source_connection, target_connection]

    monkeypatch.setattr(unified_memory, "_load_psycopg", lambda: psycopg)
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://localhost/doppelganger")
    monkeypatch.setenv(
        "INTERNAL_DOCUMENTS_POSTGRES_DSN",
        "postgresql://localhost/internal_documents",
    )

    result = unified_memory.backfill_document_chunks_to_unified_memory()

    assert result == {"chunk_count": 1, "node_count": 1, "edge_count": 1}
    assert psycopg.calls == [
        "postgresql://localhost/doppelganger",
        "postgresql://localhost/internal_documents",
        "postgresql://localhost/doppelganger",
    ]
    assert source_cursor.executed[0][0].strip().startswith("SELECT")
    inserted_node_call = next(
        entry for entry in target_cursor.executed if entry[0].strip().startswith("INSERT INTO memory_nodes")
    )
    inserted_edge_call = next(
        entry for entry in target_cursor.executed if entry[0].strip().startswith("INSERT INTO memory_edges")
    )
    assert inserted_node_call[1]["node_id"] == "document_chunk:gdoc:abc123:chunk:0"
    assert inserted_node_call[1]["embedding"] == "[0.1,0.2]"
    assert inserted_edge_call[1]["source_node_id"] == "document_chunk:gdoc:abc123:chunk:0"
    assert inserted_edge_call[1]["target_node_id"] == "document_chunk:gdoc:abc123:chunk:1"


def test_backfill_message_document_semantic_edges_inserts_bidirectional_cross_type_edges(
    monkeypatch,
) -> None:
    unified_memory.ensure_unified_memory_schema.cache_clear()
    cursor = FakeCursor(
        fetchall_result=[
            ("message:telegram:1", "document_chunk:gdoc:abc:chunk:0", 0.88),
            ("message:telegram:1", "document_chunk:gdoc:abc:chunk:1", 0.64),
        ]
    )
    connection = FakeConnection(cursor)
    psycopg = FakePsycopg(connection)

    monkeypatch.setattr(unified_memory, "_load_psycopg", lambda: psycopg)
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://localhost/doppelganger")

    result = unified_memory.backfill_message_document_semantic_edges()

    assert result == {
        "message_count": 1,
        "document_chunk_count": 2,
        "edge_count": 4,
    }
    assert psycopg.calls == [
        "postgresql://localhost/doppelganger",
        "postgresql://localhost/doppelganger",
    ]
    assert any(
        sql.strip().startswith("DELETE FROM memory_edges") for sql, _ in cursor.executed
    )
    assert any("FROM memory_nodes AS message" in sql for sql, _ in cursor.executed)
    inserted_edge_calls = [
        entry for entry in cursor.executed if entry[0].strip().startswith("INSERT INTO memory_edges")
    ]
    assert len(inserted_edge_calls) == 4
    first_edge_params = inserted_edge_calls[0][1]
    assert first_edge_params["source_node_id"] == "message:telegram:1"
    assert first_edge_params["target_node_id"] == "document_chunk:gdoc:abc:chunk:0"
    assert json.loads(first_edge_params["edge_types"]) == ["semantic", "message_document"]
    assert json.loads(first_edge_params["signals"]) == {
        "semantic": 0.88,
        "message_document": 1.0,
    }


def test_main_prints_json_summary_for_full_backfill(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        unified_memory,
        "get_unified_memory_dsn",
        lambda: "postgresql://localhost/doppelganger",
    )
    monkeypatch.setattr(unified_memory, "get_embedding_dimension", lambda: 1536)
    ensure_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        unified_memory,
        "ensure_unified_memory_schema",
        lambda dsn, embedding_dimension: ensure_calls.append((dsn, embedding_dimension)),
    )
    monkeypatch.setattr(
        unified_memory,
        "backfill_message_sessions_to_unified_memory",
        lambda target_dsn=None: {"session_count": 1, "node_count": 2, "edge_count": 3},
    )
    monkeypatch.setattr(
        unified_memory,
        "backfill_document_chunks_to_unified_memory",
        lambda target_dsn=None: {"chunk_count": 4, "node_count": 4, "edge_count": 5},
    )
    monkeypatch.setattr(
        unified_memory,
        "backfill_message_document_semantic_edges",
        lambda target_dsn=None: {
            "message_count": 2,
            "document_chunk_count": 3,
            "edge_count": 6,
        },
    )

    unified_memory.main([])

    assert ensure_calls == [
        ("postgresql://localhost/doppelganger", 1536),
        ("postgresql://localhost/doppelganger", 1536),
    ]
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["status"] == "ok"
    assert payload["message_sessions"]["session_count"] == 1
    assert payload["document_chunks"]["chunk_count"] == 4
    assert payload["message_document_edges"]["edge_count"] == 6
