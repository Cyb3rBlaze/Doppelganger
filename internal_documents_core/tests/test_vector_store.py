"""Tests for pgvector storage helpers in core."""

from __future__ import annotations

import pytest

from core import vector_store


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object | None]] = []
        self.fetchall_result = []

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
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.calls: list[str] = []
        self.connect_kwargs: list[dict[str, object]] = []
        self.side_effects: list[object] = []

    class OperationalError(Exception):
        pass

    def connect(self, dsn: str, **kwargs) -> FakeConnection:
        self.calls.append(dsn)
        self.connect_kwargs.append(kwargs)
        if self.side_effects:
            next_result = self.side_effects.pop(0)
            if isinstance(next_result, Exception):
                raise next_result
            return next_result
        return self.connection


def test_get_vector_store_config_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_DOCUMENTS_POSTGRES_DSN", "postgresql://localhost/internal_docs")
    monkeypatch.setenv("INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION", "1536")

    config = vector_store.get_vector_store_config()

    assert config.postgres_dsn == "postgresql://localhost/internal_docs"
    assert config.embedding_dimension == 1536


def test_get_embedding_dimension_requires_positive_integer(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION", "0")

    with pytest.raises(RuntimeError):
        vector_store.get_embedding_dimension()


def test_ensure_pgvector_schema_executes_extension_and_table_sql(monkeypatch) -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    psycopg = FakePsycopg(connection)

    monkeypatch.setattr(vector_store, "_load_psycopg", lambda: psycopg)
    monkeypatch.setattr(vector_store, "ensure_database_exists", lambda **kwargs: False)

    stored = vector_store.ensure_pgvector_schema(
        config=vector_store.VectorStoreConfig(
            postgres_dsn="postgresql://localhost/internal_docs",
            embedding_dimension=1536,
        )
    )

    assert stored is True
    assert psycopg.calls == ["postgresql://localhost/internal_docs"]
    assert cursor.executed[0][0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert cursor.executed[1][0].strip().startswith("CREATE TABLE IF NOT EXISTS documents")
    assert "embedding VECTOR(1536) NOT NULL" in cursor.executed[1][0]
    assert cursor.executed[1][1] is None
    assert cursor.executed[2][0].strip().startswith("CREATE INDEX IF NOT EXISTS")
    assert connection.commits == 1


def test_build_create_documents_table_sql_includes_literal_dimension() -> None:
    sql = vector_store.build_create_documents_table_sql(1536)

    assert "embedding VECTOR(1536) NOT NULL" in sql


def test_get_database_name_from_dsn_extracts_path() -> None:
    assert (
        vector_store.get_database_name_from_dsn(
            "postgresql://anshul:secret@127.0.0.1:5432/internal_documents"
        )
        == "internal_documents"
    )


def test_build_maintenance_dsn_retargets_database() -> None:
    assert (
        vector_store.build_maintenance_dsn(
            "postgresql://anshul:secret@127.0.0.1:5432/internal_documents"
        )
        == "postgresql://anshul:secret@127.0.0.1:5432/postgres"
    )


def test_ensure_database_exists_creates_missing_database(monkeypatch) -> None:
    target_connection = FakeConnection(FakeCursor())
    admin_cursor = FakeCursor()
    admin_connection = FakeConnection(admin_cursor)
    psycopg = FakePsycopg(target_connection)
    psycopg.side_effects = [
        psycopg.OperationalError('connection failed: FATAL:  database "internal_documents" does not exist'),
        admin_connection,
    ]

    monkeypatch.setattr(vector_store, "_load_psycopg", lambda: psycopg)

    created = vector_store.ensure_database_exists(
        config=vector_store.VectorStoreConfig(
            postgres_dsn="postgresql://anshul:secret@127.0.0.1:5432/internal_documents",
            embedding_dimension=1536,
        )
    )

    assert created is True
    assert psycopg.calls == [
        "postgresql://anshul:secret@127.0.0.1:5432/internal_documents",
        "postgresql://anshul:secret@127.0.0.1:5432/postgres",
    ]
    assert psycopg.connect_kwargs[1] == {"autocommit": True}
    assert admin_cursor.executed == [('CREATE DATABASE "internal_documents"', None)]


def test_format_embedding_literal_returns_pgvector_literal() -> None:
    assert vector_store.format_embedding_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def test_upsert_document_writes_one_document_row(monkeypatch) -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    psycopg = FakePsycopg(connection)

    monkeypatch.setattr(vector_store, "_load_psycopg", lambda: psycopg)

    document = vector_store.InternalDocument(
        document_id="gdoc:abc123",
        source_path="/tmp/note.gdoc",
        source_kind="gdoc",
        title="note",
        content="hello world",
        metadata={"doc_id": "abc123"},
    )

    stored = vector_store.upsert_document(
        document,
        [0.1, 0.2],
        config=vector_store.VectorStoreConfig(
            postgres_dsn="postgresql://localhost/internal_docs",
            embedding_dimension=1536,
        ),
    )

    assert stored is True
    assert cursor.executed[0][0].strip().startswith("INSERT INTO documents")
    params = cursor.executed[0][1]
    assert params["document_id"] == "gdoc:abc123"
    assert params["source_kind"] == "gdoc"
    assert params["embedding"] == "[0.1,0.2]"


def test_search_documents_returns_ranked_rows(monkeypatch) -> None:
    cursor = FakeCursor()
    cursor.fetchall_result = [
        ("gdoc:abc123", "/tmp/note.gdoc", "gdoc", "note", {"doc_id": "abc123"}, 0.9),
    ]
    connection = FakeConnection(cursor)
    psycopg = FakePsycopg(connection)

    monkeypatch.setattr(vector_store, "_load_psycopg", lambda: psycopg)

    results = vector_store.search_documents(
        [0.1, 0.2],
        limit=3,
        config=vector_store.VectorStoreConfig(
            postgres_dsn="postgresql://localhost/internal_docs",
            embedding_dimension=1536,
        ),
    )

    assert results == [
        {
            "document_id": "gdoc:abc123",
            "source_path": "/tmp/note.gdoc",
            "source_kind": "gdoc",
            "title": "note",
            "metadata": {"doc_id": "abc123"},
            "score": 0.9,
        }
    ]
    assert cursor.executed[0][0].strip().startswith("SELECT")
