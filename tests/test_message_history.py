"""Tests for Postgres-backed message session history helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.core.models import Message
from app.services import message_history


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object | None]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))


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

    def connect(self, dsn: str) -> FakeConnection:
        self.calls.append(dsn)
        return self.connection


def test_build_session_id_uses_daily_key() -> None:
    message = Message(
        channel="telegram",
        user_id="6891176979",
        text="hello",
        conversation_id="6891176979",
    )

    session_id = message_history.build_session_id(
        message,
        session_date=date(2026, 4, 14),
    )

    assert session_id == "telegram:6891176979:6891176979:2026-04-14"


def test_build_message_event_returns_raw_dict(monkeypatch) -> None:
    monkeypatch.setattr(
        message_history,
        "_utc_now",
        lambda: datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
    )
    message = Message(
        channel="api",
        user_id="anshul",
        text="hello",
        message_id="msg-1",
        metadata={"source": "api"},
    )

    event = message_history._build_message_event(
        message=message,
        direction="inbound",
        text="hello",
        metadata={"kind": "raw"},
    )

    assert event["direction"] == "inbound"
    assert event["text"] == "hello"
    assert event["message_id"] == "msg-1"
    assert event["metadata"] == {"source": "api", "kind": "raw"}
    assert event["created_at"] == "2026-04-14T12:00:00+00:00"


def test_append_message_event_returns_false_when_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    stored = message_history.append_message_event(
        message=Message(channel="api", user_id="anshul", text="hello"),
        direction="inbound",
        text="hello",
    )

    assert stored is False


def test_append_message_event_initializes_schema_and_upserts(monkeypatch) -> None:
    message_history.ensure_schema.cache_clear()
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    psycopg = FakePsycopg(connection)

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://localhost/doppelganger")
    monkeypatch.setattr(message_history, "_load_psycopg", lambda: psycopg)
    monkeypatch.setattr(
        message_history,
        "_utc_now",
        lambda: datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
    )

    stored = message_history.append_message_event(
        message=Message(
            channel="api",
            user_id="anshul",
            text="hello",
            conversation_id="thread-1",
            message_id="msg-1",
        ),
        direction="inbound",
        text="hello",
    )

    assert stored is True
    assert psycopg.calls == [
        "postgresql://localhost/doppelganger",
        "postgresql://localhost/doppelganger",
    ]
    assert cursor.executed[0][0].strip().startswith("CREATE TABLE IF NOT EXISTS message_sessions")
    assert cursor.executed[1][0].strip().startswith("INSERT INTO message_sessions")
    row = cursor.executed[1][1]
    assert row["session_id"] == "api:anshul:thread-1:2026-04-14"
    assert row["conversation_id"] == "thread-1"
    assert '"direction": "inbound"' in row["message_history"]
    assert '"text": "hello"' in row["message_history"]
