"""Postgres-backed daily message sessions for the AI doppelganger."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any, Literal

from app.core.models import Message

POSTGRES_DSN_ENV = "POSTGRES_DSN"

logger = logging.getLogger("doppelganger.history")

CREATE_MESSAGE_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS message_sessions (
    session_id TEXT PRIMARY KEY,
    session_date DATE NOT NULL,
    channel TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NULL,
    session_summary TEXT NULL,
    message_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

ALTER_MESSAGE_SESSIONS_ADD_SUMMARY_SQL = """
ALTER TABLE message_sessions
ADD COLUMN IF NOT EXISTS session_summary TEXT NULL
"""

UPSERT_MESSAGE_SESSION_SQL = """
INSERT INTO message_sessions (
    session_id,
    session_date,
    channel,
    user_id,
    conversation_id,
    message_history
) VALUES (
    %(session_id)s,
    %(session_date)s,
    %(channel)s,
    %(user_id)s,
    %(conversation_id)s,
    %(message_history)s::jsonb
)
ON CONFLICT (session_id) DO UPDATE SET
    message_history = message_sessions.message_history || EXCLUDED.message_history,
    updated_at = NOW()
"""

UPDATE_SESSION_SUMMARY_SQL = """
UPDATE message_sessions
SET session_summary = %(session_summary)s,
    updated_at = NOW()
WHERE session_id = %(session_id)s
"""


def _load_psycopg() -> Any:
    """Import psycopg lazily so the app can still run without Postgres configured."""
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Postgres dependencies are not installed. Run `pip install -e .` first."
        ) from exc
    return psycopg


def get_postgres_dsn() -> str | None:
    """Return the configured Postgres DSN if present."""
    return os.getenv(POSTGRES_DSN_ENV)


def is_configured() -> bool:
    """Return whether Postgres-backed message history is configured."""
    return bool(get_postgres_dsn())


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def _local_today() -> date:
    """Return the current local calendar day for session rollover."""
    return datetime.now().astimezone().date()


def build_session_id(message: Message, *, session_date: date | None = None) -> str:
    """Build the daily session identifier for a message."""
    resolved_date = session_date or _local_today()
    conversation_key = message.conversation_id or "unknown"
    return f"{message.channel}:{message.user_id}:{conversation_key}:{resolved_date.isoformat()}"


def _build_message_event(
    *,
    message: Message,
    direction: Literal["inbound", "outbound"],
    text: str,
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build one raw message event dict for the session history array."""
    resolved_created_at = created_at or _utc_now()
    merged_metadata = dict(message.metadata)
    if metadata:
        merged_metadata.update(metadata)
    return {
        "direction": direction,
        "text": text,
        "message_id": message.message_id,
        "metadata": merged_metadata,
        "created_at": resolved_created_at.isoformat(),
    }


def _build_session_row(
    *,
    message: Message,
    event: dict[str, Any],
    session_date: date | None = None,
) -> dict[str, Any]:
    """Build one upsert payload for the message_sessions table."""
    resolved_date = session_date or _local_today()
    return {
        "session_id": build_session_id(message, session_date=resolved_date),
        "session_date": resolved_date,
        "channel": message.channel,
        "user_id": message.user_id,
        "conversation_id": message.conversation_id,
        "message_history": json.dumps([event], default=str),
    }


@lru_cache(maxsize=None)
def ensure_schema(dsn: str) -> None:
    """Create the message_sessions table once per process when configured."""
    psycopg = _load_psycopg()
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_MESSAGE_SESSIONS_TABLE_SQL)
            cursor.execute(ALTER_MESSAGE_SESSIONS_ADD_SUMMARY_SQL)
        connection.commit()


def append_message_event(
    *,
    message: Message,
    direction: Literal["inbound", "outbound"],
    text: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Append one raw message event into the daily session row when configured."""
    dsn = get_postgres_dsn()
    if not dsn:
        return False

    ensure_schema(dsn)
    event = _build_message_event(
        message=message,
        direction=direction,
        text=text,
        metadata=metadata,
    )
    row = _build_session_row(message=message, event=event)
    psycopg = _load_psycopg()
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(UPSERT_MESSAGE_SESSION_SQL, row)
        connection.commit()

    logger.info(
        "status=stored direction=%s channel=%s user_id=%s session_id=%s message_id=%s",
        direction,
        message.channel,
        message.user_id,
        row["session_id"],
        message.message_id,
    )
    return True


async def append_message_event_async(
    *,
    message: Message,
    direction: Literal["inbound", "outbound"],
    text: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Append one raw message event without blocking the async caller."""
    return await asyncio.to_thread(
        append_message_event,
        message=message,
        direction=direction,
        text=text,
        metadata=metadata,
    )


def _normalize_message_history(value: Any) -> list[dict[str, Any]]:
    """Normalize a JSONB message history value into a Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def get_current_session_history(message: Message) -> list[dict[str, Any]]:
    """Return the stored message history for the message's current daily session."""
    dsn = get_postgres_dsn()
    if not dsn:
        return []

    ensure_schema(dsn)
    psycopg = _load_psycopg()
    session_id = build_session_id(message)
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT message_history
                FROM message_sessions
                WHERE session_id = %(session_id)s
                """,
                {"session_id": session_id},
            )
            row = cursor.fetchone()
    if not row:
        return []
    return _normalize_message_history(row[0])


def get_previous_session_summaries(message: Message, *, limit: int = 5) -> list[str]:
    """Return recent non-empty summaries from older sessions in the same thread."""
    dsn = get_postgres_dsn()
    if not dsn:
        return []

    ensure_schema(dsn)
    psycopg = _load_psycopg()
    session_date = _local_today()
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_summary
                FROM message_sessions
                WHERE channel = %(channel)s
                  AND user_id = %(user_id)s
                  AND conversation_id IS NOT DISTINCT FROM %(conversation_id)s
                  AND session_date < %(session_date)s
                  AND session_summary IS NOT NULL
                  AND session_summary <> ''
                ORDER BY session_date DESC
                LIMIT %(limit)s
                """,
                {
                    "channel": message.channel,
                    "user_id": message.user_id,
                    "conversation_id": message.conversation_id,
                    "session_date": session_date,
                    "limit": limit,
                },
            )
            rows = cursor.fetchall()
    return [row[0] for row in rows]


def get_current_session_summary(message: Message) -> str | None:
    """Return the stored summary for the message's current daily session."""
    dsn = get_postgres_dsn()
    if not dsn:
        return None

    ensure_schema(dsn)
    psycopg = _load_psycopg()
    session_id = build_session_id(message)
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_summary
                FROM message_sessions
                WHERE session_id = %(session_id)s
                """,
                {"session_id": session_id},
            )
            row = cursor.fetchone()
    if not row:
        return None
    return row[0]


async def get_current_session_history_async(message: Message) -> list[dict[str, Any]]:
    """Return the current daily session history without blocking the async caller."""
    return await asyncio.to_thread(get_current_session_history, message)


async def get_previous_session_summaries_async(
    message: Message,
    *,
    limit: int = 5,
) -> list[str]:
    """Return recent prior session summaries without blocking the async caller."""
    return await asyncio.to_thread(get_previous_session_summaries, message, limit=limit)


async def get_current_session_summary_async(message: Message) -> str | None:
    """Return the current daily session summary without blocking the async caller."""
    return await asyncio.to_thread(get_current_session_summary, message)


def update_session_summary(message: Message, summary: str) -> bool:
    """Persist the latest summary for the message's current daily session."""
    dsn = get_postgres_dsn()
    if not dsn:
        return False

    ensure_schema(dsn)
    psycopg = _load_psycopg()
    session_id = build_session_id(message)
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                UPDATE_SESSION_SUMMARY_SQL,
                {
                    "session_id": session_id,
                    "session_summary": summary,
                },
            )
        connection.commit()

    logger.info(
        "status=summary_updated channel=%s user_id=%s session_id=%s",
        message.channel,
        message.user_id,
        session_id,
    )
    return True


async def update_session_summary_async(message: Message, summary: str) -> bool:
    """Persist the current session summary without blocking the async caller."""
    return await asyncio.to_thread(update_session_summary, message, summary)
