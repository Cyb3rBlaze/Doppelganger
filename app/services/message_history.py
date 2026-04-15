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
    message_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
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


def build_session_id(message: Message, *, session_date: date | None = None) -> str:
    """Build the daily session identifier for a message."""
    resolved_date = session_date or _utc_now().date()
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
    resolved_date = session_date or _utc_now().date()
    return {
        "session_id": build_session_id(message, session_date=resolved_date),
        "session_date": resolved_date,
        "channel": message.channel,
        "user_id": message.user_id,
        "conversation_id": message.conversation_id,
        "message_history": json.dumps([event], default=str),
    }


@lru_cache
def ensure_schema() -> None:
    """Create the message_sessions table once per process when configured."""
    dsn = get_postgres_dsn()
    if not dsn:
        return
    psycopg = _load_psycopg()
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_MESSAGE_SESSIONS_TABLE_SQL)
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

    ensure_schema()
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
