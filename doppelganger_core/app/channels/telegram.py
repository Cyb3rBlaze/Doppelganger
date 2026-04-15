"""Telegram long-polling adapter for the AI doppelganger."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.assistant import handle_message
from app.core.models import Message
from app.logging_utils import configure_logging

TELEGRAM_CHANNEL = "telegram"
TELEGRAM_API_BASE_URL = "https://api.telegram.org"
TELEGRAM_ALLOWED_USER_IDS_ENV = "TELEGRAM_ALLOWED_USER_IDS"
DEFAULT_POLL_TIMEOUT_SECONDS = 30
DEFAULT_ERROR_BACKOFF_SECONDS = 5

configure_logging()

logger = logging.getLogger("doppelganger.telegram")


class TelegramUser(BaseModel):
    """Subset of Telegram user fields needed by the adapter."""

    id: int
    username: str | None = None
    first_name: str | None = None


class TelegramChat(BaseModel):
    """Subset of Telegram chat fields needed by the adapter."""

    id: int
    type: str
    title: str | None = None


class TelegramMessage(BaseModel):
    """Subset of Telegram message fields needed by the adapter."""

    message_id: int
    date: int | None = None
    text: str | None = None
    caption: str | None = None
    from_user: TelegramUser | None = Field(default=None, alias="from")
    chat: TelegramChat

    model_config = {"populate_by_name": True}

    @property
    def body_text(self) -> str | None:
        """Return the best plain-text body available on the update."""
        return self.text or self.caption


class TelegramUpdate(BaseModel):
    """Subset of Telegram update fields needed by the adapter."""

    update_id: int
    message: TelegramMessage | None = None


def normalize_telegram_update(update: TelegramUpdate) -> Message | None:
    """Convert a Telegram update into the shared internal message model."""
    telegram_message = update.message
    if telegram_message is None or telegram_message.body_text is None:
        return None

    sender = telegram_message.from_user
    user_id = str(sender.id) if sender is not None else str(telegram_message.chat.id)

    metadata: dict[str, Any] = {
        "telegram_update_id": update.update_id,
        "telegram_chat_id": telegram_message.chat.id,
        "telegram_chat_type": telegram_message.chat.type,
    }
    if sender is not None and sender.username:
        metadata["telegram_username"] = sender.username
    if telegram_message.chat.title:
        metadata["telegram_chat_title"] = telegram_message.chat.title

    return Message(
        channel=TELEGRAM_CHANNEL,
        user_id=user_id,
        text=telegram_message.body_text,
        conversation_id=str(telegram_message.chat.id),
        message_id=str(telegram_message.message_id),
        metadata=metadata,
    )


def build_send_message_payload(
    chat_id: int,
    text: str,
    *,
    reply_to_message_id: int | None = None,
) -> dict[str, Any]:
    """Build the payload for Telegram's sendMessage endpoint."""
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
    }
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
    return payload


def _telegram_meta(update: TelegramUpdate) -> dict[str, Any]:
    """Collect compact update metadata for logs."""
    telegram_message = update.message
    sender = telegram_message.from_user if telegram_message is not None else None
    chat = telegram_message.chat if telegram_message is not None else None
    return {
        "update_id": update.update_id,
        "message_id": getattr(telegram_message, "message_id", None),
        "chat_id": getattr(chat, "id", None),
        "chat_type": getattr(chat, "type", None),
        "chat_title": getattr(chat, "title", None),
        "user_id": getattr(sender, "id", None),
        "username": getattr(sender, "username", None),
        "first_name": getattr(sender, "first_name", None),
    }


def get_telegram_bot_token() -> str:
    """Return the configured Telegram bot token."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")
    return token


def get_telegram_allowed_user_ids() -> set[str]:
    """Return the configured Telegram user IDs allowed to receive replies."""
    raw_value = os.getenv(TELEGRAM_ALLOWED_USER_IDS_ENV, "")
    return {
        user_id.strip()
        for user_id in raw_value.split(",")
        if user_id.strip()
    }


def is_telegram_user_allowed(update: TelegramUpdate) -> bool:
    """Check whether the Telegram sender is allowed to use the bot."""
    allowed_user_ids = get_telegram_allowed_user_ids()
    telegram_message = update.message
    sender = telegram_message.from_user if telegram_message is not None else None
    sender_id = str(sender.id) if sender is not None else None
    return sender_id is not None and sender_id in allowed_user_ids


def build_telegram_api_url(method: str) -> str:
    """Build the Telegram Bot API URL for a method name."""
    token = get_telegram_bot_token()
    return f"{TELEGRAM_API_BASE_URL}/bot{token}/{method}"


async def fetch_telegram_updates(
    *,
    offset: int | None = None,
    timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> list[TelegramUpdate]:
    """Long-poll Telegram for new updates."""
    payload: dict[str, Any] = {
        "timeout": timeout_seconds,
        "allowed_updates": ["message"],
    }
    if offset is not None:
        payload["offset"] = offset

    logger.info(
        "status=polling queue=waiting channel=%s timeout_seconds=%s offset=%s",
        TELEGRAM_CHANNEL,
        timeout_seconds,
        offset,
    )
    async with httpx.AsyncClient(timeout=timeout_seconds + 10) as client:
        response = await client.post(
            build_telegram_api_url("getUpdates"),
            json=payload,
        )
    response.raise_for_status()
    data = response.json()
    if not data.get("ok", False):
        raise RuntimeError(f"Telegram getUpdates failed: {data}")
    updates = [TelegramUpdate.model_validate(item) for item in data.get("result", [])]
    logger.info(
        "status=polled queue=received channel=%s updates_count=%s next_offset_candidate=%s",
        TELEGRAM_CHANNEL,
        len(updates),
        updates[-1].update_id + 1 if updates else offset,
    )
    return updates


async def send_telegram_reply(
    chat_id: int,
    text: str,
    *,
    reply_to_message_id: int | None = None,
) -> dict[str, Any]:
    """Send a plain-text reply back to Telegram."""
    payload = build_send_message_payload(
        chat_id,
        text,
        reply_to_message_id=reply_to_message_id,
    )
    logger.info(
        "status=reply_sending channel=%s chat_id=%s reply_to_message_id=%s text=%r",
        TELEGRAM_CHANNEL,
        chat_id,
        reply_to_message_id,
        text,
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            build_telegram_api_url("sendMessage"),
            json=payload,
        )
    response.raise_for_status()
    data = response.json()
    if not data.get("ok", False):
        raise RuntimeError(f"Telegram sendMessage failed: {data}")
    logger.info(
        "status=reply_sent channel=%s chat_id=%s sent_message_id=%s",
        TELEGRAM_CHANNEL,
        chat_id,
        data.get("result", {}).get("message_id"),
    )
    return data


async def handle_telegram_update(update: TelegramUpdate) -> None:
    """Run one Telegram update through the doppelganger loop and reply."""
    meta = _telegram_meta(update)
    logger.info(
        "status=update_received queue=queued channel=%s date=%s meta=%s text=%r",
        TELEGRAM_CHANNEL,
        update.message.date if update.message and hasattr(update.message, "date") else None,
        meta,
        update.message.body_text if update.message else None,
    )
    message = normalize_telegram_update(update)
    if message is None or update.message is None:
        logger.info(
            "status=update_ignored queue=done channel=%s meta=%s reason=%s",
            TELEGRAM_CHANNEL,
            meta,
            "non_text_or_unsupported",
        )
        return

    if not is_telegram_user_allowed(update):
        logger.info(
            "status=update_ignored queue=done channel=%s meta=%s reason=%s",
            TELEGRAM_CHANNEL,
            meta,
            "unauthorized_user",
        )
        return

    logger.info(
        "status=processing queue=dequeued channel=%s user_id=%s conversation_id=%s message_id=%s",
        message.channel,
        message.user_id,
        message.conversation_id,
        message.message_id,
    )
    reply = await handle_message(message)
    await send_telegram_reply(
        chat_id=update.message.chat.id,
        text=reply.reply_text,
        reply_to_message_id=update.message.message_id,
    )
    logger.info(
        "status=processed queue=done channel=%s user_id=%s conversation_id=%s message_id=%s",
        message.channel,
        message.user_id,
        message.conversation_id,
        message.message_id,
    )


async def run_polling_loop(
    *,
    timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> None:
    """Run the Telegram adapter with long polling."""
    logger.info(
        "status=started channel=%s poll_timeout_seconds=%s error_backoff_seconds=%s",
        TELEGRAM_CHANNEL,
        timeout_seconds,
        DEFAULT_ERROR_BACKOFF_SECONDS,
    )
    next_offset: int | None = None

    while True:
        try:
            updates = await fetch_telegram_updates(
                offset=next_offset,
                timeout_seconds=timeout_seconds,
            )
            for update in updates:
                next_offset = update.update_id + 1
                try:
                    await handle_telegram_update(update)
                except Exception:
                    logger.exception(
                        "status=update_failed channel=%s update_id=%s next_offset=%s",
                        TELEGRAM_CHANNEL,
                        update.update_id,
                        next_offset,
                    )
        except Exception as exc:
            logger.info(
                "status=error channel=%s next_offset=%s error=%r",
                TELEGRAM_CHANNEL,
                next_offset,
                exc,
            )
            await asyncio.sleep(DEFAULT_ERROR_BACKOFF_SECONDS)


def main() -> None:
    """Run the Telegram long-polling adapter."""
    asyncio.run(run_polling_loop())


if __name__ == "__main__":
    main()
