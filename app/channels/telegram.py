"""Telegram adapter helpers for normalizing updates and sending replies."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.models import Message

TELEGRAM_CHANNEL = "telegram"


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
    """Subset of Telegram webhook update fields needed by the adapter."""

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
    """Build the payload you would send to Telegram's sendMessage endpoint."""
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
    }
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
    return payload
