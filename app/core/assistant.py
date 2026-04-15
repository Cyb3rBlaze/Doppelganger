"""Assistant orchestration for normalized messages."""

from __future__ import annotations

import logging

from app.core.models import Message, MessageResponse
from app.services import message_history
from app.services.openai_agent import generate_reply

logger = logging.getLogger("doppelganger.server")


async def handle_message(message: Message) -> MessageResponse:
    """Produce a plain-text doppelganger reply for one normalized inbound message."""
    logger.info(
        "status=received channel=%s user_id=%s conversation_id=%s message_id=%s text=%r metadata=%s",
        message.channel,
        message.user_id,
        message.conversation_id,
        message.message_id,
        message.text,
        message.metadata,
    )
    if message_history.is_configured():
        await message_history.append_message_event_async(
            message=message,
            direction="inbound",
            text=message.text,
        )
    response_text = await generate_reply(message)
    if message_history.is_configured():
        await message_history.append_message_event_async(
            message=message,
            direction="outbound",
            text=response_text,
        )
    logger.info(
        "status=responded channel=%s user_id=%s conversation_id=%s message_id=%s reply=%r",
        message.channel,
        message.user_id,
        message.conversation_id,
        message.message_id,
        response_text,
    )
    return MessageResponse(reply_text=response_text)
