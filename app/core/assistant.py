"""Assistant orchestration for normalized messages."""

from app.core.models import Message, MessageResponse
from app.services.openai_agent import generate_reply


async def handle_message(message: Message) -> MessageResponse:
    """Produce a plain-text doppelganger reply for one normalized inbound message."""
    response_text = await generate_reply(message)
    return MessageResponse(reply_text=response_text)
