"""Assistant orchestration for normalized messages."""

from app.core.models import MessageRequest, MessageResponse


async def handle_message(payload: MessageRequest) -> MessageResponse:
    """Produce a placeholder response for an incoming normalized message."""
    response_text = (
        f"Received your message from {payload.channel}. "
        "The assistant loop is not connected yet."
    )
    return MessageResponse(reply_text=response_text)
