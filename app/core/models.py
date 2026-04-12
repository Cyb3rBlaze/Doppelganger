"""Shared request and response models."""

from typing import Literal

from pydantic import BaseModel, Field


Channel = Literal["telegram", "gmail"]


class HealthResponse(BaseModel):
    status: str = "ok"


class MessageRequest(BaseModel):
    channel: Channel
    user_id: str = Field(
        ..., description="Stable identifier for the single user."
    )
    message_text: str = Field(..., min_length=1)
    conversation_id: str | None = Field(
        default=None,
        description="Channel-specific thread or conversation identifier.",
    )


class MessageResponse(BaseModel):
    reply_text: str
