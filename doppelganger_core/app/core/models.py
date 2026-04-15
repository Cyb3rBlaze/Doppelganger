"""Shared request, response, and normalized message models."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class Message(BaseModel):
    """Channel-agnostic message passed through the doppelganger loop."""

    channel: str = Field(
        ...,
        min_length=1,
        description="Originating channel such as api, telegram, slack, or gmail.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="Stable identifier for the single user.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Normalized plain-text body from any inbound channel.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Channel-specific thread or conversation identifier.",
    )
    message_id: str | None = Field(
        default=None,
        description="Optional message identifier from the originating channel.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional channel-specific metadata that the core loop can ignore.",
    )


class MessageRequest(BaseModel):
    """Current HTTP payload for posting a normalized inbound message."""

    channel: str = Field(
        ...,
        min_length=1,
        description="Originating channel such as api, telegram, slack, or gmail.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="Stable identifier for the single user.",
    )
    message_text: str = Field(
        ...,
        min_length=1,
        description="Plain-text message body from the inbound channel.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Channel-specific thread or conversation identifier.",
    )
    message_id: str | None = Field(
        default=None,
        description="Optional message identifier from the originating channel.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional channel-specific metadata kept outside the core model.",
    )

    def to_message(self) -> Message:
        """Normalize the HTTP payload into the internal channel-agnostic model."""
        return Message(
            channel=self.channel,
            user_id=self.user_id,
            text=self.message_text,
            conversation_id=self.conversation_id,
            message_id=self.message_id,
            metadata=self.metadata,
        )


class MessageResponse(BaseModel):
    reply_text: str
