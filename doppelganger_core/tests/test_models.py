"""Tests for shared core models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.models import HealthResponse, Message, MessageRequest, MessageResponse


def test_health_response_defaults_to_ok() -> None:
    response = HealthResponse()
    assert response.status == "ok"


def test_message_request_normalizes_to_message() -> None:
    request = MessageRequest(
        channel="telegram",
        user_id="anshul",
        message_text="hello",
        conversation_id="thread-1",
        message_id="msg-1",
        metadata={"source": "test"},
    )

    message = request.to_message()

    assert message == Message(
        channel="telegram",
        user_id="anshul",
        text="hello",
        conversation_id="thread-1",
        message_id="msg-1",
        metadata={"source": "test"},
    )


def test_message_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        Message(channel="api", user_id="anshul", text="")


def test_message_request_requires_non_empty_channel() -> None:
    with pytest.raises(ValidationError):
        MessageRequest(channel="", user_id="anshul", message_text="hello")


def test_message_response_holds_reply_text() -> None:
    response = MessageResponse(reply_text="hi")
    assert response.reply_text == "hi"
