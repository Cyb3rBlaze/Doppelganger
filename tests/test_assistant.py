"""Tests for core doppelganger orchestration."""

from __future__ import annotations

import pytest

from app.core.assistant import handle_message
from app.core.models import Message
from tests.helpers import AsyncSpy


pytestmark = pytest.mark.asyncio


async def test_handle_message_returns_reply_from_generator(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    mock_generate_reply = AsyncSpy(result="hi there")
    monkeypatch.setattr("app.core.assistant.generate_reply", mock_generate_reply)

    response = await handle_message(message)

    mock_generate_reply.assert_awaited_once_with(message)
    assert response.reply_text == "hi there"


async def test_handle_message_logs_received_and_responded(monkeypatch, caplog) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    monkeypatch.setattr(
        "app.core.assistant.generate_reply",
        AsyncSpy(result="hi there"),
    )

    with caplog.at_level("INFO", logger="doppelganger.server"):
        await handle_message(message)

    joined = "\n".join(caplog.messages)
    assert "status=received" in joined
    assert "status=responded" in joined
