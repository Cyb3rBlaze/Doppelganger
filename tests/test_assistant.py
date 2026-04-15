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


async def test_handle_message_appends_inbound_and_outbound_history(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    appended_events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.core.assistant.generate_reply",
        AsyncSpy(result="hi there"),
    )
    monkeypatch.setattr("app.core.assistant.message_history.is_configured", lambda: True)

    async def fake_append_message_event_async(*, message, direction, text, metadata=None):
        _ = message, metadata
        appended_events.append((direction, text))
        return True

    monkeypatch.setattr(
        "app.core.assistant.message_history.append_message_event_async",
        fake_append_message_event_async,
    )

    response = await handle_message(message)

    assert response.reply_text == "hi there"
    assert appended_events == [
        ("inbound", "hello"),
        ("outbound", "hi there"),
    ]
