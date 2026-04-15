"""Tests for the terminal adapter."""

from __future__ import annotations

import pytest

from app.channels import terminal
from app.core.models import MessageResponse
from tests.helpers import AsyncSpy


def test_build_terminal_message_uses_defaults() -> None:
    message = terminal.build_terminal_message("hello")
    assert message.channel == terminal.DEFAULT_TERMINAL_CHANNEL
    assert message.user_id == terminal.DEFAULT_TERMINAL_USER_ID
    assert message.conversation_id == terminal.DEFAULT_TERMINAL_CONVERSATION_ID
    assert message.text == "hello"


@pytest.mark.asyncio
async def test_send_terminal_message_calls_handle_message(monkeypatch) -> None:
    mock_handle_message = AsyncSpy(result=MessageResponse(reply_text="reply"))
    monkeypatch.setattr("app.channels.terminal.handle_message", mock_handle_message)

    reply = await terminal.send_terminal_message("hello")

    assert reply == "reply"
    mock_handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_terminal_session_exits_on_exit(monkeypatch) -> None:
    printed: list[str] = []
    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("builtins.print", lambda value: printed.append(str(value)))

    await terminal.run_terminal_session()

    joined = " ".join(printed)
    assert "Terminal doppelganger session started" in joined
    assert "bye" in joined


@pytest.mark.asyncio
async def test_run_terminal_session_prints_reply(monkeypatch) -> None:
    printed: list[str] = []
    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(
        "app.channels.terminal.send_terminal_message",
        AsyncSpy(result="hi"),
    )
    monkeypatch.setattr("builtins.print", lambda value: printed.append(str(value)))

    await terminal.run_terminal_session()

    joined = " ".join(printed)
    assert "doppelganger> hi" in joined
