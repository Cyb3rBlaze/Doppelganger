"""Tests for the Telegram polling adapter."""

from __future__ import annotations

import asyncio

import pytest
from app.channels import telegram
from app.core.models import MessageResponse
from tests.helpers import AsyncSpy


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class FakeAsyncClient:
    def __init__(self, response):
        self._response = response
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        self.posts.append((url, json))
        return self._response


def test_normalize_telegram_update_returns_message(sample_telegram_update_dict) -> None:
    update = telegram.TelegramUpdate.model_validate(sample_telegram_update_dict)

    message = telegram.normalize_telegram_update(update)

    assert message is not None
    assert message.channel == "telegram"
    assert message.user_id == "999"
    assert message.text == "hello there"
    assert message.conversation_id == "12345"
    assert message.message_id == "55"
    assert message.metadata["telegram_update_id"] == 101


def test_normalize_telegram_update_returns_none_for_non_text() -> None:
    update = telegram.TelegramUpdate.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 2,
                "chat": {"id": 3, "type": "private"},
            },
        }
    )

    assert telegram.normalize_telegram_update(update) is None


def test_build_send_message_payload_includes_reply_id_when_present() -> None:
    payload = telegram.build_send_message_payload(1, "hi", reply_to_message_id=2)
    assert payload["chat_id"] == 1
    assert payload["text"] == "hi"
    assert payload["reply_to_message_id"] == 2


def test_telegram_meta_extracts_useful_fields(sample_telegram_update_dict) -> None:
    update = telegram.TelegramUpdate.model_validate(sample_telegram_update_dict)
    meta = telegram._telegram_meta(update)
    assert meta["update_id"] == 101
    assert meta["chat_id"] == 12345
    assert meta["user_id"] == 999
    assert meta["username"] == "anshul"


def test_get_telegram_bot_token_raises_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        telegram.get_telegram_bot_token()


def test_build_telegram_api_url_uses_token(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    url = telegram.build_telegram_api_url("sendMessage")
    assert url == "https://api.telegram.org/botabc/sendMessage"


def test_get_telegram_allowed_user_ids_parses_csv(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "999, 12345 , ,777")
    assert telegram.get_telegram_allowed_user_ids() == {"999", "12345", "777"}


def test_is_telegram_user_allowed_returns_true_for_allowed_sender(
    monkeypatch,
    sample_telegram_update_dict,
) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "999")
    update = telegram.TelegramUpdate.model_validate(sample_telegram_update_dict)

    assert telegram.is_telegram_user_allowed(update) is True


def test_is_telegram_user_allowed_returns_false_for_disallowed_sender(
    monkeypatch,
    sample_telegram_update_dict,
) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "111")
    update = telegram.TelegramUpdate.model_validate(sample_telegram_update_dict)

    assert telegram.is_telegram_user_allowed(update) is False


@pytest.mark.asyncio
async def test_fetch_telegram_updates_returns_validated_updates(
    monkeypatch,
    sample_telegram_update_dict,
) -> None:
    fake_client = FakeAsyncClient(
        FakeResponse({"ok": True, "result": [sample_telegram_update_dict]})
    )
    monkeypatch.setattr("app.channels.telegram.httpx.AsyncClient", lambda *args, **kwargs: fake_client)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")

    updates = await telegram.fetch_telegram_updates(offset=10, timeout_seconds=5)

    assert len(updates) == 1
    assert updates[0].update_id == 101
    assert fake_client.posts[0][0] == "https://api.telegram.org/botabc/getUpdates"
    assert fake_client.posts[0][1]["offset"] == 10
    assert fake_client.posts[0][1]["timeout"] == 5


@pytest.mark.asyncio
async def test_send_telegram_reply_posts_payload(monkeypatch) -> None:
    fake_client = FakeAsyncClient(FakeResponse({"ok": True, "result": {"message_id": 999}}))
    monkeypatch.setattr("app.channels.telegram.httpx.AsyncClient", lambda *args, **kwargs: fake_client)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")

    result = await telegram.send_telegram_reply(123, "hello", reply_to_message_id=55)

    assert result["result"]["message_id"] == 999
    assert fake_client.posts[0][0] == "https://api.telegram.org/botabc/sendMessage"
    assert fake_client.posts[0][1]["chat_id"] == 123


@pytest.mark.asyncio
async def test_handle_telegram_update_ignores_unsupported_updates(monkeypatch) -> None:
    update = telegram.TelegramUpdate(update_id=1, message=None)
    mock_handle_message = AsyncSpy()
    mock_send = AsyncSpy()
    monkeypatch.setattr("app.channels.telegram.handle_message", mock_handle_message)
    monkeypatch.setattr("app.channels.telegram.send_telegram_reply", mock_send)

    await telegram.handle_telegram_update(update)

    mock_handle_message.assert_not_called()
    mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_telegram_update_ignores_unauthorized_user(
    monkeypatch,
    sample_telegram_update_dict,
) -> None:
    update = telegram.TelegramUpdate.model_validate(sample_telegram_update_dict)
    mock_handle_message = AsyncSpy()
    mock_send = AsyncSpy()
    monkeypatch.setattr("app.channels.telegram.handle_message", mock_handle_message)
    monkeypatch.setattr("app.channels.telegram.send_telegram_reply", mock_send)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "111")

    await telegram.handle_telegram_update(update)

    mock_handle_message.assert_not_called()
    mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_telegram_update_runs_core_loop_and_sends_reply(
    monkeypatch,
    sample_telegram_update_dict,
) -> None:
    update = telegram.TelegramUpdate.model_validate(sample_telegram_update_dict)
    mock_handle_message = AsyncSpy(result=MessageResponse(reply_text="hi there"))
    mock_send = AsyncSpy(result={"ok": True})
    monkeypatch.setattr("app.channels.telegram.handle_message", mock_handle_message)
    monkeypatch.setattr("app.channels.telegram.send_telegram_reply", mock_send)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "999")

    await telegram.handle_telegram_update(update)

    mock_handle_message.assert_awaited_once()
    mock_send.assert_awaited_once_with(
        chat_id=12345,
        text="hi there",
        reply_to_message_id=55,
    )


@pytest.mark.asyncio
async def test_run_polling_loop_logs_errors_and_retries(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.channels.telegram.fetch_telegram_updates",
        AsyncSpy(side_effect=[RuntimeError("boom"), asyncio.CancelledError()]),
    )
    mock_sleep = AsyncSpy()
    monkeypatch.setattr("app.channels.telegram.asyncio.sleep", mock_sleep)

    with pytest.raises(asyncio.CancelledError):
        await telegram.run_polling_loop(timeout_seconds=1)

    mock_sleep.assert_awaited_once()
