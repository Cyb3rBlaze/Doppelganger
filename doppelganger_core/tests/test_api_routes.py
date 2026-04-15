"""Tests for FastAPI routes."""

from __future__ import annotations

from app.core.models import MessageResponse
from tests.helpers import AsyncSpy


def test_health_route_returns_ok(app_client) -> None:
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_handle_message_route_normalizes_and_returns_reply(app_client, monkeypatch) -> None:
    mock_handle_message = AsyncSpy(result=MessageResponse(reply_text="hello back"))
    monkeypatch.setattr("app.api.routes.handle_message", mock_handle_message)

    response = app_client.post(
        "/messages/handle",
        json={
            "channel": "api",
            "user_id": "anshul",
            "message_text": "hello",
            "conversation_id": "thread-1",
            "message_id": "msg-1",
            "metadata": {"source": "test"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"reply_text": "hello back"}
    mock_handle_message.assert_awaited_once()
    normalized_message = mock_handle_message.await_args.args[0]
    assert normalized_message.channel == "api"
    assert normalized_message.text == "hello"
    assert normalized_message.conversation_id == "thread-1"


def test_handle_message_route_rejects_invalid_payload(app_client) -> None:
    response = app_client.post(
        "/messages/handle",
        json={
            "channel": "api",
            "user_id": "anshul",
            "message_text": "",
        },
    )

    assert response.status_code == 422
