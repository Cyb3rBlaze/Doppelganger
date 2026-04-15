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
    monkeypatch.setattr("app.core.assistant.message_history.is_configured", lambda: False)
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.looks_like_knowledge_seeking_query",
        lambda message: False,
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.retrieve_internal_document_context",
        AsyncSpy(result=[]),
    )

    response = await handle_message(message)

    mock_generate_reply.assert_awaited_once_with(
        message,
        current_session_history=[],
        current_session_summary=None,
        previous_session_summaries=[],
        retrieved_documents=[],
    )
    assert response.reply_text == "hi there"


async def test_handle_message_logs_received_and_responded(monkeypatch, caplog) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    monkeypatch.setattr(
        "app.core.assistant.generate_reply",
        AsyncSpy(result="hi there"),
    )
    monkeypatch.setattr("app.core.assistant.message_history.is_configured", lambda: False)
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.looks_like_knowledge_seeking_query",
        lambda message: False,
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.retrieve_internal_document_context",
        AsyncSpy(result=[]),
    )

    with caplog.at_level("INFO", logger="doppelganger.server"):
        await handle_message(message)

    joined = "\n".join(caplog.messages)
    assert "status=received" in joined
    assert "status=responded" in joined


async def test_handle_message_skips_internal_document_retrieval_for_non_knowledge_query(
    monkeypatch,
) -> None:
    message = Message(channel="api", user_id="anshul", text="sounds good thanks")
    retrieval_spy = AsyncSpy(result=[{"title": "Should not be used"}])
    generate_reply_spy = AsyncSpy(result="hi there")

    monkeypatch.setattr("app.core.assistant.message_history.is_configured", lambda: False)
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.looks_like_knowledge_seeking_query",
        lambda message: False,
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.retrieve_internal_document_context",
        retrieval_spy,
    )
    monkeypatch.setattr("app.core.assistant.generate_reply", generate_reply_spy)

    response = await handle_message(message)

    assert response.reply_text == "hi there"
    retrieval_spy.assert_not_called()
    generate_reply_spy.assert_awaited_once_with(
        message,
        current_session_history=[],
        current_session_summary=None,
        previous_session_summaries=[],
        retrieved_documents=[],
    )


async def test_handle_message_appends_inbound_and_outbound_history(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    appended_events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.core.assistant.generate_reply",
        AsyncSpy(result="hi there"),
    )
    monkeypatch.setattr(
        "app.core.assistant.generate_session_summary",
        AsyncSpy(result="summary text"),
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

    async def fake_get_current_session_history_async(message):
        _ = message
        return [{"direction": "inbound", "text": "hello"}]

    async def fake_get_current_session_summary_async(message):
        _ = message
        return "summary so far"

    async def fake_get_previous_session_summaries_async(message):
        _ = message
        return []

    monkeypatch.setattr(
        "app.core.assistant.message_history.get_current_session_history_async",
        fake_get_current_session_history_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_current_session_summary_async",
        fake_get_current_session_summary_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_previous_session_summaries_async",
        fake_get_previous_session_summaries_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.update_session_summary_async",
        AsyncSpy(result=True),
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.looks_like_knowledge_seeking_query",
        lambda message: False,
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.retrieve_internal_document_context",
        AsyncSpy(result=[]),
    )

    response = await handle_message(message)

    assert response.reply_text == "hi there"
    assert appended_events == [
        ("inbound", "hello"),
        ("outbound", "hi there"),
    ]


async def test_handle_message_passes_current_session_history_into_generate_reply(
    monkeypatch,
) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    captured_kwargs = {}
    monkeypatch.setattr("app.core.assistant.message_history.is_configured", lambda: True)

    async def fake_append_message_event_async(*, message, direction, text, metadata=None):
        _ = message, direction, text, metadata
        return True

    async def fake_get_current_session_history_async(message):
        _ = message
        return [{"direction": "inbound", "text": "hello"}]

    async def fake_get_current_session_summary_async(message):
        _ = message
        return "current summary"

    async def fake_get_previous_session_summaries_async(message):
        _ = message
        return ["Prior session summary"]

    async def fake_generate_reply(message, **kwargs):
        _ = message
        captured_kwargs.update(kwargs)
        return "hi there"

    async def fake_generate_session_summary(message, **kwargs):
        _ = message, kwargs
        return "summary text"

    monkeypatch.setattr(
        "app.core.assistant.message_history.append_message_event_async",
        fake_append_message_event_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_current_session_history_async",
        fake_get_current_session_history_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_current_session_summary_async",
        fake_get_current_session_summary_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_previous_session_summaries_async",
        fake_get_previous_session_summaries_async,
    )
    monkeypatch.setattr("app.core.assistant.generate_reply", fake_generate_reply)
    monkeypatch.setattr(
        "app.core.assistant.generate_session_summary",
        fake_generate_session_summary,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.update_session_summary_async",
        AsyncSpy(result=True),
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.looks_like_knowledge_seeking_query",
        lambda message: True,
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.retrieve_internal_document_context",
        AsyncSpy(
            result=[
                {
                    "title": "Investing Notes",
                    "source_path": "/docs/investing.gdoc",
                    "score": 0.91,
                    "content": "Key investing principles",
                }
            ]
        ),
    )

    response = await handle_message(message)

    assert response.reply_text == "hi there"
    assert captured_kwargs["current_session_history"] == [
        {"direction": "inbound", "text": "hello"}
    ]
    assert captured_kwargs["current_session_summary"] == "current summary"
    assert captured_kwargs["previous_session_summaries"] == ["Prior session summary"]
    assert captured_kwargs["retrieved_documents"] == [
        {
            "title": "Investing Notes",
            "source_path": "/docs/investing.gdoc",
            "score": 0.91,
            "content": "Key investing principles",
        }
    ]


async def test_handle_message_updates_session_summary_after_reply(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    recorded_events: list[tuple[str, str]] = []
    update_summary_spy = AsyncSpy(result=True)
    generate_session_summary_spy = AsyncSpy(result="fresh summary")

    monkeypatch.setattr("app.core.assistant.message_history.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.core.assistant.generate_reply",
        AsyncSpy(result="hi there"),
    )
    monkeypatch.setattr(
        "app.core.assistant.generate_session_summary",
        generate_session_summary_spy,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.update_session_summary_async",
        update_summary_spy,
    )

    async def fake_append_message_event_async(*, message, direction, text, metadata=None):
        _ = message, metadata
        recorded_events.append((direction, text))
        return True

    async def fake_get_current_session_history_async(message):
        _ = message
        if len(recorded_events) == 1:
            return [{"direction": "inbound", "text": "hello"}]
        return [
            {"direction": "inbound", "text": "hello"},
            {"direction": "outbound", "text": "hi there"},
        ]

    async def fake_get_previous_session_summaries_async(message):
        _ = message
        return []

    async def fake_get_current_session_summary_async(message):
        _ = message
        return "summary so far"

    monkeypatch.setattr(
        "app.core.assistant.message_history.append_message_event_async",
        fake_append_message_event_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_current_session_history_async",
        fake_get_current_session_history_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_current_session_summary_async",
        fake_get_current_session_summary_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_previous_session_summaries_async",
        fake_get_previous_session_summaries_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.looks_like_knowledge_seeking_query",
        lambda message: False,
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.retrieve_internal_document_context",
        AsyncSpy(result=[]),
    )

    response = await handle_message(message)

    assert response.reply_text == "hi there"
    generate_session_summary_spy.assert_awaited_once_with(
        message,
        existing_session_summary="summary so far",
        current_session_history=[
            {"direction": "inbound", "text": "hello"},
            {"direction": "outbound", "text": "hi there"},
        ],
    )
    update_summary_spy.assert_awaited_once_with(message, "fresh summary")


async def test_handle_message_still_returns_reply_when_history_load_fails(
    monkeypatch,
    caplog,
) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    generate_reply_spy = AsyncSpy(result="hi there")

    monkeypatch.setattr("app.core.assistant.message_history.is_configured", lambda: True)
    monkeypatch.setattr("app.core.assistant.generate_reply", generate_reply_spy)
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.looks_like_knowledge_seeking_query",
        lambda message: False,
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.retrieve_internal_document_context",
        AsyncSpy(result=[]),
    )

    async def fake_append_message_event_async(*, message, direction, text, metadata=None):
        _ = message, direction, text, metadata
        raise RuntimeError("db down")

    monkeypatch.setattr(
        "app.core.assistant.message_history.append_message_event_async",
        fake_append_message_event_async,
    )

    with caplog.at_level("ERROR", logger="doppelganger.server"):
        response = await handle_message(message)

    assert response.reply_text == "hi there"
    generate_reply_spy.assert_awaited_once_with(
        message,
        current_session_history=[],
        current_session_summary=None,
        previous_session_summaries=[],
        retrieved_documents=[],
    )
    assert "status=history_load_failed" in "\n".join(caplog.messages)


async def test_handle_message_still_returns_reply_when_history_update_fails(
    monkeypatch,
    caplog,
) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")

    monkeypatch.setattr("app.core.assistant.message_history.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.core.assistant.generate_reply",
        AsyncSpy(result="hi there"),
    )
    monkeypatch.setattr(
        "app.core.assistant.internal_documents.looks_like_knowledge_seeking_query",
        lambda message: False,
    )
    monkeypatch.setattr(
        "app.core.assistant.generate_session_summary",
        AsyncSpy(result="fresh summary"),
    )

    recorded_events: list[tuple[str, str]] = []

    async def fake_append_message_event_async(*, message, direction, text, metadata=None):
        _ = message, metadata
        recorded_events.append((direction, text))
        return True

    async def fake_get_current_session_history_async(message):
        _ = message
        return [{"direction": "inbound", "text": "hello"}]

    async def fake_get_current_session_summary_async(message):
        _ = message
        return "summary so far"

    async def fake_get_previous_session_summaries_async(message):
        _ = message
        return []

    async def fake_update_session_summary_async(message, summary):
        _ = message, summary
        raise RuntimeError("db write failed")

    monkeypatch.setattr(
        "app.core.assistant.message_history.append_message_event_async",
        fake_append_message_event_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_current_session_history_async",
        fake_get_current_session_history_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_current_session_summary_async",
        fake_get_current_session_summary_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.get_previous_session_summaries_async",
        fake_get_previous_session_summaries_async,
    )
    monkeypatch.setattr(
        "app.core.assistant.message_history.update_session_summary_async",
        fake_update_session_summary_async,
    )

    with caplog.at_level("ERROR", logger="doppelganger.server"):
        response = await handle_message(message)

    assert response.reply_text == "hi there"
    assert "status=history_update_failed" in "\n".join(caplog.messages)
