"""Assistant orchestration for normalized messages."""

from __future__ import annotations

import logging

from app.core.models import Message, MessageResponse
from app.services import internal_documents, message_history
from app.services.openai_agent import generate_reply, generate_session_summary

logger = logging.getLogger("doppelganger.server")


async def handle_message(message: Message) -> MessageResponse:
    """Produce a plain-text doppelganger reply for one normalized inbound message."""
    logger.info(
        "status=received channel=%s user_id=%s conversation_id=%s message_id=%s text=%r metadata=%s",
        message.channel,
        message.user_id,
        message.conversation_id,
        message.message_id,
        message.text,
        message.metadata,
    )
    history_enabled = message_history.is_configured()
    current_session_history: list[dict] = []
    current_session_summary: str | None = None
    previous_session_summaries: list[str] = []
    retrieved_documents: list[dict] = []

    if history_enabled:
        try:
            await message_history.append_message_event_async(
                message=message,
                direction="inbound",
                text=message.text,
            )
            current_session_history = await message_history.get_current_session_history_async(message)
            current_session_summary = await message_history.get_current_session_summary_async(message)
            previous_session_summaries = await message_history.get_previous_session_summaries_async(
                message
            )
        except Exception:
            logger.exception(
                "status=history_load_failed channel=%s user_id=%s conversation_id=%s message_id=%s",
                message.channel,
                message.user_id,
                message.conversation_id,
                message.message_id,
            )

    if internal_documents.looks_like_knowledge_seeking_query(message):
        try:
            retrieved_documents = await internal_documents.retrieve_internal_document_context(message)
        except Exception:
            logger.exception(
                "status=internal_documents_retrieval_failed channel=%s user_id=%s conversation_id=%s message_id=%s",
                message.channel,
                message.user_id,
                message.conversation_id,
                message.message_id,
            )

    response_text = await generate_reply(
        message,
        current_session_history=current_session_history,
        current_session_summary=current_session_summary,
        previous_session_summaries=previous_session_summaries,
        retrieved_documents=retrieved_documents,
    )
    if history_enabled:
        try:
            await message_history.append_message_event_async(
                message=message,
                direction="outbound",
                text=response_text,
            )
            updated_session_history = await message_history.get_current_session_history_async(message)
            session_summary = await generate_session_summary(
                message,
                existing_session_summary=current_session_summary,
                current_session_history=updated_session_history,
            )
            if session_summary:
                await message_history.update_session_summary_async(message, session_summary)
        except Exception:
            logger.exception(
                "status=history_update_failed channel=%s user_id=%s conversation_id=%s message_id=%s",
                message.channel,
                message.user_id,
                message.conversation_id,
                message.message_id,
            )
    logger.info(
        "status=responded channel=%s user_id=%s conversation_id=%s message_id=%s reply=%r",
        message.channel,
        message.user_id,
        message.conversation_id,
        message.message_id,
        response_text,
    )
    return MessageResponse(reply_text=response_text)
