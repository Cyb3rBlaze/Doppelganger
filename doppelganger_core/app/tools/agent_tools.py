"""OpenAI Agents SDK tool registration for the AI doppelganger."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.services import internal_documents
from app.tools import gmail_client


def build_agent_tools(function_tool: Callable[..., Any]) -> list[Any]:
    """Build the current set of custom tools available to the doppelganger."""

    @function_tool(
        name_override="send_gmail",
        description_override=(
            "Send a plain-text Gmail message on behalf of the user. "
            "Use this only when the user explicitly wants to send an email."
        ),
    )
    def send_gmail(
        to: list[str],
        subject: str,
        body_text: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        thread_id: str | None = None,
        from_email: str | None = None,
    ) -> dict[str, Any]:
        """Send a Gmail message with variable To, Cc, and Bcc recipients."""
        email = gmail_client.OutboundEmail(
            to=to,
            cc=cc or [],
            bcc=bcc or [],
            subject=subject,
            body_text=body_text,
            thread_id=thread_id,
            from_email=from_email,
        )
        result = gmail_client.send_gmail_message(email)
        return {
            "status": "sent",
            "id": result.get("id"),
            "threadId": result.get("threadId"),
            "labelIds": result.get("labelIds"),
        }

    @function_tool(
        name_override="read_gmail",
        description_override=(
            "Read recent Gmail messages for the user. "
            "Use this when the user explicitly wants to inspect, summarize, or reference email."
        ),
    )
    def read_gmail(
        query: str | None = None,
        max_results: int = 5,
    ) -> dict[str, Any]:
        """Read recent Gmail messages with an optional Gmail search query."""
        messages = gmail_client.read_gmail_messages(
            query=query,
            max_results=max_results,
        )
        return {
            "status": "ok",
            "messages": messages,
            "count": len(messages),
        }

    @function_tool(
        name_override="search_internal_documents",
        description_override=(
            "Search the user's internal notes and documents by semantic similarity. "
            "Use this when you deliberately want to inspect private notes, documents, "
            "or prior written material beyond the automatic lightweight retrieval."
        ),
    )
    def search_internal_documents(
        query: str,
        max_results: int = 3,
    ) -> dict[str, Any]:
        """Search the internal documents pgvector store with an explicit query."""
        documents = internal_documents.search_internal_documents_for_query(
            query,
            limit=max_results,
        )
        return {
            "status": "ok",
            "documents": documents,
            "count": len(documents),
        }

    return [send_gmail, read_gmail, search_internal_documents]
