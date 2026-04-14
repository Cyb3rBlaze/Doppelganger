"""OpenAI Agents SDK tool registration for the AI doppelganger."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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

    return [send_gmail]
