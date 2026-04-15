"""OpenAI Agents SDK integration for the first doppelganger loop."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.core.models import Message
from app.logging_utils import configure_logging
from app.tools.agent_tools import build_agent_tools

DEFAULT_ASSISTANT_NAME = "Personal Doppelganger"
DEFAULT_ASSISTANT_MODEL = "gpt-5.4"
DEFAULT_SUMMARY_AGENT_NAME = "Session Summarizer"
DEFAULT_REPLY_HISTORY_EVENT_LIMIT = 10
DEFAULT_SUMMARY_HISTORY_EVENT_LIMIT = 5
MIND_DIR = Path(__file__).resolve().parents[2] / "mind"
MIND_FILES = ("SOUL.md", "DIRECTIVES.md")
DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(DOTENV_PATH)
configure_logging()

logger = logging.getLogger("doppelganger.server.agent")


def _load_agents_sdk() -> tuple[Any, Any, Any]:
    """Import the OpenAI Agents SDK lazily."""
    try:
        from agents import Agent, Runner, function_tool
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Agents SDK is not installed. Run `pip install -e .` first."
        ) from exc
    return Agent, Runner, function_tool


@lru_cache
def load_mind_instructions() -> str:
    """Load the doppelganger prompt from markdown files in the mind directory."""
    sections: list[str] = []
    for filename in MIND_FILES:
        path = MIND_DIR / filename
        if not path.exists():
            raise RuntimeError(
                f"Missing mind file: {path}. Create the required markdown files first."
            )
        sections.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(section for section in sections if section)


@lru_cache
def get_agent() -> Any:
    """Build and cache the single-agent runtime for the current process."""
    Agent, _, function_tool = _load_agents_sdk()
    name = os.getenv("ASSISTANT_NAME", DEFAULT_ASSISTANT_NAME)
    model = os.getenv("ASSISTANT_MODEL", DEFAULT_ASSISTANT_MODEL)
    return Agent(
        name=name,
        instructions=load_mind_instructions(),
        model=model,
        tools=build_agent_tools(function_tool),
    )


@lru_cache
def get_summary_agent() -> Any:
    """Build and cache a lightweight agent for session summarization."""
    Agent, _, _ = _load_agents_sdk()
    model = os.getenv("ASSISTANT_MODEL", DEFAULT_ASSISTANT_MODEL)
    return Agent(
        name=DEFAULT_SUMMARY_AGENT_NAME,
        instructions=(
            "You summarize one session of conversation for an AI doppelganger. "
            "Write a concise factual summary of the important context, preferences, "
            "plans, commitments, people, and open loops. "
            "Do not invent facts. Keep it under 180 words. "
            "Return plain text only."
        ),
        model=model,
        tools=[],
    )


def build_agent_input(
    message: Message,
    *,
    current_session_history: list[dict[str, Any]] | None = None,
    current_session_summary: str | None = None,
    previous_session_summaries: list[str] | None = None,
) -> str:
    """Render a normalized message into a compact text input for the agent."""
    conversation_id = message.conversation_id or "unknown"
    message_id = message.message_id or "unknown"
    sections = [
        "You are replying to one inbound message.\n"
        f"Channel: {message.channel}\n"
        f"User ID: {message.user_id}\n"
        f"Conversation ID: {conversation_id}\n"
        f"Message ID: {message_id}\n"
        f"Message:\n{message.text}"
    ]

    if previous_session_summaries:
        summary_lines = "\n".join(f"- {summary}" for summary in previous_session_summaries)
        sections.append(f"Relevant summaries from previous sessions:\n{summary_lines}")

    if current_session_summary:
        sections.append(f"Current session summary so far:\n{current_session_summary}")

    history_for_context = _select_reply_context_history(
        message,
        current_session_history=current_session_history,
    )
    if history_for_context:
        history_lines = []
        for event in history_for_context:
            direction = event.get("direction", "unknown")
            event_text = event.get("text", "")
            history_lines.append(f"- [{direction}] {event_text}")
        sections.append("Recent current session history:\n" + "\n".join(history_lines))

    return "\n\n".join(sections)


def build_session_summary_input(
    message: Message,
    *,
    existing_session_summary: str | None = None,
    current_session_history: list[dict[str, Any]] | None = None,
) -> str:
    """Render the current session history into a summary prompt."""
    conversation_id = message.conversation_id or "unknown"
    history_lines: list[str] = []
    recent_events = _select_summary_context_history(current_session_history=current_session_history)
    for event in recent_events:
        direction = event.get("direction", "unknown")
        event_text = event.get("text", "")
        history_lines.append(f"- [{direction}] {event_text}")

    sections = [
        "Summarize this daily conversation session for future context.\n"
        f"Channel: {message.channel}\n"
        f"User ID: {message.user_id}\n"
        f"Conversation ID: {conversation_id}\n"
    ]
    if existing_session_summary:
        sections.append(f"Existing session summary:\n{existing_session_summary}")
    history_block = "\n".join(history_lines) if history_lines else "- [unknown] No new messages."
    sections.append("Recent session updates:\n" + history_block)
    return "\n\n".join(sections)


def _select_reply_context_history(
    message: Message,
    *,
    current_session_history: list[dict[str, Any]] | None = None,
    limit: int = DEFAULT_REPLY_HISTORY_EVENT_LIMIT,
) -> list[dict[str, Any]]:
    """Trim current session history for reply context and avoid duplicating the latest inbound."""
    if not current_session_history:
        return []

    trimmed_history = list(current_session_history[-limit:])
    if not trimmed_history:
        return trimmed_history

    last_event = trimmed_history[-1]
    if (
        last_event.get("direction") == "inbound"
        and last_event.get("text") == message.text
        and str(last_event.get("message_id") or "unknown") == (message.message_id or "unknown")
    ):
        return trimmed_history[:-1]
    return trimmed_history


def _select_summary_context_history(
    *,
    current_session_history: list[dict[str, Any]] | None = None,
    limit: int = DEFAULT_SUMMARY_HISTORY_EVENT_LIMIT,
) -> list[dict[str, Any]]:
    """Trim session history for rolling summary refreshes."""
    if not current_session_history:
        return []
    return list(current_session_history[-limit:])


def _truncate(value: Any, limit: int = 300) -> str:
    """Render a compact string for logs."""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str)
        except TypeError:
            value = str(value)
    return value if len(value) <= limit else f"{value[:limit]}..."


def _extract_tool_call_fields(item: Any) -> tuple[str, str, str]:
    """Extract the most useful tool call fields for logging."""
    raw_item = getattr(item, "raw_item", item)
    tool_name = getattr(raw_item, "name", None) or getattr(item, "title", None) or "unknown"
    call_id = getattr(raw_item, "call_id", None) or getattr(raw_item, "id", None) or "unknown"
    arguments = getattr(raw_item, "arguments", None)
    return str(tool_name), str(call_id), _truncate(arguments)


def _extract_tool_output_fields(item: Any) -> tuple[str, str]:
    """Extract compact tool output details for logging."""
    raw_item = getattr(item, "raw_item", item)
    call_id = getattr(raw_item, "call_id", None) or getattr(raw_item, "id", None) or "unknown"
    output = getattr(item, "output", None) or getattr(raw_item, "output", None)
    return str(call_id), _truncate(output)


def _extract_reasoning_text(item: Any) -> str:
    """Extract a short reasoning summary when present."""
    raw_item = getattr(item, "raw_item", item)
    summary = getattr(raw_item, "summary", None)
    if isinstance(summary, list):
        parts: list[str] = []
        for entry in summary:
            text = getattr(entry, "text", None)
            if text:
                parts.append(text)
        if parts:
            return _truncate(" ".join(parts))
    return _truncate(getattr(raw_item, "content", None) or getattr(raw_item, "text", None))


def log_stream_event(event: Any, *, message: Message) -> None:
    """Log streamed agent events relevant to reasoning and tool usage."""
    event_type = getattr(event, "type", None)
    if event_type == "agent_updated_stream_event":
        new_agent = getattr(event, "new_agent", None)
        logger.info(
            "status=agent_updated channel=%s user_id=%s agent=%s",
            message.channel,
            message.user_id,
            getattr(new_agent, "name", "unknown"),
        )
        return

    if event_type != "run_item_stream_event":
        return

    event_name = getattr(event, "name", "unknown")
    item = getattr(event, "item", None)

    if event_name == "reasoning_item_created":
        logger.info(
            "status=reasoning channel=%s user_id=%s conversation_id=%s message_id=%s summary=%r",
            message.channel,
            message.user_id,
            message.conversation_id,
            message.message_id,
            _extract_reasoning_text(item),
        )
        return

    if event_name == "tool_called":
        tool_name, call_id, arguments = _extract_tool_call_fields(item)
        logger.info(
            "status=tool_called channel=%s user_id=%s conversation_id=%s message_id=%s tool=%s call_id=%s arguments=%r",
            message.channel,
            message.user_id,
            message.conversation_id,
            message.message_id,
            tool_name,
            call_id,
            arguments,
        )
        return

    if event_name == "tool_output":
        call_id, output = _extract_tool_output_fields(item)
        logger.info(
            "status=tool_output channel=%s user_id=%s conversation_id=%s message_id=%s call_id=%s output=%r",
            message.channel,
            message.user_id,
            message.conversation_id,
            message.message_id,
            call_id,
            output,
        )
        return

    if event_name == "message_output_created":
        logger.info(
            "status=model_message_created channel=%s user_id=%s conversation_id=%s message_id=%s",
            message.channel,
            message.user_id,
            message.conversation_id,
            message.message_id,
        )


async def generate_reply(
    message: Message,
    *,
    current_session_history: list[dict[str, Any]] | None = None,
    current_session_summary: str | None = None,
    previous_session_summaries: list[str] | None = None,
) -> str:
    """Run one OpenAI Agents SDK turn and return plain text for the caller."""
    _, Runner, _ = _load_agents_sdk()
    result = Runner.run_streamed(
        get_agent(),
        build_agent_input(
            message,
            current_session_history=current_session_history,
            current_session_summary=current_session_summary,
            previous_session_summaries=previous_session_summaries,
        ),
    )
    async for event in result.stream_events():
        log_stream_event(event, message=message)
    final_output = result.final_output
    if isinstance(final_output, str):
        return final_output
    return str(final_output)


async def generate_session_summary(
    message: Message,
    *,
    existing_session_summary: str | None = None,
    current_session_history: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a concise summary for the current daily session."""
    _, Runner, _ = _load_agents_sdk()
    result = Runner.run_streamed(
        get_summary_agent(),
        build_session_summary_input(
            message,
            existing_session_summary=existing_session_summary,
            current_session_history=current_session_history,
        ),
    )
    async for event in result.stream_events():
        log_stream_event(event, message=message)
    final_output = result.final_output
    if isinstance(final_output, str):
        return final_output.strip()
    return str(final_output).strip()
