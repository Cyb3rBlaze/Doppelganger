"""OpenAI Agents SDK integration for the first doppelganger loop."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.core.models import Message

DEFAULT_ASSISTANT_NAME = "Personal Doppelganger"
DEFAULT_ASSISTANT_MODEL = "gpt-5.4"
MIND_DIR = Path(__file__).resolve().parents[2] / "mind"
MIND_FILES = ("SOUL.md", "DIRECTIVES.md")
DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(DOTENV_PATH)


def _load_agents_sdk() -> tuple[Any, Any]:
    """Import the OpenAI Agents SDK lazily."""
    try:
        from agents import Agent, Runner
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Agents SDK is not installed. Run `pip install -e .` first."
        ) from exc
    return Agent, Runner


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
    Agent, _ = _load_agents_sdk()
    name = os.getenv("ASSISTANT_NAME", DEFAULT_ASSISTANT_NAME)
    model = os.getenv("ASSISTANT_MODEL", DEFAULT_ASSISTANT_MODEL)
    return Agent(
        name=name,
        instructions=load_mind_instructions(),
        model=model,
    )


def build_agent_input(message: Message) -> str:
    """Render a normalized message into a compact text input for the agent."""
    conversation_id = message.conversation_id or "unknown"
    message_id = message.message_id or "unknown"
    return (
        "You are replying to one inbound message.\n"
        f"Channel: {message.channel}\n"
        f"User ID: {message.user_id}\n"
        f"Conversation ID: {conversation_id}\n"
        f"Message ID: {message_id}\n"
        f"Message:\n{message.text}"
    )


async def generate_reply(message: Message) -> str:
    """Run one OpenAI Agents SDK turn and return plain text for the caller."""
    _, Runner = _load_agents_sdk()
    result = await Runner.run(get_agent(), build_agent_input(message))
    final_output = result.final_output
    if isinstance(final_output, str):
        return final_output
    return str(final_output)
