"""Terminal adapter for talking to the doppelganger locally."""

from __future__ import annotations

import asyncio

from app.core.assistant import handle_message
from app.core.models import Message

DEFAULT_TERMINAL_CHANNEL = "terminal"
DEFAULT_TERMINAL_USER_ID = "local-user"
DEFAULT_TERMINAL_CONVERSATION_ID = "terminal-session"


def build_terminal_message(
    text: str,
    *,
    user_id: str = DEFAULT_TERMINAL_USER_ID,
    conversation_id: str = DEFAULT_TERMINAL_CONVERSATION_ID,
) -> Message:
    """Normalize terminal input into the shared channel-agnostic message model."""
    return Message(
        channel=DEFAULT_TERMINAL_CHANNEL,
        user_id=user_id,
        text=text,
        conversation_id=conversation_id,
    )


async def send_terminal_message(
    text: str,
    *,
    user_id: str = DEFAULT_TERMINAL_USER_ID,
    conversation_id: str = DEFAULT_TERMINAL_CONVERSATION_ID,
) -> str:
    """Send one terminal message directly through the core doppelganger loop."""
    message = build_terminal_message(
        text,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    response = await handle_message(message)
    return response.reply_text


async def run_terminal_session() -> None:
    """Run a tiny local REPL that talks directly to the doppelganger loop."""
    print("Terminal doppelganger session started. Type 'exit' to quit.")
    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("bye")
            return

        reply_text = await send_terminal_message(user_text)
        print(f"doppelganger> {reply_text}")


def main() -> None:
    """Run the terminal adapter."""
    asyncio.run(run_terminal_session())


if __name__ == "__main__":
    main()
