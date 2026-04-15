# AGENTS.md

## Project Intent
- Build an AI doppelganger for a single user.
- Support Telegram and Gmail as the first channels.
- Run locally on a FastAPI server.

## V0 Scope
- Accept inbound messages from Telegram and Gmail.
- Normalize incoming messages into one internal format.
- Pass messages through one doppelganger loop.
- Return a plain-text response through the originating channel.

## TODOs
- Additional external tools or side-effecting actions beyond the current Gmail send tool.
- Long-term memory beyond what is required to answer within a conversation.
- Memory hierarchy

## Architecture Direction
- Keep one backend service.
- Keep channel adapters isolated from core doppelganger logic.
- Keep the doppelganger loop channel-agnostic.
- Prefer simple, inspectable components over abstractions.

## Suggested Layout
- `app/main.py`: FastAPI entrypoint
- `app/api/`: webhook and health endpoints
- `app/channels/`: Telegram and Gmail adapters
- `app/core/`: message models, orchestration, doppelganger loop
- `app/services/`: LLM client and shared services
- `design_docs/`: architecture and implementation notes

## Development Rules
- Keep the codebase small and readable.
- Add types and docstrings where they clarify boundaries.
- Avoid premature abstractions.
- Update `design_docs/` before expanding scope.
- Keep secrets in environment variables, never in code.

## Near-Term Goal
- Build a clean end-to-end loop: inbound message -> normalize -> doppelganger response -> outbound reply
