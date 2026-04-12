# AGENTS.md

## Project Intent
- Build a personal assistant for a single user.
- Support Telegram and Gmail as the first channels.
- Run locally on a FastAPI server.

## V0 Scope
- Accept inbound messages from Telegram and Gmail.
- Normalize incoming messages into one internal format.
- Pass messages through one assistant loop.
- Return a plain-text response through the originating channel.

## TODOs
- External tools or side-effecting actions beyond replying.
- Long-term memory beyond what is required to answer within a conversation.
- Memory hierarchy

## Architecture Direction
- Keep one backend service.
- Keep channel adapters isolated from core assistant logic.
- Keep the assistant loop channel-agnostic.
- Prefer simple, inspectable components over abstractions.

## Suggested Layout
- `app/main.py`: FastAPI entrypoint
- `app/api/`: webhook and health endpoints
- `app/channels/`: Telegram and Gmail adapters
- `app/core/`: message models, orchestration, assistant loop
- `app/services/`: LLM client and shared services
- `design_docs/`: architecture and implementation notes

## Development Rules
- Keep the codebase small and readable.
- Add types and docstrings where they clarify boundaries.
- Avoid premature abstractions.
- Update `design_docs/` before expanding scope.
- Keep secrets in environment variables, never in code.

## Near-Term Goal
- Build a clean end-to-end loop: inbound message -> normalize -> assistant response -> outbound reply
