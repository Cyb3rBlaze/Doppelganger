# AGENTS.md

## Project Intent
- Build an AI doppelganger for a single user.
- Support Telegram first as a live inbound channel and keep Gmail available as a tool-backed capability.
- Run locally on a FastAPI server.

## Current Scope
- Accept inbound messages from API, terminal, and Telegram.
- Normalize incoming messages into one internal format.
- Pass messages through one doppelganger loop.
- Return a plain-text response through the originating channel.
- Store daily session history and rolling summaries in Postgres.
- Retrieve top internal documents from the pgvector store for knowledge-seeking queries and add them to reply context.
- Allow explicit Gmail read/send tool calls.
- Allow explicit internal-document search tool calls for deliberate note lookup.
- Allow constrained text file inspection and targeted-edit tool calls inside the project, especially for `mind/SOUL.md` and `mind/DIRECTIVES.md`.

## TODOs
- Inbound Gmail channel adapter.
- Additional external tools or side-effecting actions beyond the current Gmail tools.
- Long-term memory beyond daily-session summaries and internal-doc retrieval.
- Memory hierarchy

## Architecture Direction
- Keep one backend service.
- Keep channel adapters isolated from core doppelganger logic.
- Keep the doppelganger loop channel-agnostic.
- Prefer simple, inspectable components over abstractions.

## Suggested Layout
- `app/main.py`: FastAPI entrypoint
- `app/api/`: API and health endpoints
- `app/channels/`: terminal and Telegram adapters
- `app/core/`: message models and orchestration
- `app/services/`: LLM client, memory, and retrieval services
- `app/tools/`: external tool integrations like Gmail
- `design_docs/`: architecture and implementation notes

## Development Rules
- Keep the codebase small and readable.
- Add types and docstrings where they clarify boundaries.
- Avoid premature abstractions.
- Update `design_docs/` before expanding scope.
- Keep secrets in environment variables, never in code.

## Near-Term Goal
- Keep the current end-to-end loop stable while tightening Gmail, retrieval quality, and memory quality.
