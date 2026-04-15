# V0 Agent Loop

## Goal

Keep the doppelganger loop inspectable even as more context sources are layered in:

`inbound message -> normalize -> gather context -> doppelganger run -> plain-text reply`

## Current Decisions

- Keep one internal `Message` model in `app/core/models.py`.
- Let channel-specific payloads normalize into `Message` before the doppelganger loop.
- Keep orchestration in `app/core/assistant.py`.
- Keep the OpenAI Agents SDK integration isolated in `app/services/openai_agent.py`.
- Keep retrieval and persistence in separate services instead of baking them into channel adapters.

## Current Context Sources

The reply path now combines:

- the current inbound message
- previous session summaries
- the current session summary
- recent current-session message history
- top retrieved internal documents from the pgvector store

## Why This Shape

- It keeps the doppelganger loop channel-agnostic.
- It avoids mixing Telegram, Gmail, or internal-doc implementation details into the reply boundary.
- It keeps the OpenAI-specific integration small enough to replace or expand later.
- It keeps retrieval and memory best-effort instead of making them transport concerns.

## Current Gaps

- Gmail is available as tools, but not yet as a live inbound adapter.
- Internal-doc retrieval is still one embedding per document, not chunk-level retrieval.
- Memory is still shallow relative to a full hierarchy.
