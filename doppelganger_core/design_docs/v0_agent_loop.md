# V0 Agent Loop

## Goal

Build the smallest inspectable end-to-end doppelganger loop:

`inbound message -> normalize -> doppelganger run -> plain-text reply`

## Decisions

- Keep one internal `Message` model in `app/core/models.py`.
- Let channel-specific payloads normalize into `Message` before the doppelganger loop.
- Keep orchestration in `app/core/assistant.py`.
- Keep the OpenAI Agents SDK integration isolated in `app/services/openai_agent.py`.
- Run one SDK turn per inbound message and return `final_output` as plain text.

## Why this shape

- It keeps the doppelganger loop channel-agnostic.
- It avoids mixing Telegram or Gmail details into the doppelganger logic.
- It keeps the OpenAI-specific integration small enough to replace or expand later.

## Next likely steps

- Add channel adapters under `app/channels/` that normalize raw input into `Message`.
- Add a Telegram adapter that converts updates into `Message`.
- Add a Gmail adapter that converts email threads into `Message`.
- Add conversation state once one-turn replies feel solid.
