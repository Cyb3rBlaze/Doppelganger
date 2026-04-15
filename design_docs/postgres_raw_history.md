# Postgres Raw History

## Goal

Add the first persistence layer for conversation history and session tracking using Postgres.

This first slice stores raw message events only. It does not yet feed prior messages back into model context.

## Scope

For now we want:

- one Postgres table
- one row per raw inbound or outbound message
- a derived session identifier that rolls over daily
- enough fields to reconstruct the raw thread later

We do not want yet:

- memory summarization
- semantic retrieval
- multi-table abstractions
- ORM-heavy modeling

## Table Shape

Use one table named `message_sessions`.

Suggested columns:

- `session_id text primary key`
- `session_date date not null`
- `channel text not null`
- `user_id text not null`
- `conversation_id text null`
- `message_history jsonb not null default '[]'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

## Session Rule

Create a new session per day.

For this first slice, a session is derived from:

- `channel`
- `user_id`
- `conversation_id` if present, otherwise `"unknown"`
- current UTC date

That produces a stable daily session key like:

`telegram:6891176979:6891176979:2026-04-14`

This gives us:

- a clear reset every day
- thread grouping inside a channel
- no extra session table yet

## Write Rule

Append both:

- the inbound user message before the model runs
- the outbound doppelganger reply after the model returns

Each `message_history` item should be a dict like:

- `direction`
- `text`
- `message_id`
- `metadata`
- `created_at`

## Environment

Add:

- `POSTGRES_DSN`

Example:

`POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/doppelganger`

## Bootstrap

Keep bootstrap simple:

- one helper to create the table if it does not exist
- one helper to append a message event into the session row

No migration framework yet.

## Integration Point

Persist in `app/core/assistant.py`.

Why:

- every normalized channel already passes through that boundary
- it gives us one place to capture inbound and outbound raw events
- it stays channel-agnostic

## Tests

Add tests for:

- daily session id generation
- insert payload construction
- assistant storing inbound and outbound events in the right order
- graceful behavior when Postgres is not configured

## Next Step After This

Once raw persistence is stable, the next step is reading recent session history back out and using it as model context.
