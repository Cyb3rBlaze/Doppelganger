# Gmail Integration Plan

## Goal

Add a Gmail channel adapter that can:

- read inbound Gmail messages for the single allowed user
- normalize those messages into the shared internal `Message` model
- send outbound Gmail messages with variable `to`, `cc`, and `bcc` recipients
- keep Gmail-specific logic isolated from the core doppelganger loop

## V0 Shape

For the first Gmail slice, we will support:

- local OAuth for one Gmail account
- listing and reading messages from that account
- sending plain-text email with `to`, `cc`, and `bcc`
- turning inbound Gmail content into the shared `Message`
- keeping reply generation channel-agnostic

We will not support yet:

- attachments
- HTML rendering beyond a plain-text first pass
- push notifications or Gmail watch subscriptions
- mailbox state changes like labeling or marking messages read
- multi-account support

## OAuth Decision

Use the Gmail API with a local desktop OAuth flow for one user.

Why:

- this project is local-first
- the repo is currently single-user
- it avoids introducing a public callback/webhook requirement on day one
- it matches the current Telegram adapter style: one local process owns one channel

## Minimum Scopes

Use the narrowest scopes needed for the first slice:

- `https://www.googleapis.com/auth/gmail.readonly`
- `https://www.googleapis.com/auth/gmail.send`

These are enough for:

- listing and fetching messages
- sending email to `To`, `Cc`, and `Bcc` recipients via MIME headers

We should avoid `gmail.modify` unless we intentionally add mailbox mutations later.

## Proposed Files

- `app/channels/gmail.py`
  Gmail adapter entrypoints and normalization helpers
- `app/tools/gmail_client.py`
  OAuth, Gmail API client creation, low-level list/get/send helpers
- `tests/test_gmail_client.py`
  unit tests for auth helpers, client setup, payload construction
- `tests/test_gmail.py`
  adapter tests for normalization, read flow, send flow, and guardrails

## Data Boundaries

### Shared core model

We will keep using `app.core.models.Message` for the core loop.

### Gmail-specific models

The Gmail adapter will likely need a few Gmail-only models/helpers:

- a parsed Gmail sender/recipient representation
- a compact inbound email structure before normalization
- an outbound email request structure with:
  - `to: list[str]`
  - `cc: list[str]`
  - `bcc: list[str]`
  - `subject: str`
  - `body_text: str`
  - optional `thread_id`

These should stay in the Gmail adapter/service layer and not leak into `app/core/`.

## Implementation Phases

### Phase 1: Auth and client bootstrap

Build:

- environment-driven paths for Gmail credentials and token storage
- a helper to construct the Gmail API client
- a local OAuth token bootstrap/refresh flow

Expected environment:

- `GMAIL_OAUTH_CLIENT_SECRET_PATH`
- `GMAIL_OAUTH_TOKEN_PATH`

Behavior:

- if a valid token exists, reuse it
- if the token is expired and refreshable, refresh it
- otherwise run the local OAuth installed-app flow

Tests:

- missing credentials path raises clearly
- existing valid token is reused
- expired refreshable token refreshes
- missing token falls back to OAuth flow

### Phase 2: Outbound send

Build:

- a helper to compose a plain-text MIME email
- support for variable `to`, `cc`, and `bcc`
- base64url encoding into Gmail `messages.send` format

Behavior:

- `To`, `Cc`, and `Bcc` are optional lists, but at least one recipient must be present
- `bcc` must be included in the MIME headers so Gmail sends it correctly
- use `userId="me"`

Tests:

- MIME payload contains subject and all recipient headers
- encoded payload is generated correctly
- send helper calls `users().messages().send(userId="me", body=...)`
- sending with no recipients raises validation error

### Phase 3: Inbound read/list

Build:

- a helper to list messages, probably filtered to inbox/unread first
- a helper to fetch one Gmail message in a readable format
- text extraction for the first plain-text pass

Behavior:

- prefer plain-text body when present
- fall back carefully when a message body is nested in MIME parts
- preserve Gmail metadata such as:
  - message id
  - thread id
  - sender
  - subject
  - internal date

Tests:

- list helper parses message ids from Gmail API response
- read helper extracts subject/from/body/thread id
- nested plain-text MIME part is handled
- empty/unsupported bodies are ignored safely

### Phase 4: Gmail adapter

Build:

- normalize inbound Gmail content into `Message`
- optionally expose adapter-level functions like:
  - `fetch_gmail_messages()`
  - `read_gmail_message(...)`
  - `normalize_gmail_message(...)`
  - `send_gmail_message(...)`

Behavior:

- adapter stays channel-specific
- core loop still only sees `Message`
- Gmail metadata is stored in `message.metadata`

Tests:

- normalization maps Gmail message -> `Message`
- outbound send path does not call the core loop
- inbound processing path calls the core loop once

## Guardrails

We should keep Gmail as locked down as Telegram.

Planned guardrails:

- only operate on the configured Gmail account tied to the OAuth token
- fail closed on missing auth files
- do not send if all recipient lists are empty
- do not silently mutate mailbox state

## Recommended First Coding Slice

Start with outbound send first, then inbound read.

Why:

- it is smaller than inbound parsing
- `to` / `cc` / `bcc` support is clearly bounded
- the MIME composition logic is easy to test thoroughly
- once send works, reply flows become much simpler

## First Deliverable

The first implementation step should produce:

- `app/tools/gmail_client.py`
- tests for Gmail auth bootstrap
- tests for MIME composition and send payload generation
- no live network dependency in the default test suite
