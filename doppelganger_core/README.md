# AI Doppelganger

This project is the start of a local, single-user AI doppelganger. The current slice is intentionally small:

- accept one normalized inbound message
- run one OpenAI Agents SDK turn
- return one plain-text reply

## Current shape

- `app/api/` exposes the FastAPI routes
- `app/channels/` holds channel adapters for terminal, Telegram, and later Gmail
- `app/core/models.py` defines the channel-agnostic `Message` model
- `app/core/assistant.py` orchestrates one inbound message through the doppelganger loop
- `app/services/openai_agent.py` contains the OpenAI Agents SDK integration
- `mind/` stores the doppelganger mind as markdown files like `SOUL.md` and `DIRECTIVES.md`

## Message model

The core loop now uses one internal `Message` model regardless of where the text came from. That gives us a stable contract for future adapters like Telegram, Gmail, Slack, or a plain API endpoint.

Example normalized message:

```json
{
  "channel": "telegram",
  "user_id": "anshul",
  "text": "Draft a short reply to this email.",
  "conversation_id": "thread-123",
  "message_id": "msg-456",
  "metadata": {
    "from": "me@example.com"
  }
}
```

## Mind

The doppelganger prompt now lives in versioned markdown files under `mind/`.

- `mind/SOUL.md` holds the core identity and inner voice
- `mind/DIRECTIVES.md` holds the operating rules and response style

`app/services/openai_agent.py` loads those files and combines them into the final doppelganger instructions at runtime.

## Channel adapters

Channel adapters should normalize raw channel input into `app.core.models.Message` and then call the core loop directly.

- `app/channels/terminal.py` is a working local adapter that sends terminal input straight into `handle_message(...)`
- `app/channels/telegram.py` runs a Telegram long-polling loop and sends replies back with `sendMessage`

This keeps the doppelganger loop as the internal interface and HTTP as an external transport boundary.

## Local setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e .
```

For tests, install the optional test extras:

```bash
pip install -e ".[test]"
```

3. Fill in `.env` with your OpenAI API key.
4. Start the server:

```bash
uvicorn app.main:app --reload
```

5. Open the FastAPI docs at `http://127.0.0.1:8000/docs`.

You can also use the repo launcher:

```bash
./start.sh api
```

## Environment Variables

Current `.env` variables:

```dotenv
OPENAI_API_KEY=your-openai-api-key
ASSISTANT_MODEL=gpt-5.4
ASSISTANT_NAME=Personal Doppelganger

TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_ALLOWED_USER_IDS=123456789

GMAIL_OAUTH_CLIENT_SECRET_PATH=oauth_secret.json
GMAIL_OAUTH_TOKEN_PATH=.gmail_token.json
GMAIL_ALLOWED_SENDER_DOMAINS=

POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/doppelganger
```

Notes:

- `OPENAI_API_KEY` is required for the doppelganger loop.
- `TELEGRAM_ALLOWED_USER_IDS` is a comma-separated list of Telegram numeric user IDs allowed to receive replies.
- `GMAIL_OAUTH_CLIENT_SECRET_PATH` should point to your Google OAuth desktop client JSON file.
- `GMAIL_OAUTH_TOKEN_PATH` is where the local Gmail OAuth token cache will be stored after first login.
- `GMAIL_ALLOWED_SENDER_DOMAINS` is reserved for Gmail inbound guardrails and can stay empty for now.
- `POSTGRES_DSN` enables raw conversation history persistence in Postgres.

## Raw History

When `POSTGRES_DSN` is set, the app stores raw conversation history in Postgres.

Current shape:

- one row per daily session
- session key derived from `channel`, `user_id`, `conversation_id`, and the local server date
- rolling `session_summary` stored per session
- `message_history` stored as a JSONB list of dicts
- inbound and outbound events appended in order

The agent now uses:
- recent current-session history
- the current session summary so far
- recent prior-session summaries

After each reply, the app regenerates the current session summary and stores it on the same daily row.
When a new daily session starts, recent prior-session summaries are included as cross-session context.

## Test the first loop

Send a request to the local API:

```bash
curl -X POST http://127.0.0.1:8000/messages/handle \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "api",
    "user_id": "anshul",
    "message_text": "Help me plan my day.",
    "conversation_id": "local-test"
  }'
```

If `OPENAI_API_KEY` is set, the request runs one OpenAI Agents SDK turn and returns the final plain-text answer. If the key is missing, the API returns a clear setup message instead.

## Telegram setup

The first Telegram integration slice uses long polling. There are no webhooks yet.

1. Add your bot token to `.env`:

```dotenv
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_ALLOWED_USER_IDS=123456789,987654321
```

2. Install dependencies:

```bash
python -m pip install -e .
```

3. Start the Telegram adapter:

```bash
./start.sh telegram
```

4. Send your bot a plain-text message in Telegram.

The polling adapter uses Telegram `getUpdates` and waits up to 30 seconds per poll before checking again.
It only responds to Telegram senders whose numeric user IDs are listed in `TELEGRAM_ALLOWED_USER_IDS`.

## Gmail setup

The first Gmail slice now includes:

- local OAuth bootstrap for one Gmail account
- Gmail API client construction
- plain-text outbound sending with variable `to`, `cc`, and `bcc`

Add these to `.env`:

```dotenv
GMAIL_OAUTH_CLIENT_SECRET_PATH=oauth_secret.json
GMAIL_OAUTH_TOKEN_PATH=.gmail_token.json
GMAIL_ALLOWED_SENDER_DOMAINS=
```

When Gmail auth is used for the first time, the local OAuth flow opens a browser window and saves the token JSON to `GMAIL_OAUTH_TOKEN_PATH`.

You can also bootstrap or refresh Gmail auth directly:

```bash
./start.sh gmail-auth
```

The Gmail adapter itself is the next step. Right now the Gmail service layer is ready for auth and sending.

## Run tests

Run the suite with pytest:

```bash
python -m pytest
```

## Launcher

Use the repo-level launcher to start the current entrypoints:

```bash
./start.sh api
./start.sh terminal
./start.sh telegram
./start.sh gmail-auth
```

## Notes

- The doppelganger loop is channel-agnostic.
- The doppelganger is intentionally a single specialist for now.
- This is the right base for adding Telegram and Gmail adapters next.
