# AI Doppelganger

This project is a local, single-user AI doppelganger. The current system supports a normalized message loop, Telegram polling, Gmail tools, Postgres-backed session memory, and retrieval from the internal documents pgvector database.

## Current Capabilities

- normalize inbound messages into one shared `Message` model
- reply through one OpenAI Agents SDK loop
- run via API, terminal, or Telegram polling
- store daily message sessions in Postgres
- maintain a rolling summary for each daily session
- retrieve the top 5 internal chunk/window seed matches by cosine similarity, expand 2 hops across graph edges, and inject that neighborhood into the reply context
- expose Gmail read/send as explicit agent tools
- expose a deliberate internal-document retrieval tool for deeper note lookups
- expose constrained file-inspection and targeted-edit tools for project text files, including the doppelganger mind files

## Current Shape

- `app/api/`: FastAPI routes
- `app/channels/`: terminal and Telegram adapters
- `app/core/`: message model and orchestration
- `app/services/openai_agent.py`: agent loop, prompt building, and summary generation
- `app/services/message_history.py`: Postgres-backed session history
- `app/services/internal_documents.py`: internal document retrieval for prompt context
- `app/tools/agent_tools.py`: Gmail tool registration
- `mind/`: `SOUL.md` and `DIRECTIVES.md`

## What Is Live Right Now

- API endpoint via FastAPI
- terminal adapter
- Telegram long polling adapter
- Gmail OAuth bootstrap and Gmail read/send tools
- Postgres session memory
- internal-doc retrieval from the `internal_documents` pgvector database

Not live yet:

- inbound Gmail channel adapter
- richer memory hierarchy beyond daily session summaries

## Environment Variables

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

INTERNAL_DOCUMENTS_POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/internal_documents
INTERNAL_DOCUMENTS_EMBEDDING_MODEL=text-embedding-3-small
INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION=1536
```

Notes:

- `POSTGRES_DSN` controls the doppelganger's session-memory database.
- `INTERNAL_DOCUMENTS_POSTGRES_DSN` points at the separate pgvector database used for retrieval.
- Telegram replies are restricted to `TELEGRAM_ALLOWED_USER_IDS`.
- Gmail tools use local OAuth and act on the account bound to the saved token.

## Setup

Install dependencies:

```bash
python -m pip install -e .
python -m pip install -e ".[test]"
```

Start the API:

```bash
./start.sh api
```

Other launch targets:

```bash
./start.sh terminal
./start.sh telegram
./start.sh gmail-auth
```

## Memory and Retrieval

When `POSTGRES_DSN` is set, the app stores one row per local-date session in `message_sessions` and appends both inbound and outbound messages to `message_history`.

The reply prompt currently includes:

- previous session summaries
- current session summary
- recent current-session message history
- the top 5 retrieved internal chunk/window seed matches from the pgvector store plus a 2-hop graph expansion through `connected_nodes` when the message looks knowledge-seeking

After each reply, the app refreshes the current session summary and stores it back on the same row.

## Gmail Tools

The agent has two Gmail tools:

- `read_gmail`
- `send_gmail`

These are tool calls, not a full Gmail inbound channel yet. They are intended for explicit user requests to inspect or send email.

## Internal Document Retrieval

The doppelganger now has two retrieval modes for internal documents:

- automatic gated retrieval for knowledge-seeking prompts
- explicit tool-driven retrieval through `search_internal_documents`

The explicit tool is for deliberate note/document lookup when the agent wants to search the internal-doc database on purpose instead of relying only on the lightweight automatic context injection. Retrieval currently returns chunk/window rows rather than grouped whole-document answers, and those rows now include `connected_nodes` graph metadata.

## File Tools

The agent now also has:

- `read_file`
- `get_file_info`
- `read_file_window`
- `search_in_file`
- `write_file`
- `replace_in_file`

These are constrained to safe text-like files inside `doppelganger_core/`. They are intended mainly for deliberate self-editing of files like:

- `mind/SOUL.md`
- `mind/DIRECTIVES.md`

The recommended edit flow is:

- inspect file size and hash with `get_file_info`
- read targeted windows with `read_file_window`
- locate exact sections with `search_in_file`
- make exact block replacements with `replace_in_file`

Blocked targets include secrets, token files, and paths outside the project root.

## Test the API

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

## Run Tests

```bash
python -m pytest
```
