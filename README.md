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
- `app/channels/telegram.py` is a stub adapter that normalizes Telegram webhook updates and builds outbound reply payloads

This keeps the doppelganger loop as the internal interface and HTTP as an external transport boundary.

## Local setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e .
```

3. Fill in `.env` with your OpenAI API key.
4. Start the server:

```bash
uvicorn app.main:app --reload
```

5. Open the FastAPI docs at `http://127.0.0.1:8000/docs`.

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

## Notes

- The doppelganger loop is channel-agnostic.
- The doppelganger is intentionally a single specialist for now.
- This is the right base for adding Telegram and Gmail adapters next.
