# Internal Documents Core

This project ingests internal documents into PostgreSQL + pgvector and exposes simple vector search over one embedding per document.

## Current Capabilities

- walk a local source directory recursively
- load `.gdoc`, `.md`, and `.txt`
- fetch real Google Doc text from local `.gdoc` pointer files through Google Drive export
- create one embedding per full document
- store documents in Postgres with pgvector
- run cosine-similarity search over stored documents
- show tqdm-style progress during ingest
- skip problematic documents and write them to a plaintext report

## Current Storage Shape

The `documents` table stores:

- `document_id`
- `source_path`
- `source_kind`
- `title`
- `content`
- `metadata`
- `embedding`
- timestamps

This is intentionally one row per document, not chunked retrieval yet.

## Environment

The ingester looks for `.env` in this directory first and then falls back to `../doppelganger_core/.env`.

```dotenv
INTERNAL_DOCUMENTS_POSTGRES_DSN=postgresql://localhost:5432/internal_documents
INTERNAL_DOCUMENTS_SOURCE_DIR=./documents
INTERNAL_DOCUMENTS_EMBEDDING_MODEL=text-embedding-3-small
INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION=1536
INTERNAL_DOCUMENTS_GOOGLE_OAUTH_CLIENT_SECRET_PATH=oauth_secret.json
INTERNAL_DOCUMENTS_GOOGLE_OAUTH_TOKEN_PATH=.internal_documents_google_token.json
```

Notes:

- the current default embedding model is `text-embedding-3-small`
- the current default dimension is `1536`
- if `INTERNAL_DOCUMENTS_GOOGLE_OAUTH_TOKEN_PATH` is empty or unset, the project falls back to `.internal_documents_google_token.json`

## Install

```bash
python -m pip install -e ".[test]"
```

## Run

Ingest:

```bash
./ingest.sh ingest "/Users/anshul/My Drive/Notes + Ideas"
```

Search:

```bash
./ingest.sh search "notes about investing"
```

You can also call the module directly:

```bash
python -m core.ingest ingest --source-dir "/Users/anshul/My Drive/Notes + Ideas"
python -m core.ingest search "notes about investing"
```

## Ingest Behavior

During ingest the CLI now:

- creates the target database if it does not exist yet
- ensures the pgvector schema exists
- shows tqdm-style progress in the terminal
- retries transient Google export failures
- skips documents that fail to load
- skips documents that exceed the embedding context length

Skipped documents are written to:

```text
internal_documents_core/skipped_documents.txt
```

## Notes

- Supported content sources right now are `.gdoc`, `.md`, and `.txt`.
- Local `.gdoc` files are only Google Drive pointer files, so real content is fetched through Google Drive export.
- Unsupported files such as `.gslides`, `.gsheet`, images, and diagrams are currently ignored.
- This first slice is optimized for simplicity, not chunk quality.
