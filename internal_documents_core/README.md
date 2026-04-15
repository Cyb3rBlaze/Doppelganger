# Internal Documents Core

This project ingests internal documents into PostgreSQL + pgvector and exposes vector search over adaptive document chunks/windows.

## Current Capabilities

- walk a local source directory recursively
- load `.gdoc`, `.md`, and `.txt`
- fetch real Google Doc text from local `.gdoc` pointer files through Google Drive export
- split documents into character-budget base chunks with newline-preferred boundaries
- compare neighboring chunk/window similarity and store adaptive sliding windows
- store document chunks in Postgres with pgvector
- run cosine-similarity search over stored chunks
- show tqdm-style progress during ingest
- skip problematic documents and write them to a plaintext report
- write a JSON report showing which chunk/window steps were merged and their similarity scores

## Current Storage Shape

The `document_chunks` table stores:

- `document_id`
- `chunk_id`
- `source_path`
- `source_kind`
- `title`
- `content`
- `metadata`
- `chunk_index`
- `window_start_chunk_index`
- `window_end_chunk_index`
- `embedding`
- timestamps

Rows now represent adaptive chunk windows. Every embedding payload includes the document title plus compact metadata so same-document chunks share some common semantic framing.

## Chunking and Windowing

The current ingest pipeline works like this:

- split each document into base chunks by character count
- prefer newline boundaries when ending a chunk, then sentence punctuation, then whitespace
- include the document title and compact metadata in every embedding payload
- embed the first base chunk as the first stored window
- for each next base chunk, compare its embedding to the current window embedding
- if similarity is low, start a fresh window
- if similarity is high, merge it into the current window and re-embed the combined window
- if the combined window gets too large, pop oldest base chunks from the front so the window slides forward

This means stored rows are adaptive windows over ordered base chunks rather than one row per whole document.

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
- splits each document into character-budget base chunks and stores adaptive sliding windows
- retries transient Google export failures
- skips documents that fail to load
- skips documents that exceed the embedding context length

Skipped documents are written to:

```text
internal_documents_core/skipped_documents.txt
```

Adaptive chunk/window decisions are written to:

```text
internal_documents_core/chunk_merge_report.json
```

## Notes

- Supported content sources right now are `.gdoc`, `.md`, and `.txt`.
- Local `.gdoc` files are only Google Drive pointer files, so real content is fetched through Google Drive export.
- Unsupported files such as `.gslides`, `.gsheet`, images, and diagrams are currently ignored.
- Retrieval currently returns top matching chunk/window rows rather than grouped whole-document answers.
