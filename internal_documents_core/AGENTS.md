# AGENTS.md

## Project Intent
- Build a separate internal-documents pipeline inside this monorepo.
- Store internal documents and embeddings in PostgreSQL with pgvector.
- Keep this project mostly independent from the doppelganger runtime, while allowing the doppelganger to retrieve from its database.

## Step-By-Step Scope
- Start with database configuration and schema bootstrap.
- Add document discovery and whole-document persistence first.
- Keep storage in chunk/window rows.
- Produce adaptive chunk/window embeddings with similarity-based merging over character-budget base chunks.
- Attach graph-style relations between final chunk/window nodes and store them on each row.
- Add retrieval/query behavior after ingestion is stable.

## Architecture Direction
- Keep one small Python package in `core/`.
- Prefer direct SQL with psycopg over an ORM.
- Keep document parsing, embedding generation, and retrieval as separate layers.
- Make configuration explicit through environment variables.

## Storage Direction
- Use PostgreSQL as the source of truth.
- Use the pgvector extension for embedding storage and similarity search.
- Store one row per chunk/window with enough document metadata to trace back to the source file, plus `connected_nodes` graph metadata.

## Development Rules
- Add tests for each slice before expanding scope.
- Keep schema helpers small and inspectable.
- Keep chunking logic inspectable and deterministic enough to unit test without live API calls.
- Prefer character-budget chunking with newline-aware boundaries because most source material is bullet-note style.
- Keep secrets in environment variables, never in code.
