# AGENTS.md

## Project Intent
- Build a separate internal-documents pipeline inside this monorepo.
- Store internal documents and embeddings in PostgreSQL with pgvector.
- Keep this project mostly independent from the doppelganger runtime, while allowing the doppelganger to retrieve from its database.

## Step-By-Step Scope
- Start with database configuration and schema bootstrap.
- Add document discovery and whole-document persistence first.
- Add embedding production for one embedding per document.
- Add retrieval/query behavior after ingestion is stable.
- Revisit chunking only after the simple document-level baseline is solid.

## Architecture Direction
- Keep one small Python package in `core/`.
- Prefer direct SQL with psycopg over an ORM.
- Keep document parsing, embedding generation, and retrieval as separate layers.
- Make configuration explicit through environment variables.

## Storage Direction
- Use PostgreSQL as the source of truth.
- Use the pgvector extension for embedding storage and similarity search.
- Store one row per document with enough metadata to trace back to the source file.

## Development Rules
- Add tests for each slice before expanding scope.
- Keep schema helpers small and inspectable.
- Keep the current default simple: one embedding per document with `text-embedding-3-small` at `1536` dimensions unless the user chooses differently.
- Keep secrets in environment variables, never in code.
