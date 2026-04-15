# AGENTS.md

## Monorepo Intent
- Keep this repository as a small monorepo with clearly separated subprojects.
- Avoid duplicating one app at the repo root and again inside a subdirectory.
- Keep project-specific instructions inside each subproject when possible.

## Current Subprojects
- `doppelganger_core/`: the AI doppelganger backend and channel adapters
- `internal_documents_core/`: internal document ingestion and vector store population
- `postgresql_viewer/`: UI for inspecting Postgres data

## Root Rules
- The repo root should only contain monorepo-level files plus the three subprojects.
- Do not add app code directly at the root.
- Keep generated artifacts out of the repository root and out of subproject trees.
- Update the root `README.md` and root `.gitignore` when subproject structure changes.
- Keep each subproject's own `README.md` and `AGENTS.md` aligned with the real implementation state.
