# AI Workspace

This repository is a small monorepo with three active subprojects:

- `doppelganger_core/`: the local AI doppelganger runtime
- `internal_documents_core/`: internal document ingestion into Postgres + pgvector
- `postgresql_viewer/`: a read-only web UI for browsing Postgres tables

## Doppelganger Core

Run the backend:

```bash
cd doppelganger_core
./start.sh api
```

Other entrypoints:

```bash
./start.sh terminal
./start.sh telegram
./start.sh gmail-auth
```

## Internal Documents Core

Install and ingest documents:

```bash
cd internal_documents_core
python -m pip install -e ".[test]"
./ingest.sh ingest "/Users/anshul/My Drive/Notes + Ideas"
```

Run vector search:

```bash
./ingest.sh search "notes about investing"
```

## PostgreSQL Viewer

Run the viewer:

```bash
cd postgresql_viewer
npm run dev
```
