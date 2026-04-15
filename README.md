# AI Workspace

This repo now has three top-level cores:

- `doppelganger_core/`: the AI doppelganger app
- `internal_documents_core/`: internal document ingestion and vector DB population
- `postgresql_viewer/`: web app to easily visualize postgres table contents

## Run The Doppelganger

```bash
cd doppelganger_core
./start.sh api
```

## Internal Documents Core

```bash
cd internal_documents_core
python -m pip install -e ".[test]"
python -m internal_documents_core.ingest --help
```

## Run The PostgreSQL Viewer

```bash
cd postgresql_viewer
npm run dev
```
