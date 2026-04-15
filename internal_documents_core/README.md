# Internal Documents Core

This package is the start of a separate internal-documents pipeline. Its first slice is a simple OpenAI Vector Stores ingester that:

- walks a local source directory
- collects supported document files
- creates or reuses a vector store
- uploads files and polls until ingestion completes

## Environment

The ingester looks for `.env` in this directory first and then falls back to `../doppelganger_core/.env`.

Expected variables:

```dotenv
OPENAI_API_KEY=your-openai-api-key
OPENAI_VECTOR_STORE_ID=
INTERNAL_DOCUMENTS_SOURCE_DIR=./documents
INTERNAL_DOCUMENTS_VECTOR_STORE_NAME=Internal Documents
```

## Install

```bash
python -m pip install -e ".[test]"
```

## Run

Create or reuse a vector store and upload documents:

```bash
python -m internal_documents_core.ingest --source-dir ./documents
```

Reuse an existing vector store:

```bash
python -m internal_documents_core.ingest \
  --source-dir ./documents \
  --vector-store-id vs_123
```

## Notes

- This first slice uses the OpenAI Vector Stores API and the SDK upload-and-poll helper.
- Supported file extensions are currently `.md`, `.txt`, `.pdf`, and `.docx`.
