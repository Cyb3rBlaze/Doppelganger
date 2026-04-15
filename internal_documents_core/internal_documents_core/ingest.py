"""CLI entrypoint for internal document vector store ingestion."""

from __future__ import annotations

import argparse
import json

from internal_documents_core.vector_store import ingest_documents_to_vector_store


def build_parser() -> argparse.ArgumentParser:
    """Build the ingestion CLI parser."""
    parser = argparse.ArgumentParser(description="Populate an OpenAI vector store with documents.")
    parser.add_argument("--source-dir", default=None, help="Directory containing documents.")
    parser.add_argument("--vector-store-id", default=None, help="Existing vector store id.")
    parser.add_argument("--vector-store-name", default=None, help="Name for a new vector store.")
    return parser


def main() -> None:
    """Run one ingestion job and print the result as JSON."""
    parser = build_parser()
    args = parser.parse_args()
    result = ingest_documents_to_vector_store(
        source_dir=args.source_dir,
        vector_store_id=args.vector_store_id,
        vector_store_name=args.vector_store_name,
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
