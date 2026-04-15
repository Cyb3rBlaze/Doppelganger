"""CLI entrypoint for internal document ingestion and search."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

from core.chunking import build_adaptive_document_chunk_result
from core.document_sources import (
    build_google_drive_service,
    collect_document_paths,
    load_document,
    resolve_source_dir,
)
from core.embeddings import embed_text, is_context_length_error
from core.vector_store import (
    ensure_pgvector_schema,
    get_vector_store_config,
    replace_document_chunks,
    search_documents,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIPPED_DOCUMENTS_REPORT = PROJECT_ROOT / "skipped_documents.txt"
CHUNK_MERGE_REPORT = PROJECT_ROOT / "chunk_merge_report.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the ingestion CLI parser."""
    parser = argparse.ArgumentParser(
        description="Ingest internal documents into PostgreSQL/pgvector or run vector search."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest adaptive chunk/window embeddings per document.",
    )
    ingest_parser.add_argument(
        "--source-dir",
        default=None,
        help="Directory containing internal documents.",
    )

    search_parser = subparsers.add_parser("search", help="Run a pgvector similarity search.")
    search_parser.add_argument("query", help="Natural-language search query.")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return.")
    return parser


def _load_tqdm():
    """Import tqdm lazily so the CLI can still run with a plain fallback."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm


def iter_with_progress(document_paths: list[Path]) -> Iterable[Path]:
    """Wrap the ingest iterable with a tqdm progress bar when available."""
    tqdm = _load_tqdm()
    if tqdm is None:
        return document_paths
    return tqdm(document_paths, desc="Ingesting documents", unit="doc")


def progress_write(message: str) -> None:
    """Write a progress message without breaking the tqdm bar when available."""
    tqdm = _load_tqdm()
    if tqdm is not None:
        tqdm.write(message)
        return
    print(message)


def write_skipped_documents_report(
    skipped_documents: list[dict[str, str]],
    *,
    report_path: Path = SKIPPED_DOCUMENTS_REPORT,
) -> str | None:
    """Write a plaintext report of documents skipped during ingest."""
    if not skipped_documents:
        if report_path.exists():
            report_path.unlink()
        return None

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Skipped documents during ingest.",
        "",
    ]
    for item in skipped_documents:
        lines.append(f"- path: {item['source_path']}")
        lines.append(f"  title: {item['title']}")
        lines.append(f"  reason: {item['reason']}")
        lines.append("")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(report_path)


def write_chunk_merge_report(
    report: list[dict[str, Any]],
    *,
    report_path: Path = CHUNK_MERGE_REPORT,
) -> str | None:
    """Write a JSON report of adaptive chunk/window merge decisions."""
    if not report:
        if report_path.exists():
            report_path.unlink()
        return None

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(report_path)


def ingest_documents(
    source_dir: Path,
    *,
    config: Any,
    report_path: Path = SKIPPED_DOCUMENTS_REPORT,
    chunk_merge_report_path: Path = CHUNK_MERGE_REPORT,
) -> dict[str, Any]:
    """Ingest supported documents into adaptive chunk/window rows."""
    document_paths = collect_document_paths(source_dir)
    progress_write(f"Found {len(document_paths)} supported documents in {source_dir}.")
    drive_service = (
        build_google_drive_service()
        if any(path.suffix.lower() == ".gdoc" for path in document_paths)
        else None
    )
    stored_document_count = 0
    stored_chunk_count = 0
    skipped_documents: list[dict[str, str]] = []
    chunk_merge_report: list[dict[str, Any]] = []
    progress = iter_with_progress(document_paths)
    for path in progress:
        try:
            document = load_document(path, drive_service=drive_service)
        except Exception as exc:
            skipped_documents.append(
                {
                    "source_path": str(path),
                    "title": path.stem,
                    "reason": str(exc),
                }
            )
            progress_write(f"Skipped load failure: {path}")
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(docs=stored_document_count, chunks=stored_chunk_count, skipped=len(skipped_documents))
            continue

        try:
            chunk_result = build_adaptive_document_chunk_result(
                document,
                embed_fn=lambda text: embed_text(text, dimensions=config.embedding_dimension),
            )
        except Exception as exc:
            if not is_context_length_error(exc):
                raise
            skipped_documents.append(
                {
                    "source_path": document.source_path,
                    "title": document.title,
                    "reason": str(exc),
                }
            )
            progress_write(f"Skipped oversized document: {document.source_path}")
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(docs=stored_document_count, chunks=stored_chunk_count, skipped=len(skipped_documents))
            continue
        replace_document_chunks(document.document_id, chunk_result.embedded_chunks, config=config)
        stored_document_count += 1
        stored_chunk_count += len(chunk_result.embedded_chunks)
        chunk_merge_report.append(
            {
                "document_id": document.document_id,
                "source_path": document.source_path,
                "title": document.title,
                "base_chunk_count": len(chunk_result.decisions),
                "stored_window_count": len(chunk_result.embedded_chunks),
                "decisions": [asdict(decision) for decision in chunk_result.decisions],
            }
        )
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(docs=stored_document_count, chunks=stored_chunk_count, skipped=len(skipped_documents))

    return {
        "status": "ingested",
        "source_dir": str(source_dir),
        "document_count": len(document_paths),
        "stored_count": stored_document_count,
        "stored_document_count": stored_document_count,
        "stored_chunk_count": stored_chunk_count,
        "skipped_count": len(skipped_documents),
        "embedding_dimension": config.embedding_dimension,
        "skipped_report_path": write_skipped_documents_report(skipped_documents, report_path=report_path),
        "chunk_merge_report_path": write_chunk_merge_report(
            chunk_merge_report,
            report_path=chunk_merge_report_path,
        ),
    }


def main() -> None:
    """Run ingestion or search and print results as JSON."""
    parser = build_parser()
    args = parser.parse_args()
    config = get_vector_store_config()
    ensure_pgvector_schema(config=config)

    if args.command == "ingest":
        source_dir = resolve_source_dir(args.source_dir)
        print(json.dumps(ingest_documents(source_dir, config=config), indent=2))
        return

    if args.command == "search":
        query_embedding = embed_text(args.query, dimensions=config.embedding_dimension)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "results": search_documents(query_embedding, limit=args.limit, config=config),
                },
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
