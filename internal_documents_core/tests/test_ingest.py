"""Tests for ingestion flow and skipped-document reporting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core import ingest
from core.chunking import AdaptiveChunkBuildResult, ChunkDecision
from core.vector_store import DocumentChunkRecord, EmbeddedChunkRecord
from core.document_sources import InternalDocument


def test_iter_with_progress_falls_back_without_tqdm(tmp_path: Path, monkeypatch) -> None:
    paths = [tmp_path / "a.md", tmp_path / "b.md"]
    monkeypatch.setattr(ingest, "_load_tqdm", lambda: None)

    wrapped = ingest.iter_with_progress(paths)

    assert list(wrapped) == paths


def test_progress_write_falls_back_to_print(monkeypatch, capsys) -> None:
    monkeypatch.setattr(ingest, "_load_tqdm", lambda: None)

    ingest.progress_write("hello progress")

    captured = capsys.readouterr()
    assert "hello progress" in captured.out


def test_write_skipped_documents_report_writes_plaintext_file(tmp_path: Path) -> None:
    report_path = tmp_path / "skipped.txt"

    written_path = ingest.write_skipped_documents_report(
        [
            {
                "source_path": "/tmp/too-long.md",
                "title": "too-long",
                "reason": "maximum context length is 8192 tokens",
            }
        ],
        report_path=report_path,
    )

    assert written_path == str(report_path)
    assert report_path.exists()
    contents = report_path.read_text(encoding="utf-8")
    assert "/tmp/too-long.md" in contents
    assert "maximum context length is 8192 tokens" in contents


def test_write_chunk_merge_report_writes_json_file(tmp_path: Path) -> None:
    report_path = tmp_path / "chunk_merge_report.json"

    written_path = ingest.write_chunk_merge_report(
        [
            {
                "document_id": "file:test",
                "decisions": [
                    {
                        "output_index": 0,
                        "base_chunk_index": 0,
                        "merged": False,
                        "similarity_to_previous_window": None,
                    }
                ],
            }
        ],
        report_path=report_path,
    )

    assert written_path == str(report_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))[0]["document_id"] == "file:test"


def test_ingest_documents_skips_oversized_documents_and_continues(
    monkeypatch,
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    short_path = docs_dir / "short.md"
    long_path = docs_dir / "long.md"
    short_path.write_text("short", encoding="utf-8")
    long_path.write_text("long", encoding="utf-8")
    report_path = tmp_path / "skipped_long_documents.txt"
    chunk_merge_report_path = tmp_path / "chunk_merge_report.json"

    short_doc = InternalDocument(
        document_id="file:short",
        source_path=str(short_path),
        source_kind="local_text",
        title="short",
        content="short text",
        metadata={},
    )
    long_doc = InternalDocument(
        document_id="file:long",
        source_path=str(long_path),
        source_kind="local_text",
        title="long",
        content="x" * 10000,
        metadata={},
    )

    monkeypatch.setattr(ingest, "collect_document_paths", lambda source_dir: [short_path, long_path])
    monkeypatch.setattr(ingest, "load_document", lambda path, drive_service=None: short_doc if path == short_path else long_doc)

    stored_document_ids: list[str] = []

    def fake_build_adaptive_document_chunk_result(document, *, embed_fn, **kwargs):
        if document.document_id == long_doc.document_id:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': \"Invalid 'input': maximum context length is 8192 tokens.\"}}"
            )
        return AdaptiveChunkBuildResult(
            embedded_chunks=[
                EmbeddedChunkRecord(
                    record=DocumentChunkRecord(
                        document_id=document.document_id,
                        chunk_id=f"{document.document_id}:chunk:0",
                        source_path=document.source_path,
                        source_kind=document.source_kind,
                        title=document.title,
                        content=document.content,
                        metadata=document.metadata,
                        connected_nodes=[],
                        chunk_index=0,
                        window_start_chunk_index=0,
                        window_end_chunk_index=0,
                    ),
                    embedding=[0.1, 0.2],
                )
            ],
            decisions=[
                ChunkDecision(
                    output_index=0,
                    base_chunk_index=0,
                    window_start_chunk_index=0,
                    window_end_chunk_index=0,
                    merged=False,
                    similarity_to_previous_window=None,
                )
            ],
        )

    monkeypatch.setattr(ingest, "build_adaptive_document_chunk_result", fake_build_adaptive_document_chunk_result)
    monkeypatch.setattr(
        ingest,
        "replace_document_chunks",
        lambda document_id, chunk_records, config=None: stored_document_ids.append(document_id),
    )

    result = ingest.ingest_documents(
        docs_dir,
        config=SimpleNamespace(embedding_dimension=1536),
        report_path=report_path,
        chunk_merge_report_path=chunk_merge_report_path,
    )

    assert result["document_count"] == 2
    assert result["stored_count"] == 1
    assert result["stored_document_count"] == 1
    assert result["stored_chunk_count"] == 1
    assert result["skipped_count"] == 1
    assert result["skipped_report_path"] == str(report_path)
    assert result["chunk_merge_report_path"] == str(chunk_merge_report_path)
    assert stored_document_ids == ["file:short"]
    assert report_path.exists()
    assert chunk_merge_report_path.exists()
    assert str(long_path) in report_path.read_text(encoding="utf-8")
    assert json.loads(chunk_merge_report_path.read_text(encoding="utf-8"))[0]["document_id"] == "file:short"


def test_ingest_documents_skips_load_failures_and_continues(
    monkeypatch,
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    good_path = docs_dir / "good.md"
    bad_path = docs_dir / "bad.gdoc"
    good_path.write_text("good", encoding="utf-8")
    bad_path.write_text("bad", encoding="utf-8")
    report_path = tmp_path / "skipped_documents.txt"
    chunk_merge_report_path = tmp_path / "chunk_merge_report.json"

    good_doc = InternalDocument(
        document_id="file:good",
        source_path=str(good_path),
        source_kind="local_text",
        title="good",
        content="good text",
        metadata={},
    )

    monkeypatch.setattr(ingest, "collect_document_paths", lambda source_dir: [good_path, bad_path])
    monkeypatch.setattr(ingest, "build_google_drive_service", lambda: object())

    def fake_load_document(path: Path, drive_service=None):
        if path == bad_path:
            raise RuntimeError("Google export failed")
        return good_doc

    monkeypatch.setattr(ingest, "load_document", fake_load_document)
    monkeypatch.setattr(
        ingest,
        "build_adaptive_document_chunk_result",
        lambda document, *, embed_fn, **kwargs: AdaptiveChunkBuildResult(
            embedded_chunks=[
                EmbeddedChunkRecord(
                    record=DocumentChunkRecord(
                        document_id=document.document_id,
                        chunk_id=f"{document.document_id}:chunk:0",
                        source_path=document.source_path,
                        source_kind=document.source_kind,
                        title=document.title,
                        content=document.content,
                        metadata=document.metadata,
                        connected_nodes=[],
                        chunk_index=0,
                        window_start_chunk_index=0,
                        window_end_chunk_index=0,
                    ),
                    embedding=[0.1, 0.2],
                )
            ],
            decisions=[
                ChunkDecision(
                    output_index=0,
                    base_chunk_index=0,
                    window_start_chunk_index=0,
                    window_end_chunk_index=0,
                    merged=False,
                    similarity_to_previous_window=None,
                )
            ],
        ),
    )
    stored: list[str] = []
    monkeypatch.setattr(
        ingest,
        "replace_document_chunks",
        lambda document_id, chunk_records, config=None: stored.append(document_id),
    )

    result = ingest.ingest_documents(
        docs_dir,
        config=SimpleNamespace(embedding_dimension=1536),
        report_path=report_path,
        chunk_merge_report_path=chunk_merge_report_path,
    )

    assert result["document_count"] == 2
    assert result["stored_count"] == 1
    assert result["stored_document_count"] == 1
    assert result["stored_chunk_count"] == 1
    assert result["skipped_count"] == 1
    assert result["chunk_merge_report_path"] == str(chunk_merge_report_path)
    assert stored == ["file:good"]
    report_contents = report_path.read_text(encoding="utf-8")
    assert str(bad_path) in report_contents
    assert "Google export failed" in report_contents


def test_ingest_documents_writes_merge_decision_json(monkeypatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    note_path = docs_dir / "note.md"
    note_path.write_text("note", encoding="utf-8")
    chunk_merge_report_path = tmp_path / "chunk_merge_report.json"

    note_doc = InternalDocument(
        document_id="file:note",
        source_path=str(note_path),
        source_kind="local_text",
        title="note",
        content="note text",
        metadata={"kind": "note"},
    )

    monkeypatch.setattr(ingest, "collect_document_paths", lambda source_dir: [note_path])
    monkeypatch.setattr(ingest, "load_document", lambda path, drive_service=None: note_doc)
    monkeypatch.setattr(
        ingest,
        "build_adaptive_document_chunk_result",
        lambda document, *, embed_fn, **kwargs: AdaptiveChunkBuildResult(
            embedded_chunks=[
                EmbeddedChunkRecord(
                    record=DocumentChunkRecord(
                        document_id=document.document_id,
                        chunk_id=f"{document.document_id}:chunk:0",
                        source_path=document.source_path,
                        source_kind=document.source_kind,
                        title=document.title,
                        content="chunk zero\n\nchunk one",
                        metadata=document.metadata,
                        connected_nodes=[],
                        chunk_index=0,
                        window_start_chunk_index=0,
                        window_end_chunk_index=1,
                    ),
                    embedding=[0.2, 0.3],
                ),
            ],
            decisions=[
                ChunkDecision(
                    output_index=0,
                    base_chunk_index=0,
                    window_start_chunk_index=0,
                    window_end_chunk_index=0,
                    merged=False,
                    similarity_to_previous_window=None,
                ),
                ChunkDecision(
                    output_index=1,
                    base_chunk_index=1,
                    window_start_chunk_index=0,
                    window_end_chunk_index=1,
                    merged=True,
                    similarity_to_previous_window=0.83,
                ),
            ],
        ),
    )
    monkeypatch.setattr(
        ingest,
        "replace_document_chunks",
        lambda document_id, chunk_records, config=None: True,
    )

    result = ingest.ingest_documents(
        docs_dir,
        config=SimpleNamespace(embedding_dimension=1536),
        chunk_merge_report_path=chunk_merge_report_path,
    )

    report = json.loads(chunk_merge_report_path.read_text(encoding="utf-8"))
    assert result["chunk_merge_report_path"] == str(chunk_merge_report_path)
    assert report[0]["document_id"] == "file:note"
    assert report[0]["stored_window_count"] == 1
    assert report[0]["decisions"][0]["merged"] is False
    assert report[0]["decisions"][0]["similarity_to_previous_window"] is None
    assert report[0]["decisions"][1]["merged"] is True
    assert report[0]["decisions"][1]["similarity_to_previous_window"] == 0.83
