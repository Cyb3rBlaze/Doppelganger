"""Tests for ingestion flow and skipped-document reporting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core import ingest
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

    upserted_document_ids: list[str] = []

    def fake_embed_text(text: str, *, dimensions: int):
        if text == long_doc.content:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': \"Invalid 'input': maximum context length is 8192 tokens.\"}}"
            )
        return [0.1, 0.2]

    monkeypatch.setattr(ingest, "embed_text", fake_embed_text)
    monkeypatch.setattr(
        ingest,
        "upsert_document",
        lambda document, embedding, config=None: upserted_document_ids.append(document.document_id),
    )

    result = ingest.ingest_documents(
        docs_dir,
        config=SimpleNamespace(embedding_dimension=1536),
        report_path=report_path,
    )

    assert result["document_count"] == 2
    assert result["stored_count"] == 1
    assert result["skipped_count"] == 1
    assert result["skipped_report_path"] == str(report_path)
    assert upserted_document_ids == ["file:short"]
    assert report_path.exists()
    assert str(long_path) in report_path.read_text(encoding="utf-8")


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
    monkeypatch.setattr(ingest, "embed_text", lambda text, *, dimensions: [0.1, 0.2])
    stored: list[str] = []
    monkeypatch.setattr(
        ingest,
        "upsert_document",
        lambda document, embedding, config=None: stored.append(document.document_id),
    )

    result = ingest.ingest_documents(
        docs_dir,
        config=SimpleNamespace(embedding_dimension=1536),
        report_path=report_path,
    )

    assert result["document_count"] == 2
    assert result["stored_count"] == 1
    assert result["skipped_count"] == 1
    assert stored == ["file:good"]
    report_contents = report_path.read_text(encoding="utf-8")
    assert str(bad_path) in report_contents
    assert "Google export failed" in report_contents
