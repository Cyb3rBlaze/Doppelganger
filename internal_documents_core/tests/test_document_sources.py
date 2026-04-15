"""Tests for internal document discovery and loading."""

from __future__ import annotations

from pathlib import Path

from core import document_sources


def test_collect_document_paths_filters_supported_files(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.gdoc").write_text("{}", encoding="utf-8")
    (docs_dir / "b.md").write_text("b", encoding="utf-8")
    (docs_dir / "c.txt").write_text("c", encoding="utf-8")
    (docs_dir / "ignore.csv").write_text("d", encoding="utf-8")

    paths = document_sources.collect_document_paths(docs_dir)

    assert [path.name for path in paths] == ["a.gdoc", "b.md", "c.txt"]


def test_parse_google_workspace_pointer_reads_doc_id(tmp_path: Path) -> None:
    path = tmp_path / "note.gdoc"
    path.write_text('{"doc_id":"abc123","resource_key":"","email":"me@example.com"}', encoding="utf-8")

    pointer = document_sources.parse_google_workspace_pointer(path)

    assert pointer["doc_id"] == "abc123"


def test_load_document_reads_local_text_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Hello", encoding="utf-8")

    document = document_sources.load_document(path)

    assert document.document_id.startswith("file:")
    assert document.source_kind == "local_text"
    assert document.title == "note"
    assert document.content == "# Hello"


def test_load_document_exports_google_doc_text(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "note.gdoc"
    path.write_text(
        '{"doc_id":"abc123","resource_key":"","email":"me@example.com"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        document_sources,
        "export_google_doc_text",
        lambda file_id, drive_service=None: f"content for {file_id}",
    )

    document = document_sources.load_document(path)

    assert document.document_id == "gdoc:abc123"
    assert document.source_kind == "gdoc"
    assert document.content == "content for abc123"
    assert document.metadata["email"] == "me@example.com"


def test_get_google_oauth_token_path_falls_back_when_env_is_empty(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_DOCUMENTS_GOOGLE_OAUTH_TOKEN_PATH", "")

    path = document_sources.get_google_oauth_token_path()

    assert path == document_sources.PROJECT_ROOT / ".internal_documents_google_token.json"


def test_export_google_doc_text_retries_retryable_errors(monkeypatch) -> None:
    class RetryableError(Exception):
        def __init__(self, status: int) -> None:
            self.resp = type("Resp", (), {"status": status})()
            super().__init__("internal error")

    class FakeExportRequest:
        def __init__(self) -> None:
            self.attempts = 0

        def execute(self):
            self.attempts += 1
            if self.attempts < 3:
                raise RetryableError(500)
            return b"hello"

    class FakeFilesService:
        def __init__(self) -> None:
            self.request = FakeExportRequest()

        def export(self, *, fileId: str, mimeType: str):
            assert fileId == "abc123"
            assert mimeType == "text/plain"
            return self.request

    class FakeDriveService:
        def __init__(self) -> None:
            self._files = FakeFilesService()

        def files(self):
            return self._files

    monkeypatch.setattr(document_sources.time, "sleep", lambda seconds: None)

    content = document_sources.export_google_doc_text("abc123", drive_service=FakeDriveService())

    assert content == "hello"
