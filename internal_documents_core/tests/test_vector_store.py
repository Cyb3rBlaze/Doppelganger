"""Tests for internal document vector store ingestion."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from internal_documents_core import vector_store


def test_collect_document_paths_filters_supported_files(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("a", encoding="utf-8")
    (docs_dir / "b.txt").write_text("b", encoding="utf-8")
    (docs_dir / "ignore.csv").write_text("c", encoding="utf-8")

    paths = vector_store.collect_document_paths(docs_dir)

    assert [path.name for path in paths] == ["a.md", "b.txt"]


def test_ingest_documents_to_vector_store_uploads_and_polls(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("alpha", encoding="utf-8")
    (docs_dir / "b.txt").write_text("beta", encoding="utf-8")

    upload_calls: list[dict[str, object]] = []

    class FakeFileBatches:
        def upload_and_poll(self, *, vector_store_id, files):
            upload_calls.append(
                {
                    "vector_store_id": vector_store_id,
                    "file_names": [Path(file_obj.name).name for file_obj in files],
                }
            )
            return SimpleNamespace(
                status="completed",
                file_counts=SimpleNamespace(completed=2, failed=0),
            )

    class FakeVectorStores:
        def __init__(self) -> None:
            self.file_batches = FakeFileBatches()

        def create(self, *, name):
            return SimpleNamespace(id="vs_123", name=name)

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    result = vector_store.ingest_documents_to_vector_store(
        source_dir=docs_dir,
        vector_store_name="Internal Documents",
        client=fake_client,
    )

    assert result.vector_store_id == "vs_123"
    assert result.file_count == 2
    assert result.file_batch_status == "completed"
    assert result.completed_count == 2
    assert result.failed_count == 0
    assert upload_calls == [
        {
            "vector_store_id": "vs_123",
            "file_names": ["a.md", "b.txt"],
        }
    ]
