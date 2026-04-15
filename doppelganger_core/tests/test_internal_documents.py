"""Tests for internal documents retrieval helpers."""

from __future__ import annotations

from types import SimpleNamespace

from app.core.models import Message
from app.services import internal_documents


def test_get_internal_documents_embedding_dimension_defaults(monkeypatch) -> None:
    monkeypatch.delenv("INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION", raising=False)

    assert internal_documents.get_internal_documents_embedding_dimension() == 1536


def test_embed_query_text_calls_openai_embeddings_api() -> None:
    class FakeEmbeddingsClient:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.3, 0.4])])

    fake_client = SimpleNamespace(embeddings=FakeEmbeddingsClient())

    embedding = internal_documents.embed_query_text(
        "find notes about investing",
        client=fake_client,
        model="text-embedding-3-small",
        dimensions=1536,
    )

    assert embedding == [0.3, 0.4]
    assert fake_client.embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": "find notes about investing",
            "dimensions": 1536,
        }
    ]


def test_retrieve_internal_document_context_sync_returns_top_documents(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="investing notes")
    monkeypatch.setattr(
        internal_documents,
        "get_internal_documents_dsn",
        lambda: "postgresql://localhost/internal_documents",
    )
    monkeypatch.setattr(
        internal_documents,
        "embed_query_text",
        lambda text: [0.1, 0.2],
    )
    monkeypatch.setattr(
        internal_documents,
        "search_internal_documents",
        lambda query_embedding, limit=3: [
            {
                "document_id": "gdoc:abc123",
                "source_path": "/docs/investing.gdoc",
                "source_kind": "gdoc",
                "title": "Investing",
                "content": "a" * 2000,
                "metadata": {"doc_id": "abc123"},
                "score": 0.91,
            }
        ],
    )

    results = internal_documents.retrieve_internal_document_context_sync(message, limit=3)

    assert len(results) == 1
    assert results[0]["document_id"] == "gdoc:abc123"
    assert results[0]["source_path"] == "/docs/investing.gdoc"
    assert results[0]["score"] == 0.91
    assert results[0]["content"].endswith("…")


def test_search_internal_documents_for_query_returns_top_documents(monkeypatch) -> None:
    monkeypatch.setattr(
        internal_documents,
        "get_internal_documents_dsn",
        lambda: "postgresql://localhost/internal_documents",
    )
    monkeypatch.setattr(
        internal_documents,
        "embed_query_text",
        lambda text: [0.1, 0.2],
    )
    monkeypatch.setattr(
        internal_documents,
        "search_internal_documents",
        lambda query_embedding, limit=3: [
            {
                "document_id": "gdoc:abc123",
                "source_path": "/docs/investing.gdoc",
                "source_kind": "gdoc",
                "title": "Investing",
                "content": "useful notes",
                "metadata": {"doc_id": "abc123"},
                "score": 0.91,
            }
        ],
    )

    results = internal_documents.search_internal_documents_for_query("investing notes", limit=3)

    assert results == [
        {
            "document_id": "gdoc:abc123",
            "source_path": "/docs/investing.gdoc",
            "source_kind": "gdoc",
            "title": "Investing",
            "content": "useful notes",
            "metadata": {"doc_id": "abc123"},
            "score": 0.91,
        }
    ]


def test_retrieve_internal_document_context_sync_returns_empty_without_dsn(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="investing notes")
    monkeypatch.setattr(internal_documents, "get_internal_documents_dsn", lambda: None)

    assert internal_documents.retrieve_internal_document_context_sync(message) == []


def test_looks_like_knowledge_seeking_query_detects_questions() -> None:
    message = Message(channel="api", user_id="anshul", text="What did I write about investing?")

    assert internal_documents.looks_like_knowledge_seeking_query(message) is True


def test_looks_like_knowledge_seeking_query_ignores_chitchat() -> None:
    message = Message(channel="api", user_id="anshul", text="sounds good thanks")

    assert internal_documents.looks_like_knowledge_seeking_query(message) is False
