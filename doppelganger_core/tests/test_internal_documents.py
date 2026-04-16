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
                "chunk_id": "gdoc:abc123:chunk:0",
                "source_path": "/docs/investing.gdoc",
                "source_kind": "gdoc",
                "title": "Investing",
                "content": "a" * 2000,
                "metadata": {"doc_id": "abc123"},
                "score": 0.91,
                "connected_nodes": [],
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
                "chunk_id": "gdoc:abc123:chunk:0",
                "source_path": "/docs/investing.gdoc",
                "source_kind": "gdoc",
                "title": "Investing",
                "content": "useful notes",
                "metadata": {"doc_id": "abc123"},
                "score": 0.91,
                "connected_nodes": [],
            }
        ],
    )

    results = internal_documents.search_internal_documents_for_query("investing notes", limit=3)

    assert results == [
        {
            "document_id": "gdoc:abc123",
            "chunk_id": "gdoc:abc123:chunk:0",
            "source_path": "/docs/investing.gdoc",
            "source_kind": "gdoc",
            "title": "Investing",
            "content": "useful notes",
            "metadata": {"doc_id": "abc123"},
            "score": 0.91,
            "connected_nodes": [],
            "retrieval_layer": 0,
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


def test_expand_internal_document_subgraph_adds_two_hops(monkeypatch) -> None:
    seed_documents = [
        {
            "document_id": "gdoc:seed",
            "chunk_id": "seed",
            "source_path": "/docs/seed.gdoc",
            "source_kind": "gdoc",
            "title": "Seed",
            "content": "seed content",
            "metadata": {"doc_id": "seed"},
            "connected_nodes": [
                {"chunk_id": "hop1a", "score": 0.82},
                {"chunk_id": "hop1b", "score": 0.78},
            ],
            "score": 0.93,
        }
    ]

    fetched_by_call = [
        [
            {
                "document_id": "gdoc:hop1a",
                "chunk_id": "hop1a",
                "source_path": "/docs/hop1a.gdoc",
                "source_kind": "gdoc",
                "title": "Hop 1A",
                "content": "hop1a content",
                "metadata": {"doc_id": "hop1a"},
                "connected_nodes": [{"chunk_id": "hop2", "score": 0.74}],
            },
            {
                "document_id": "gdoc:hop1b",
                "chunk_id": "hop1b",
                "source_path": "/docs/hop1b.gdoc",
                "source_kind": "gdoc",
                "title": "Hop 1B",
                "content": "hop1b content",
                "metadata": {"doc_id": "hop1b"},
                "connected_nodes": [{"chunk_id": "hop2", "score": 0.69}],
            },
        ],
        [
            {
                "document_id": "gdoc:hop2",
                "chunk_id": "hop2",
                "source_path": "/docs/hop2.gdoc",
                "source_kind": "gdoc",
                "title": "Hop 2",
                "content": "hop2 content",
                "metadata": {"doc_id": "hop2"},
                "connected_nodes": [],
            }
        ],
    ]

    def fake_fetch(chunk_ids, postgres_dsn=None):
        _ = postgres_dsn
        if chunk_ids == ["hop1a", "hop1b"]:
            return fetched_by_call[0]
        if chunk_ids == ["hop2"]:
            return fetched_by_call[1]
        raise AssertionError(f"Unexpected chunk ids: {chunk_ids!r}")

    monkeypatch.setattr(
        internal_documents,
        "fetch_internal_document_chunks_by_ids",
        fake_fetch,
    )

    expanded = internal_documents.expand_internal_document_subgraph(seed_documents, steps=2)

    assert [document["chunk_id"] for document in expanded] == ["seed", "hop1a", "hop1b", "hop2"]
    assert [document["retrieval_layer"] for document in expanded] == [0, 1, 1, 2]
    assert expanded[0]["score"] == 0.93
    assert expanded[1]["score"] == 0.82
    assert expanded[3]["score"] == 0.74
