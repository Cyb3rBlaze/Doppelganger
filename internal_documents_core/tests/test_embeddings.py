"""Tests for embedding helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core import embeddings


def test_get_embedding_dimension_uses_model_defaults(monkeypatch) -> None:
    monkeypatch.delenv("INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION", raising=False)

    assert embeddings.get_embedding_dimension("text-embedding-3-small") == 1536
    assert embeddings.get_embedding_dimension("text-embedding-3-large") == 3072


def test_get_embedding_dimension_prefers_env_override(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION", "1024")

    assert embeddings.get_embedding_dimension("text-embedding-3-small") == 1024


def test_embed_text_calls_openai_embeddings_api(monkeypatch) -> None:
    class FakeEmbeddingsClient:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    fake_client = SimpleNamespace(embeddings=FakeEmbeddingsClient())

    embedding = embeddings.embed_text(
        "hello",
        client=fake_client,
        model="text-embedding-3-small",
        dimensions=1536,
    )

    assert embedding == [0.1, 0.2]
    assert fake_client.embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": "hello",
            "dimensions": 1536,
        }
    ]


def test_get_embedding_dimension_raises_for_unknown_model_without_override(monkeypatch) -> None:
    monkeypatch.delenv("INTERNAL_DOCUMENTS_EMBEDDING_DIMENSION", raising=False)

    with pytest.raises(RuntimeError):
        embeddings.get_embedding_dimension("unknown-model")


def test_is_context_length_error_detects_openai_bad_request() -> None:
    exc = RuntimeError("Error code: 400 - Invalid 'input': maximum context length is 8192 tokens.")

    assert embeddings.is_context_length_error(exc) is True


def test_is_context_length_error_ignores_other_errors() -> None:
    exc = RuntimeError("temporary network failure")

    assert embeddings.is_context_length_error(exc) is False
