"""Shared pytest fixtures for the AI doppelganger test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def sample_telegram_update_dict() -> dict:
    """Return a representative Telegram update payload for adapter tests."""
    return {
        "update_id": 101,
        "message": {
            "message_id": 55,
            "date": 1713123456,
            "text": "hello there",
            "from": {
                "id": 999,
                "username": "anshul",
                "first_name": "Anshul",
            },
            "chat": {
                "id": 12345,
                "type": "private",
            },
        },
    }


@pytest.fixture
def app_client() -> TestClient:
    """Return a FastAPI test client for API route tests."""
    return TestClient(create_app())
