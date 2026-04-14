"""Tests for application creation."""

from __future__ import annotations

from app.main import create_app


def test_create_app_sets_title_and_version() -> None:
    app = create_app()
    assert app.title == "AI Doppelganger API"
    assert app.version == "0.1.0"


def test_create_app_registers_routes() -> None:
    app = create_app()
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/messages/handle" in paths
