"""Tests for safe agent-accessible file helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools import file_tools


def test_read_file_reads_project_text_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "mind" / "SOUL.md"
    target.parent.mkdir(parents=True)
    target.write_text("hello world", encoding="utf-8")

    result = file_tools.read_file("mind/SOUL.md")

    assert result["status"] == "ok"
    assert result["content"] == "hello world"
    assert result["truncated"] is False


def test_read_file_blocks_env_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)
    target = tmp_path / ".env"
    target.write_text("OPENAI_API_KEY=secret", encoding="utf-8")

    with pytest.raises(RuntimeError):
        file_tools.read_file(".env")


def test_write_file_writes_project_text_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)

    result = file_tools.write_file("mind/DIRECTIVES.md", "# DIRECTIVES\n- test\n")

    target = tmp_path / "mind" / "DIRECTIVES.md"
    assert result["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "# DIRECTIVES\n- test\n"


def test_write_file_append_mode_appends_content(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "notes.md"
    target.write_text("hello", encoding="utf-8")

    file_tools.write_file("notes.md", "\nworld", append=True)

    assert target.read_text(encoding="utf-8") == "hello\nworld"


def test_write_file_blocks_paths_outside_project(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError):
        file_tools.write_file("../outside.md", "bad")
