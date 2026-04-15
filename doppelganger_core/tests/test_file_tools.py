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


def test_get_file_info_returns_hash_and_counts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "mind" / "SOUL.md"
    target.parent.mkdir(parents=True)
    target.write_text("line one\nline two\n", encoding="utf-8")

    result = file_tools.get_file_info("mind/SOUL.md")

    assert result["status"] == "ok"
    assert result["char_count"] == len("line one\nline two\n")
    assert result["line_count"] == 2
    assert isinstance(result["sha256"], str)
    assert len(result["sha256"]) == 64


def test_read_file_window_returns_requested_line_slice(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "mind" / "DIRECTIVES.md"
    target.parent.mkdir(parents=True)
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    result = file_tools.read_file_window("mind/DIRECTIVES.md", 2, 3)

    assert result["status"] == "ok"
    assert result["content"] == "two\nthree\n"
    assert result["actual_start_line"] == 2
    assert result["actual_end_line"] == 3
    assert result["has_more_before"] is True
    assert result["has_more_after"] is True


def test_search_in_file_returns_matching_snippet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "notes.md"
    target.write_text("alpha\nbeta clue\ngamma\n", encoding="utf-8")

    result = file_tools.search_in_file("notes.md", "clue")

    assert result["status"] == "ok"
    assert result["match_count"] == 1
    assert result["truncated"] is False
    assert result["matches"][0]["line_number"] == 2
    assert "beta clue" in result["matches"][0]["snippet"]


def test_replace_in_file_replaces_exact_block_with_hash_check(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "mind" / "DIRECTIVES.md"
    target.parent.mkdir(parents=True)
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    file_info = file_tools.get_file_info("mind/DIRECTIVES.md")

    result = file_tools.replace_in_file(
        "mind/DIRECTIVES.md",
        "beta",
        "updated beta",
        expected_hash=file_info["sha256"],
    )

    assert result["status"] == "ok"
    assert result["replacements"] == 1
    assert result["old_sha256"] == file_info["sha256"]
    assert target.read_text(encoding="utf-8") == "alpha\nupdated beta\ngamma\n"


def test_replace_in_file_rejects_non_unique_match(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "notes.md"
    target.write_text("repeat\nrepeat\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="multiple times"):
        file_tools.replace_in_file("notes.md", "repeat", "updated")


def test_replace_in_file_rejects_stale_hash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_tools, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "notes.md"
    target.write_text("hello\nworld\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed since they were last read"):
        file_tools.replace_in_file(
            "notes.md",
            "world",
            "updated",
            expected_hash="not-the-real-hash",
        )
