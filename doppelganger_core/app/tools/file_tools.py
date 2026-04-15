"""Safe text file helpers for agent-accessible file reads and writes."""

from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_READ_CHARS = 20_000
MAX_WRITE_CHARS = 100_000
DEFAULT_SEARCH_CONTEXT_LINES = 2
DEFAULT_SEARCH_MAX_MATCHES = 20
ALLOWED_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".css",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".html",
    ".sql",
    ".sh",
}
BLOCKED_FILENAMES = {
    ".env",
    ".gmail_token.json",
    ".internal_documents_google_token.json",
}
BLOCKED_PATH_PARTS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".next",
    "secrets",
}


def _sha256_text(content: str) -> str:
    """Return a stable SHA-256 hash for file contents."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_validated_text_file(path: str) -> tuple[Path, str]:
    """Resolve, validate, and read a safe project text file."""
    resolved = resolve_workspace_path(path)
    validate_text_file_path(resolved, must_exist=True)
    return resolved, resolved.read_text(encoding="utf-8")


def resolve_workspace_path(raw_path: str) -> Path:
    """Resolve a user-supplied path within the doppelganger project root."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise RuntimeError("File access is restricted to the doppelganger_core project.") from exc
    return resolved


def validate_text_file_path(path: Path, *, must_exist: bool) -> None:
    """Validate that a path is a safe project text file target."""
    if any(part in BLOCKED_PATH_PARTS for part in path.parts):
        raise RuntimeError("Access to that path is blocked.")
    if path.name in BLOCKED_FILENAMES:
        raise RuntimeError("Access to that file is blocked.")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise RuntimeError("Only approved text-like project files can be read or written.")
    if must_exist and not path.exists():
        raise RuntimeError(f"File does not exist: {path}")
    if path.exists() and path.is_dir():
        raise RuntimeError(f"Path is a directory, not a file: {path}")


def read_file(path: str, *, max_chars: int = MAX_READ_CHARS) -> dict[str, object]:
    """Read a text file from the project root with a bounded response size."""
    if max_chars <= 0:
        raise RuntimeError("max_chars must be greater than zero.")

    resolved, content = _read_validated_text_file(path)
    truncated = len(content) > max_chars
    if truncated:
        content = content[: max_chars - 1] + "…"
    return {
        "status": "ok",
        "path": str(resolved),
        "content": content,
        "truncated": truncated,
    }


def get_file_info(path: str) -> dict[str, object]:
    """Return basic metadata for a safe project text file."""
    resolved, content = _read_validated_text_file(path)
    return {
        "status": "ok",
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "char_count": len(content),
        "line_count": len(content.splitlines()),
        "sha256": _sha256_text(content),
    }


def read_file_window(path: str, start_line: int, end_line: int) -> dict[str, object]:
    """Read an inclusive line window from a project text file."""
    if start_line <= 0 or end_line <= 0:
        raise RuntimeError("start_line and end_line must both be greater than zero.")
    if end_line < start_line:
        raise RuntimeError("end_line must be greater than or equal to start_line.")

    resolved, content = _read_validated_text_file(path)
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    selected = lines[start_line - 1 : end_line]
    window_content = "".join(selected)
    actual_start_line = start_line if selected else None
    actual_end_line = min(end_line, total_lines) if selected else None

    return {
        "status": "ok",
        "path": str(resolved),
        "start_line": start_line,
        "end_line": end_line,
        "actual_start_line": actual_start_line,
        "actual_end_line": actual_end_line,
        "line_count": total_lines,
        "has_more_before": bool(selected) and start_line > 1,
        "has_more_after": bool(selected) and end_line < total_lines,
        "content": window_content,
        "sha256": _sha256_text(content),
    }


def search_in_file(
    path: str,
    query: str,
    *,
    case_sensitive: bool = False,
    context_lines: int = DEFAULT_SEARCH_CONTEXT_LINES,
    max_matches: int = DEFAULT_SEARCH_MAX_MATCHES,
) -> dict[str, object]:
    """Search a file for matching lines and return nearby snippets."""
    if not query:
        raise RuntimeError("query must not be empty.")
    if context_lines < 0:
        raise RuntimeError("context_lines must be zero or greater.")
    if max_matches <= 0:
        raise RuntimeError("max_matches must be greater than zero.")

    resolved, content = _read_validated_text_file(path)
    lines = content.splitlines()
    haystack_query = query if case_sensitive else query.casefold()
    matches: list[dict[str, object]] = []
    hit_limit = False

    for index, line in enumerate(lines):
        haystack_line = line if case_sensitive else line.casefold()
        if haystack_query not in haystack_line:
            continue
        snippet_start = max(0, index - context_lines)
        snippet_end = min(len(lines), index + context_lines + 1)
        matches.append(
            {
                "line_number": index + 1,
                "line_text": line,
                "context_start_line": snippet_start + 1,
                "context_end_line": snippet_end,
                "snippet": "\n".join(lines[snippet_start:snippet_end]),
            }
        )
        if len(matches) >= max_matches:
            hit_limit = True
            break

    return {
        "status": "ok",
        "path": str(resolved),
        "query": query,
        "case_sensitive": case_sensitive,
        "match_count": len(matches),
        "matches": matches,
        "truncated": hit_limit,
        "sha256": _sha256_text(content),
    }


def write_file(
    path: str,
    content: str,
    *,
    append: bool = False,
) -> dict[str, object]:
    """Write or append text content to a project file with basic safety limits."""
    if len(content) > MAX_WRITE_CHARS:
        raise RuntimeError(f"File content exceeds the {MAX_WRITE_CHARS} character write limit.")

    resolved = resolve_workspace_path(path)
    validate_text_file_path(resolved, must_exist=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    existing = resolved.read_text(encoding="utf-8") if append and resolved.exists() else ""
    resolved.write_text(existing + content, encoding="utf-8")
    return {
        "status": "ok",
        "path": str(resolved),
        "bytes_written": len((existing + content).encode("utf-8")),
        "append": append,
    }


def replace_in_file(
    path: str,
    old_text: str,
    new_text: str,
    *,
    expected_hash: str | None = None,
) -> dict[str, object]:
    """Replace one exact text block inside a project text file."""
    if not old_text:
        raise RuntimeError("old_text must not be empty.")
    if len(old_text) > MAX_WRITE_CHARS or len(new_text) > MAX_WRITE_CHARS:
        raise RuntimeError(f"Replacement text exceeds the {MAX_WRITE_CHARS} character limit.")

    resolved, content = _read_validated_text_file(path)
    current_hash = _sha256_text(content)
    if expected_hash is not None and expected_hash != current_hash:
        raise RuntimeError("File contents changed since they were last read.")

    occurrences = content.count(old_text)
    if occurrences == 0:
        raise RuntimeError("old_text was not found in the file.")
    if occurrences > 1:
        raise RuntimeError("old_text appears multiple times; provide a more specific block.")

    updated = content.replace(old_text, new_text, 1)
    resolved.write_text(updated, encoding="utf-8")
    return {
        "status": "ok",
        "path": str(resolved),
        "replacements": 1,
        "old_sha256": current_hash,
        "new_sha256": _sha256_text(updated),
        "bytes_written": len(updated.encode("utf-8")),
    }
