"""Safe text file helpers for agent-accessible file reads and writes."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_READ_CHARS = 20_000
MAX_WRITE_CHARS = 100_000
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

    resolved = resolve_workspace_path(path)
    validate_text_file_path(resolved, must_exist=True)
    content = resolved.read_text(encoding="utf-8")
    truncated = len(content) > max_chars
    if truncated:
        content = content[: max_chars - 1] + "…"
    return {
        "status": "ok",
        "path": str(resolved),
        "content": content,
        "truncated": truncated,
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
