"""Document discovery and loading for internal_documents_core."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DOTENV_PATH = PROJECT_ROOT / ".env"
FALLBACK_DOTENV_PATH = PROJECT_ROOT.parent / "doppelganger_core" / ".env"
SUPPORTED_DOCUMENT_EXTENSIONS = (".gdoc", ".md", ".txt")
GOOGLE_DRIVE_SCOPES = ("https://www.googleapis.com/auth/drive.readonly",)
GOOGLE_CLIENT_SECRET_ENV = "INTERNAL_DOCUMENTS_GOOGLE_OAUTH_CLIENT_SECRET_PATH"
GOOGLE_TOKEN_ENV = "INTERNAL_DOCUMENTS_GOOGLE_OAUTH_TOKEN_PATH"
FALLBACK_GOOGLE_CLIENT_SECRET_ENV = "GMAIL_OAUTH_CLIENT_SECRET_PATH"
DEFAULT_GOOGLE_TOKEN_PATH = ".internal_documents_google_token.json"
GOOGLE_EXPORT_MAX_ATTEMPTS = 3
RETRYABLE_GOOGLE_STATUS_CODES = {429, 500, 502, 503, 504}

load_dotenv(LOCAL_DOTENV_PATH)
if FALLBACK_DOTENV_PATH.exists():
    load_dotenv(FALLBACK_DOTENV_PATH, override=False)


@dataclass(frozen=True)
class InternalDocument:
    """Resolved document content ready for embedding and storage."""

    document_id: str
    source_path: str
    source_kind: str
    title: str
    content: str
    metadata: dict[str, Any]


def _load_google_sdk() -> tuple[Any, Any, Any, Any]:
    """Import Google auth and Drive SDK dependencies lazily."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive dependencies are not installed for internal_documents_core. "
            "Run `python -m pip install -e .` inside internal_documents_core first."
        ) from exc
    return Credentials, Request, InstalledAppFlow, build


def resolve_source_dir(source_dir: str | Path | None = None) -> Path:
    """Resolve the source directory for internal documents."""
    raw_value = str(source_dir) if source_dir is not None else os.getenv(
        "INTERNAL_DOCUMENTS_SOURCE_DIR",
        "documents",
    )
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def collect_document_paths(source_dir: str | Path | None = None) -> list[Path]:
    """Collect supported documents recursively from a source directory."""
    resolved_source_dir = resolve_source_dir(source_dir)
    normalized_extensions = {extension.lower() for extension in SUPPORTED_DOCUMENT_EXTENSIONS}
    if not resolved_source_dir.exists():
        raise RuntimeError(f"Document source directory does not exist: {resolved_source_dir}")
    if not resolved_source_dir.is_dir():
        raise RuntimeError(f"Document source path is not a directory: {resolved_source_dir}")

    return sorted(
        path
        for path in resolved_source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def _resolve_path_from_env(raw_path: str | None) -> Path | None:
    """Resolve an env-configured path relative to the project root when needed."""
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_google_oauth_client_secret_path() -> Path:
    """Return the configured Google OAuth client secret path."""
    raw_path = os.getenv(GOOGLE_CLIENT_SECRET_ENV) or os.getenv(FALLBACK_GOOGLE_CLIENT_SECRET_ENV)
    path = _resolve_path_from_env(raw_path)
    if path is None:
        raise RuntimeError(
            f"{GOOGLE_CLIENT_SECRET_ENV} is not set and {FALLBACK_GOOGLE_CLIENT_SECRET_ENV} "
            "is also unavailable."
        )
    if not path.exists():
        raise RuntimeError(f"Google OAuth client secret file does not exist: {path}")
    return path


def get_google_oauth_token_path() -> Path:
    """Return the token path used for Google Drive document reads."""
    raw_path = os.getenv(GOOGLE_TOKEN_ENV) or DEFAULT_GOOGLE_TOKEN_PATH
    path = _resolve_path_from_env(raw_path)
    if path is None:
        raise RuntimeError(f"{GOOGLE_TOKEN_ENV} is not set.")
    return path


def load_google_credentials() -> Any:
    """Load, refresh, or bootstrap OAuth credentials for Google Drive reads."""
    Credentials, Request, InstalledAppFlow, _ = _load_google_sdk()
    token_path = get_google_oauth_token_path()
    client_secret_path = get_google_oauth_client_secret_path()
    credentials = None

    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), GOOGLE_DRIVE_SCOPES)

    if credentials and getattr(credentials, "valid", False):
        return credentials

    if credentials and getattr(credentials, "expired", False) and getattr(
        credentials, "refresh_token", None
    ):
        credentials.refresh(Request())
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")
        return credentials

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), GOOGLE_DRIVE_SCOPES)
    credentials = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def build_google_drive_service() -> Any:
    """Build a Google Drive API client for exporting document text."""
    _, _, _, build = _load_google_sdk()
    return build("drive", "v3", credentials=load_google_credentials(), cache_discovery=False)


def parse_google_workspace_pointer(path: Path) -> dict[str, Any]:
    """Parse a local Google Drive pointer file such as `.gdoc`."""
    return json.loads(path.read_text(encoding="utf-8"))


def is_retryable_google_error(exc: Exception) -> bool:
    """Return whether a Google API error looks transient and safe to retry."""
    status = getattr(getattr(exc, "resp", None), "status", None)
    return status in RETRYABLE_GOOGLE_STATUS_CODES


def export_google_doc_text(file_id: str, *, drive_service: Any | None = None) -> str:
    """Export a Google Doc as plain text using the Drive API."""
    service = drive_service or build_google_drive_service()
    for attempt in range(1, GOOGLE_EXPORT_MAX_ATTEMPTS + 1):
        try:
            exported_bytes = (
                service.files()
                .export(fileId=file_id, mimeType="text/plain")
                .execute()
            )
            if isinstance(exported_bytes, bytes):
                return exported_bytes.decode("utf-8", errors="replace")
            if isinstance(exported_bytes, str):
                return exported_bytes
            raise RuntimeError(
                f"Unexpected Google Drive export payload type: {type(exported_bytes)!r}"
            )
        except Exception as exc:
            if attempt >= GOOGLE_EXPORT_MAX_ATTEMPTS or not is_retryable_google_error(exc):
                raise
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"Could not export Google Doc text for file {file_id}.")


def load_document(path: Path, *, drive_service: Any | None = None) -> InternalDocument:
    """Load one supported document into a normalized text-bearing representation."""
    suffix = path.suffix.lower()
    title = path.stem

    if suffix == ".gdoc":
        pointer = parse_google_workspace_pointer(path)
        file_id = str(pointer["doc_id"])
        return InternalDocument(
            document_id=f"gdoc:{file_id}",
            source_path=str(path),
            source_kind="gdoc",
            title=title,
            content=export_google_doc_text(file_id, drive_service=drive_service),
            metadata={
                "doc_id": file_id,
                "resource_key": pointer.get("resource_key", ""),
                "email": pointer.get("email", ""),
            },
        )

    if suffix in {".md", ".txt"}:
        return InternalDocument(
            document_id=f"file:{path.resolve()}",
            source_path=str(path),
            source_kind="local_text",
            title=title,
            content=path.read_text(encoding="utf-8", errors="replace"),
            metadata={},
        )

    raise RuntimeError(f"Unsupported document type: {path}")
