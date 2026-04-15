"""Gmail OAuth and send helpers for the AI doppelganger."""

from __future__ import annotations

import base64
import os
from email.message import EmailMessage
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = REPO_ROOT / ".env"

GMAIL_OAUTH_CLIENT_SECRET_PATH_ENV = "GMAIL_OAUTH_CLIENT_SECRET_PATH"
GMAIL_OAUTH_TOKEN_PATH_ENV = "GMAIL_OAUTH_TOKEN_PATH"
GMAIL_ALLOWED_SENDER_DOMAINS_ENV = "GMAIL_ALLOWED_SENDER_DOMAINS"

GMAIL_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
)

load_dotenv(DOTENV_PATH)


class OutboundEmail(BaseModel):
    """Plain-text outbound email payload for Gmail sending."""

    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    bcc: list[str] = Field(default_factory=list)
    subject: str = ""
    body_text: str = Field(..., min_length=1)
    thread_id: str | None = None
    from_email: str | None = None

    @model_validator(mode="after")
    def validate_recipients(self) -> "OutboundEmail":
        if not (self.to or self.cc or self.bcc):
            raise ValueError("At least one Gmail recipient must be provided.")
        return self


def _load_gmail_sdk() -> tuple[Any, Any, Any, Any]:
    """Import Google auth and Gmail SDK dependencies lazily."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Gmail dependencies are not installed. Run `pip install -e .` first."
        ) from exc
    return Credentials, Request, InstalledAppFlow, build


def _resolve_project_path(raw_path: str) -> Path:
    """Resolve an env-configured path relative to the repo root when needed."""
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def get_gmail_oauth_client_secret_path() -> Path:
    """Return the configured Gmail OAuth client secret path."""
    raw_path = os.getenv(GMAIL_OAUTH_CLIENT_SECRET_PATH_ENV)
    if not raw_path:
        raise RuntimeError(f"{GMAIL_OAUTH_CLIENT_SECRET_PATH_ENV} is not set.")
    path = _resolve_project_path(raw_path)
    if not path.exists():
        raise RuntimeError(f"Gmail OAuth client secret file does not exist: {path}")
    return path


def get_gmail_oauth_token_path() -> Path:
    """Return the configured Gmail OAuth token cache path."""
    raw_path = os.getenv(GMAIL_OAUTH_TOKEN_PATH_ENV)
    if not raw_path:
        raise RuntimeError(f"{GMAIL_OAUTH_TOKEN_PATH_ENV} is not set.")
    return _resolve_project_path(raw_path)


def get_gmail_allowed_sender_domains() -> set[str]:
    """Return the configured allowed Gmail sender domains for later adapter checks."""
    raw_value = os.getenv(GMAIL_ALLOWED_SENDER_DOMAINS_ENV, "")
    return {
        domain.strip().lower()
        for domain in raw_value.split(",")
        if domain.strip()
    }


def _write_token_file(path: Path, credentials: Any) -> None:
    """Persist Gmail OAuth credentials for reuse across local runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credentials.to_json(), encoding="utf-8")


def load_gmail_credentials() -> Any:
    """Load, refresh, or bootstrap local Gmail OAuth credentials."""
    Credentials, Request, InstalledAppFlow, _ = _load_gmail_sdk()

    token_path = get_gmail_oauth_token_path()
    client_secret_path = get_gmail_oauth_client_secret_path()
    credentials = None

    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)

    if credentials and getattr(credentials, "valid", False):
        return credentials

    if credentials and getattr(credentials, "expired", False) and getattr(
        credentials, "refresh_token", None
    ):
        credentials.refresh(Request())
        _write_token_file(token_path, credentials)
        return credentials

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), GMAIL_SCOPES)
    credentials = flow.run_local_server(port=0)
    _write_token_file(token_path, credentials)
    return credentials


@lru_cache
def build_gmail_service() -> Any:
    """Build and cache the Gmail API service for the local user."""
    _, _, _, build = _load_gmail_sdk()
    credentials = load_gmail_credentials()
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def build_gmail_mime_message(email: OutboundEmail) -> EmailMessage:
    """Build a plain-text MIME email for Gmail sending."""
    message = EmailMessage()
    if email.from_email:
        message["From"] = email.from_email
    if email.to:
        message["To"] = ", ".join(email.to)
    if email.cc:
        message["Cc"] = ", ".join(email.cc)
    if email.bcc:
        message["Bcc"] = ", ".join(email.bcc)
    message["Subject"] = email.subject
    message.set_content(email.body_text)
    return message


def build_gmail_send_body(email: OutboundEmail) -> dict[str, Any]:
    """Build the Gmail API body for `users.messages.send`."""
    encoded_message = base64.urlsafe_b64encode(
        build_gmail_mime_message(email).as_bytes()
    ).decode()
    payload: dict[str, Any] = {"raw": encoded_message}
    if email.thread_id:
        payload["threadId"] = email.thread_id
    return payload


def send_gmail_message(email: OutboundEmail, *, service: Any | None = None) -> dict[str, Any]:
    """Send one outbound Gmail message using the configured Gmail API client."""
    gmail_service = service or build_gmail_service()
    return (
        gmail_service.users()
        .messages()
        .send(userId="me", body=build_gmail_send_body(email))
        .execute()
    )


def _decode_gmail_message_data(data: str | None) -> str:
    """Decode a Gmail API body payload into plain text."""
    if not data:
        return ""
    padding = "=" * (-len(data) % 4)
    decoded = base64.urlsafe_b64decode((data + padding).encode("utf-8"))
    return decoded.decode("utf-8", errors="replace")


def _payload_headers_map(payload: dict[str, Any] | None) -> dict[str, str]:
    """Normalize Gmail payload headers into a lowercase lookup map."""
    if not payload:
        return {}
    return {
        str(header.get("name", "")).lower(): str(header.get("value", ""))
        for header in payload.get("headers", [])
        if header.get("name")
    }


def _extract_plain_text_from_payload(payload: dict[str, Any] | None) -> str:
    """Extract the best plain-text body from a Gmail message payload."""
    if not payload:
        return ""

    mime_type = payload.get("mimeType")
    body = payload.get("body", {})
    if mime_type == "text/plain":
        return _decode_gmail_message_data(body.get("data"))

    for part in payload.get("parts", []) or []:
        text = _extract_plain_text_from_payload(part)
        if text:
            return text

    if mime_type == "text/html":
        return _decode_gmail_message_data(body.get("data"))

    return _decode_gmail_message_data(body.get("data"))


def normalize_gmail_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Gmail API message into a compact agent-friendly dict."""
    payload = message.get("payload", {})
    headers = _payload_headers_map(payload)
    return {
        "id": message.get("id"),
        "threadId": message.get("threadId"),
        "labelIds": message.get("labelIds", []),
        "snippet": message.get("snippet", ""),
        "internalDate": message.get("internalDate"),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "subject": headers.get("subject", ""),
        "body_text": _extract_plain_text_from_payload(payload),
    }


def get_gmail_message(message_id: str, *, service: Any | None = None) -> dict[str, Any]:
    """Fetch and normalize one Gmail message by id."""
    gmail_service = service or build_gmail_service()
    message = (
        gmail_service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )
    return normalize_gmail_message(message)


def read_gmail_messages(
    *,
    query: str | None = None,
    max_results: int = 5,
    service: Any | None = None,
) -> list[dict[str, Any]]:
    """List and fetch recent Gmail messages for agent read access."""
    gmail_service = service or build_gmail_service()
    list_call = gmail_service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results,
    )
    response = list_call.execute()
    messages = response.get("messages", [])
    return [
        get_gmail_message(message["id"], service=gmail_service)
        for message in messages
        if message.get("id")
    ]


def run_gmail_auth() -> None:
    """Run the local Gmail OAuth flow and persist a fresh token if needed."""
    token_path = get_gmail_oauth_token_path()
    load_gmail_credentials()
    print(f"Gmail auth ready. Token path: {token_path}")


def main() -> None:
    """Run standalone Gmail auth bootstrap."""
    run_gmail_auth()


if __name__ == "__main__":
    main()
