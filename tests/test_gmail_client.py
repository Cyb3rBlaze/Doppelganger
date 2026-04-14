"""Tests for Gmail auth bootstrap and outbound send helpers."""

from __future__ import annotations

import base64
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.tools import gmail_client


class FakeCredentials:
    def __init__(
        self,
        *,
        valid: bool,
        expired: bool = False,
        refresh_token: str | None = None,
        token_json: str = '{"token": "abc"}',
    ) -> None:
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.token_json = token_json
        self.refreshed_with = None

    def refresh(self, request) -> None:
        self.refreshed_with = request
        self.valid = True
        self.expired = False

    def to_json(self) -> str:
        return self.token_json


class FakeRequest:
    pass


class FakeFlow:
    def __init__(self, credentials: FakeCredentials) -> None:
        self.credentials = credentials
        self.ports: list[int] = []

    def run_local_server(self, *, port: int) -> FakeCredentials:
        self.ports.append(port)
        return self.credentials


class FakeInstalledAppFlow:
    def __init__(self, flow: FakeFlow) -> None:
        self.flow = flow
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def from_client_secrets_file(self, path: str, scopes: tuple[str, ...]) -> FakeFlow:
        self.calls.append((path, scopes))
        return self.flow


class FakeCredentialsLoader:
    def __init__(self, credentials: FakeCredentials) -> None:
        self.credentials = credentials
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def from_authorized_user_file(
        self,
        path: str,
        scopes: tuple[str, ...],
    ) -> FakeCredentials:
        self.calls.append((path, scopes))
        return self.credentials


class FakeSendRequest:
    def __init__(self, response: dict[str, str]) -> None:
        self.response = response

    def execute(self) -> dict[str, str]:
        return self.response


class FakeMessagesResource:
    def __init__(self) -> None:
        self.user_id = None
        self.body = None

    def send(self, *, userId: str, body: dict[str, str]) -> FakeSendRequest:
        self.user_id = userId
        self.body = body
        return FakeSendRequest({"id": "sent-123"})


class FakeUsersResource:
    def __init__(self, messages_resource: FakeMessagesResource) -> None:
        self._messages_resource = messages_resource

    def messages(self) -> FakeMessagesResource:
        return self._messages_resource


class FakeGmailService:
    def __init__(self, messages_resource: FakeMessagesResource) -> None:
        self._messages_resource = messages_resource

    def users(self) -> FakeUsersResource:
        return FakeUsersResource(self._messages_resource)


@pytest.fixture(autouse=True)
def clear_gmail_service_cache() -> None:
    gmail_client.build_gmail_service.cache_clear()
    yield
    gmail_client.build_gmail_service.cache_clear()


def test_get_gmail_oauth_client_secret_path_resolves_relative_path(monkeypatch, tmp_path) -> None:
    secret_path = tmp_path / "oauth_secret.json"
    secret_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gmail_client, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_SECRET_PATH", "oauth_secret.json")

    resolved_path = gmail_client.get_gmail_oauth_client_secret_path()

    assert resolved_path == secret_path


def test_get_gmail_oauth_client_secret_path_raises_when_missing_file(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(gmail_client, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_SECRET_PATH", "oauth_secret.json")

    with pytest.raises(RuntimeError):
        gmail_client.get_gmail_oauth_client_secret_path()


def test_get_gmail_allowed_sender_domains_parses_csv(monkeypatch) -> None:
    monkeypatch.setenv("GMAIL_ALLOWED_SENDER_DOMAINS", "example.com, OpenAI.com , ,foo.dev")
    assert gmail_client.get_gmail_allowed_sender_domains() == {
        "example.com",
        "openai.com",
        "foo.dev",
    }


def test_load_gmail_credentials_reuses_valid_token(monkeypatch, tmp_path) -> None:
    token_path = tmp_path / ".gmail_token.json"
    token_path.write_text("{}", encoding="utf-8")
    secret_path = tmp_path / "oauth_secret.json"
    secret_path.write_text("{}", encoding="utf-8")
    credentials = FakeCredentials(valid=True)
    credentials_loader = FakeCredentialsLoader(credentials)
    fake_flow_loader = FakeInstalledAppFlow(FakeFlow(FakeCredentials(valid=True)))

    monkeypatch.setattr(gmail_client, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_SECRET_PATH", "oauth_secret.json")
    monkeypatch.setenv("GMAIL_OAUTH_TOKEN_PATH", ".gmail_token.json")
    monkeypatch.setattr(
        gmail_client,
        "_load_gmail_sdk",
        lambda: (credentials_loader, FakeRequest, fake_flow_loader, object()),
    )

    loaded_credentials = gmail_client.load_gmail_credentials()

    assert loaded_credentials is credentials
    assert credentials_loader.calls == [(str(token_path), gmail_client.GMAIL_SCOPES)]
    assert fake_flow_loader.calls == []


def test_load_gmail_credentials_refreshes_expired_token(monkeypatch, tmp_path) -> None:
    token_path = tmp_path / ".gmail_token.json"
    token_path.write_text("{}", encoding="utf-8")
    secret_path = tmp_path / "oauth_secret.json"
    secret_path.write_text("{}", encoding="utf-8")
    credentials = FakeCredentials(valid=False, expired=True, refresh_token="refresh-me")
    credentials_loader = FakeCredentialsLoader(credentials)
    fake_flow_loader = FakeInstalledAppFlow(FakeFlow(FakeCredentials(valid=True)))

    monkeypatch.setattr(gmail_client, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_SECRET_PATH", "oauth_secret.json")
    monkeypatch.setenv("GMAIL_OAUTH_TOKEN_PATH", ".gmail_token.json")
    monkeypatch.setattr(
        gmail_client,
        "_load_gmail_sdk",
        lambda: (credentials_loader, FakeRequest, fake_flow_loader, object()),
    )

    loaded_credentials = gmail_client.load_gmail_credentials()

    assert loaded_credentials is credentials
    assert isinstance(credentials.refreshed_with, FakeRequest)
    assert token_path.read_text(encoding="utf-8") == credentials.to_json()
    assert fake_flow_loader.calls == []


def test_load_gmail_credentials_runs_oauth_flow_when_token_missing(
    monkeypatch,
    tmp_path,
) -> None:
    secret_path = tmp_path / "oauth_secret.json"
    secret_path.write_text("{}", encoding="utf-8")
    credentials = FakeCredentials(valid=True, token_json='{"token": "fresh"}')
    credentials_loader = FakeCredentialsLoader(FakeCredentials(valid=False))
    flow = FakeFlow(credentials)
    fake_flow_loader = FakeInstalledAppFlow(flow)

    monkeypatch.setattr(gmail_client, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_SECRET_PATH", "oauth_secret.json")
    monkeypatch.setenv("GMAIL_OAUTH_TOKEN_PATH", ".gmail_token.json")
    monkeypatch.setattr(
        gmail_client,
        "_load_gmail_sdk",
        lambda: (credentials_loader, FakeRequest, fake_flow_loader, object()),
    )

    loaded_credentials = gmail_client.load_gmail_credentials()

    assert loaded_credentials is credentials
    assert fake_flow_loader.calls == [(str(secret_path), gmail_client.GMAIL_SCOPES)]
    assert flow.ports == [0]
    assert (tmp_path / ".gmail_token.json").read_text(encoding="utf-8") == credentials.to_json()


def test_build_gmail_service_uses_loaded_credentials(monkeypatch) -> None:
    fake_credentials = object()
    build_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return "gmail-service"

    monkeypatch.setattr(gmail_client, "load_gmail_credentials", lambda: fake_credentials)
    monkeypatch.setattr(
        gmail_client,
        "_load_gmail_sdk",
        lambda: (object(), object(), object(), fake_build),
    )

    service = gmail_client.build_gmail_service()

    assert service == "gmail-service"
    assert build_calls == [
        (
            ("gmail", "v1"),
            {"credentials": fake_credentials, "cache_discovery": False},
        )
    ]


def test_outbound_email_requires_at_least_one_recipient() -> None:
    with pytest.raises(ValidationError):
        gmail_client.OutboundEmail(subject="Hello", body_text="Body")


def test_build_gmail_send_body_encodes_to_cc_bcc_and_subject() -> None:
    email = gmail_client.OutboundEmail(
        to=["to@example.com"],
        cc=["cc@example.com"],
        bcc=["bcc@example.com"],
        subject="Launch",
        body_text="Ship it",
        thread_id="thread-123",
        from_email="me@example.com",
    )

    send_body = gmail_client.build_gmail_send_body(email)
    raw_bytes = base64.urlsafe_b64decode(send_body["raw"].encode("utf-8"))
    parsed_message = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    assert send_body["threadId"] == "thread-123"
    assert parsed_message["From"] == "me@example.com"
    assert parsed_message["To"] == "to@example.com"
    assert parsed_message["Cc"] == "cc@example.com"
    assert parsed_message["Bcc"] == "bcc@example.com"
    assert parsed_message["Subject"] == "Launch"
    assert "Ship it" in parsed_message.get_body(preferencelist=("plain",)).get_content()


def test_send_gmail_message_calls_messages_send_with_me_user_id() -> None:
    messages_resource = FakeMessagesResource()
    service = FakeGmailService(messages_resource)
    email = gmail_client.OutboundEmail(
        to=["to@example.com"],
        cc=["cc@example.com"],
        bcc=["bcc@example.com"],
        subject="Hello",
        body_text="Body",
    )

    response = gmail_client.send_gmail_message(email, service=service)

    assert response == {"id": "sent-123"}
    assert messages_resource.user_id == "me"
    assert isinstance(messages_resource.body, dict)
    assert "raw" in messages_resource.body
