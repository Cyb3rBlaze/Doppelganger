"""Tests for Agents SDK tool registration."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools import agent_tools


def fake_function_tool(**tool_kwargs):
    def decorator(func):
        func._tool_kwargs = tool_kwargs
        return func

    return decorator


def test_build_agent_tools_exposes_gmail_send_tool(monkeypatch) -> None:
    tool_calls: list[object] = []

    def fake_send_gmail_message(email):
        tool_calls.append(email)
        return {"id": "gmail-123", "threadId": "thread-1", "labelIds": ["SENT"]}

    monkeypatch.setattr(agent_tools.gmail_client, "send_gmail_message", fake_send_gmail_message)

    tools = agent_tools.build_agent_tools(fake_function_tool)

    assert len(tools) == 9
    send_gmail = tools[0]
    result = send_gmail(
        to=["to@example.com"],
        cc=["cc@example.com"],
        bcc=["bcc@example.com"],
        subject="Hello",
        body_text="Body",
        thread_id="thread-1",
        from_email="me@example.com",
    )

    assert send_gmail._tool_kwargs["name_override"] == "send_gmail"
    assert result == {
        "status": "sent",
        "id": "gmail-123",
        "threadId": "thread-1",
        "labelIds": ["SENT"],
    }
    assert len(tool_calls) == 1
    email = tool_calls[0]
    assert email.to == ["to@example.com"]
    assert email.cc == ["cc@example.com"]
    assert email.bcc == ["bcc@example.com"]
    assert email.subject == "Hello"
    assert email.body_text == "Body"


def test_build_agent_tools_exposes_gmail_read_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_tools.gmail_client,
        "read_gmail_messages",
        lambda query=None, max_results=5: [
            {"id": "msg-1", "subject": "Hello", "body_text": "Body"}
        ],
    )

    tools = agent_tools.build_agent_tools(fake_function_tool)

    read_gmail = tools[1]
    result = read_gmail(query="from:boss@example.com", max_results=1)

    assert read_gmail._tool_kwargs["name_override"] == "read_gmail"
    assert result == {
        "status": "ok",
        "messages": [{"id": "msg-1", "subject": "Hello", "body_text": "Body"}],
        "count": 1,
    }


def test_build_agent_tools_exposes_internal_documents_search_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_tools.internal_documents,
        "search_internal_documents_for_query",
        lambda query, limit=3: [
            {
                "document_id": "gdoc:abc123",
                "title": "Investing Notes",
                "source_path": "/docs/investing.gdoc",
                "score": 0.91,
                "content": "Key investing principles",
            }
        ],
    )

    tools = agent_tools.build_agent_tools(fake_function_tool)

    search_internal_documents = tools[2]
    result = search_internal_documents(query="investing notes", max_results=3)

    assert search_internal_documents._tool_kwargs["name_override"] == "search_internal_documents"
    assert result == {
        "status": "ok",
        "documents": [
            {
                "document_id": "gdoc:abc123",
                "title": "Investing Notes",
                "source_path": "/docs/investing.gdoc",
                "score": 0.91,
                "content": "Key investing principles",
            }
        ],
        "count": 1,
    }


def test_build_agent_tools_exposes_read_file_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_tools.file_tools,
        "read_file",
        lambda path, max_chars=agent_tools.file_tools.MAX_READ_CHARS: {
            "status": "ok",
            "path": "/tmp/test.md",
            "content": "hello",
            "truncated": False,
        },
    )

    tools = agent_tools.build_agent_tools(fake_function_tool)

    read_file = tools[3]
    result = read_file(path="mind/SOUL.md", max_chars=500)

    assert read_file._tool_kwargs["name_override"] == "read_file"
    assert result["content"] == "hello"


def test_build_agent_tools_exposes_get_file_info_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_tools.file_tools,
        "get_file_info",
        lambda path: {
            "status": "ok",
            "path": "/tmp/test.md",
            "bytes": 12,
            "char_count": 12,
            "line_count": 2,
            "sha256": "abc123",
        },
    )

    tools = agent_tools.build_agent_tools(fake_function_tool)

    get_file_info = tools[4]
    result = get_file_info(path="mind/DIRECTIVES.md")

    assert get_file_info._tool_kwargs["name_override"] == "get_file_info"
    assert result == {
        "status": "ok",
        "path": "/tmp/test.md",
        "bytes": 12,
        "char_count": 12,
        "line_count": 2,
        "sha256": "abc123",
    }


def test_build_agent_tools_exposes_read_file_window_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_tools.file_tools,
        "read_file_window",
        lambda path, start_line, end_line: {
            "status": "ok",
            "path": "/tmp/test.md",
            "content": "window",
            "start_line": start_line,
            "end_line": end_line,
        },
    )

    tools = agent_tools.build_agent_tools(fake_function_tool)

    read_file_window = tools[5]
    result = read_file_window(path="mind/SOUL.md", start_line=10, end_line=20)

    assert read_file_window._tool_kwargs["name_override"] == "read_file_window"
    assert result["content"] == "window"


def test_build_agent_tools_exposes_search_in_file_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_tools.file_tools,
        "search_in_file",
        lambda path, query, case_sensitive=False, context_lines=2, max_matches=20: {
            "status": "ok",
            "path": "/tmp/test.md",
            "query": query,
            "match_count": 1,
            "matches": [{"line_number": 3, "snippet": "matched text"}],
            "truncated": False,
        },
    )

    tools = agent_tools.build_agent_tools(fake_function_tool)

    search_in_file = tools[6]
    result = search_in_file(path="mind/DIRECTIVES.md", query="matched")

    assert search_in_file._tool_kwargs["name_override"] == "search_in_file"
    assert result["match_count"] == 1
    assert result["matches"][0]["line_number"] == 3


def test_build_agent_tools_exposes_write_file_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_tools.file_tools,
        "write_file",
        lambda path, content, append=False: {
            "status": "ok",
            "path": "/tmp/test.md",
            "bytes_written": len(content.encode("utf-8")),
            "append": append,
        },
    )

    tools = agent_tools.build_agent_tools(fake_function_tool)

    write_file = tools[7]
    result = write_file(path="mind/DIRECTIVES.md", content="updated", append=False)

    assert write_file._tool_kwargs["name_override"] == "write_file"
    assert result == {
        "status": "ok",
        "path": "/tmp/test.md",
        "bytes_written": len("updated".encode("utf-8")),
        "append": False,
    }


def test_build_agent_tools_exposes_replace_in_file_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_tools.file_tools,
        "replace_in_file",
        lambda path, old_text, new_text, expected_hash=None: {
            "status": "ok",
            "path": "/tmp/test.md",
            "replacements": 1,
            "old_sha256": expected_hash,
            "new_sha256": "newhash",
            "bytes_written": 42,
        },
    )

    tools = agent_tools.build_agent_tools(fake_function_tool)

    replace_in_file = tools[8]
    result = replace_in_file(
        path="mind/SOUL.md",
        old_text="old",
        new_text="new",
        expected_hash="oldhash",
    )

    assert replace_in_file._tool_kwargs["name_override"] == "replace_in_file"
    assert result["replacements"] == 1
    assert result["old_sha256"] == "oldhash"
