"""Tests for Agents SDK tool registration."""

from __future__ import annotations

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

    assert len(tools) == 1
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
