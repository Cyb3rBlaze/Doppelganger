"""Tests for the OpenAI agent service helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.core.models import Message
from app.services import openai_agent


class FakeStreamResult:
    def __init__(self, events: list[object], final_output: object) -> None:
        self._events = events
        self.final_output = final_output

    async def stream_events(self):
        for event in self._events:
            yield event


class FakeRunner:
    @staticmethod
    def run_streamed(agent, input_text):
        _ = agent, input_text
        return FakeStreamResult([], "final")


def fake_function_tool(**tool_kwargs):
    def decorator(func):
        func._tool_kwargs = tool_kwargs
        return func

    return decorator


@pytest.fixture(autouse=True)
def clear_openai_agent_caches():
    openai_agent.get_agent.cache_clear()
    openai_agent.get_summary_agent.cache_clear()
    openai_agent.load_mind_instructions.cache_clear()
    yield
    openai_agent.get_agent.cache_clear()
    openai_agent.get_summary_agent.cache_clear()
    openai_agent.load_mind_instructions.cache_clear()


def test_load_mind_instructions_reads_files_in_order(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        mind_dir = Path(tempdir)
        (mind_dir / "SOUL.md").write_text("soul", encoding="utf-8")
        (mind_dir / "DIRECTIVES.md").write_text("directives", encoding="utf-8")

        monkeypatch.setattr(openai_agent, "MIND_DIR", mind_dir)
        content = openai_agent.load_mind_instructions()

    assert content == "soul\n\ndirectives"


def test_load_mind_instructions_raises_when_file_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        mind_dir = Path(tempdir)
        (mind_dir / "SOUL.md").write_text("soul", encoding="utf-8")

        monkeypatch.setattr(openai_agent, "MIND_DIR", mind_dir)
        with pytest.raises(RuntimeError):
            openai_agent.load_mind_instructions()


def test_build_agent_input_includes_message_fields() -> None:
    message = Message(
        channel="api",
        user_id="anshul",
        text="hello",
        conversation_id="thread-1",
        message_id="msg-1",
    )

    input_text = openai_agent.build_agent_input(message)

    assert "Channel: api" in input_text
    assert "User ID: anshul" in input_text
    assert "Conversation ID: thread-1" in input_text
    assert "Message ID: msg-1" in input_text
    assert "Message:\nhello" in input_text


def test_build_agent_input_includes_history_and_prior_summaries() -> None:
    message = Message(
        channel="api",
        user_id="anshul",
        text="latest message",
        conversation_id="thread-1",
        message_id="msg-2",
    )

    input_text = openai_agent.build_agent_input(
        message,
        current_session_history=[
            {"direction": "inbound", "text": "hello"},
            {"direction": "outbound", "text": "hi there"},
        ],
        current_session_summary="User is planning travel.",
        previous_session_summaries=["Earlier session was about planning travel."],
    )

    assert "Relevant summaries from previous sessions:" in input_text
    assert "Earlier session was about planning travel." in input_text
    assert "Current session summary so far:" in input_text
    assert "User is planning travel." in input_text
    assert "Recent current session history:" in input_text
    assert "- [inbound] hello" in input_text
    assert "- [outbound] hi there" in input_text


def test_build_agent_input_includes_retrieved_documents() -> None:
    message = Message(
        channel="api",
        user_id="anshul",
        text="latest message",
        conversation_id="thread-1",
        message_id="msg-2",
    )

    input_text = openai_agent.build_agent_input(
        message,
        retrieved_documents=[
            {
                "title": "Investing Notes",
                "source_path": "/docs/investing.gdoc",
                "score": 0.91,
                "content": "Key investing principles",
            }
        ],
    )

    assert "Relevant retrieved internal documents:" in input_text
    assert "Title: Investing Notes" in input_text
    assert "Source path: /docs/investing.gdoc" in input_text
    assert "Similarity score: 0.91" in input_text
    assert "Content:\nKey investing principles" in input_text


def test_build_agent_input_excludes_duplicate_latest_inbound_from_history() -> None:
    message = Message(
        channel="api",
        user_id="anshul",
        text="latest message",
        conversation_id="thread-1",
        message_id="msg-2",
    )

    input_text = openai_agent.build_agent_input(
        message,
        current_session_history=[
            {"direction": "inbound", "text": "hello", "message_id": "msg-1"},
            {"direction": "inbound", "text": "latest message", "message_id": "msg-2"},
        ],
    )

    assert "- [inbound] hello" in input_text
    assert "- [inbound] latest message" not in input_text


def test_build_session_summary_input_includes_existing_summary_and_recent_updates() -> None:
    message = Message(
        channel="telegram",
        user_id="6891176979",
        text="latest message",
        conversation_id="thread-1",
    )

    input_text = openai_agent.build_session_summary_input(
        message,
        existing_session_summary="User is coordinating an email draft.",
        current_session_history=[
            {"direction": "inbound", "text": "Can you draft an email?"},
            {"direction": "outbound", "text": "Yes, who should it go to?"},
        ],
    )

    assert "Summarize this daily conversation session for future context." in input_text
    assert "Channel: telegram" in input_text
    assert "Conversation ID: thread-1" in input_text
    assert "Existing session summary:" in input_text
    assert "User is coordinating an email draft." in input_text
    assert "Recent session updates:" in input_text
    assert "- [inbound] Can you draft an email?" in input_text
    assert "- [outbound] Yes, who should it go to?" in input_text


def test_select_summary_context_history_trims_to_recent_events() -> None:
    selected = openai_agent._select_summary_context_history(
        current_session_history=[
            {"direction": "inbound", "text": "1"},
            {"direction": "outbound", "text": "2"},
            {"direction": "inbound", "text": "3"},
        ],
        limit=2,
    )

    assert selected == [
        {"direction": "outbound", "text": "2"},
        {"direction": "inbound", "text": "3"},
    ]


def test_truncate_handles_strings_and_objects() -> None:
    assert openai_agent._truncate("abc") == "abc"
    assert openai_agent._truncate({"a": 1}) == '{"a": 1}'
    assert openai_agent._truncate("x" * 400).endswith("...")


def test_extract_tool_call_fields() -> None:
    item = SimpleNamespace(
        raw_item=SimpleNamespace(name="search", call_id="call-1", arguments='{"q":"hi"}')
    )
    tool_name, call_id, arguments = openai_agent._extract_tool_call_fields(item)
    assert tool_name == "search"
    assert call_id == "call-1"
    assert '"q":"hi"' in arguments


def test_extract_tool_output_fields() -> None:
    item = SimpleNamespace(
        raw_item=SimpleNamespace(call_id="call-1"),
        output={"result": "done"},
    )
    call_id, output = openai_agent._extract_tool_output_fields(item)
    assert call_id == "call-1"
    assert '"result": "done"' in output


def test_extract_reasoning_text_from_summary() -> None:
    item = SimpleNamespace(
        raw_item=SimpleNamespace(summary=[SimpleNamespace(text="step one")])
    )
    assert openai_agent._extract_reasoning_text(item) == "step one"


def test_log_stream_event_logs_expected_events(caplog) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    events = [
        SimpleNamespace(
            type="agent_updated_stream_event",
            new_agent=SimpleNamespace(name="Tester"),
        ),
        SimpleNamespace(
            type="run_item_stream_event",
            name="reasoning_item_created",
            item=SimpleNamespace(raw_item=SimpleNamespace(summary=[SimpleNamespace(text="think")])),
        ),
        SimpleNamespace(
            type="run_item_stream_event",
            name="tool_called",
            item=SimpleNamespace(
                raw_item=SimpleNamespace(name="search", call_id="call-1", arguments='{"q":"hi"}')
            ),
        ),
        SimpleNamespace(
            type="run_item_stream_event",
            name="tool_output",
            item=SimpleNamespace(raw_item=SimpleNamespace(call_id="call-1"), output="done"),
        ),
        SimpleNamespace(
            type="run_item_stream_event",
            name="message_output_created",
            item=None,
        ),
    ]

    with caplog.at_level("INFO", logger="doppelganger.server.agent"):
        for event in events:
            openai_agent.log_stream_event(event, message=message)

    joined = "\n".join(caplog.messages)
    assert "status=agent_updated" in joined
    assert "status=reasoning" in joined
    assert "status=tool_called" in joined
    assert "status=tool_output" in joined
    assert "status=model_message_created" in joined


def test_get_agent_uses_env_and_loaded_instructions(monkeypatch) -> None:
    fake_agent_cls = lambda **kwargs: kwargs
    monkeypatch.setattr(
        openai_agent,
        "_load_agents_sdk",
        lambda: (fake_agent_cls, FakeRunner, fake_function_tool),
    )
    monkeypatch.setattr(openai_agent, "load_mind_instructions", lambda: "mind text")
    monkeypatch.setattr(
        openai_agent,
        "build_agent_tools",
        lambda function_tool: ["gmail-tool"],
    )
    monkeypatch.setenv("ASSISTANT_NAME", "Mirror")
    monkeypatch.setenv("ASSISTANT_MODEL", "gpt-test")

    agent = openai_agent.get_agent()

    assert agent["name"] == "Mirror"
    assert agent["instructions"] == "mind text"
    assert agent["model"] == "gpt-test"
    assert agent["tools"] == ["gmail-tool"]


def test_get_summary_agent_uses_summary_settings(monkeypatch) -> None:
    fake_agent_cls = lambda **kwargs: kwargs
    monkeypatch.setattr(
        openai_agent,
        "_load_agents_sdk",
        lambda: (fake_agent_cls, FakeRunner, fake_function_tool),
    )
    monkeypatch.setenv("ASSISTANT_MODEL", "gpt-summary")

    agent = openai_agent.get_summary_agent()

    assert agent["name"] == "Session Summarizer"
    assert "concise factual summary" in agent["instructions"]
    assert agent["model"] == "gpt-summary"
    assert agent["tools"] == []


async def test_generate_reply_streams_events_and_returns_string_output(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    stream = FakeStreamResult(
        [
            SimpleNamespace(
                type="run_item_stream_event",
                name="message_output_created",
                item=None,
            )
        ],
        "hello back",
    )

    monkeypatch.setattr(
        openai_agent,
        "_load_agents_sdk",
        lambda: (
            object(),
            SimpleNamespace(run_streamed=lambda *_args, **_kwargs: stream),
            fake_function_tool,
        ),
    )
    monkeypatch.setattr(openai_agent, "get_agent", lambda: "agent")
    logged_events: list[object] = []
    monkeypatch.setattr(
        openai_agent,
        "log_stream_event",
        lambda event, *, message: logged_events.append((event, message)),
    )

    result = await openai_agent.generate_reply(
        message,
        current_session_summary="summary",
    )

    assert result == "hello back"
    assert len(logged_events) == 1


async def test_generate_reply_casts_non_string_output(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    stream = FakeStreamResult([], {"reply": "hello"})

    monkeypatch.setattr(
        openai_agent,
        "_load_agents_sdk",
        lambda: (
            object(),
            SimpleNamespace(run_streamed=lambda *_args, **_kwargs: stream),
            fake_function_tool,
        ),
    )
    monkeypatch.setattr(openai_agent, "get_agent", lambda: "agent")

    result = await openai_agent.generate_reply(message)

    assert result == "{'reply': 'hello'}"


async def test_generate_session_summary_returns_trimmed_output(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    stream = FakeStreamResult([], " summary text \n")

    monkeypatch.setattr(
        openai_agent,
        "_load_agents_sdk",
        lambda: (
            object(),
            SimpleNamespace(run_streamed=lambda *_args, **_kwargs: stream),
            fake_function_tool,
        ),
    )
    monkeypatch.setattr(openai_agent, "get_summary_agent", lambda: "summary-agent")

    result = await openai_agent.generate_session_summary(
        message,
        current_session_history=[{"direction": "inbound", "text": "hello"}],
    )

    assert result == "summary text"
