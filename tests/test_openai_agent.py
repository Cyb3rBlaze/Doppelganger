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


@pytest.fixture(autouse=True)
def clear_openai_agent_caches():
    openai_agent.get_agent.cache_clear()
    openai_agent.load_mind_instructions.cache_clear()
    yield
    openai_agent.get_agent.cache_clear()
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
    monkeypatch.setattr(openai_agent, "_load_agents_sdk", lambda: (fake_agent_cls, FakeRunner))
    monkeypatch.setattr(openai_agent, "load_mind_instructions", lambda: "mind text")
    monkeypatch.setenv("ASSISTANT_NAME", "Mirror")
    monkeypatch.setenv("ASSISTANT_MODEL", "gpt-test")

    agent = openai_agent.get_agent()

    assert agent["name"] == "Mirror"
    assert agent["instructions"] == "mind text"
    assert agent["model"] == "gpt-test"


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
        lambda: (object(), SimpleNamespace(run_streamed=lambda *_args, **_kwargs: stream)),
    )
    monkeypatch.setattr(openai_agent, "get_agent", lambda: "agent")
    logged_events: list[object] = []
    monkeypatch.setattr(
        openai_agent,
        "log_stream_event",
        lambda event, *, message: logged_events.append((event, message)),
    )

    result = await openai_agent.generate_reply(message)

    assert result == "hello back"
    assert len(logged_events) == 1


async def test_generate_reply_casts_non_string_output(monkeypatch) -> None:
    message = Message(channel="api", user_id="anshul", text="hello")
    stream = FakeStreamResult([], {"reply": "hello"})

    monkeypatch.setattr(
        openai_agent,
        "_load_agents_sdk",
        lambda: (object(), SimpleNamespace(run_streamed=lambda *_args, **_kwargs: stream)),
    )
    monkeypatch.setattr(openai_agent, "get_agent", lambda: "agent")

    result = await openai_agent.generate_reply(message)

    assert result == "{'reply': 'hello'}"
