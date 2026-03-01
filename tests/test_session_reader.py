"""Tests for the session reader module."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from betty.models import Session, ToolApproval, ToolCall, Turn
from betty.session_reader import (
    SessionWatcher,
    decode_project_path,
    discover_sessions,
    encode_project_path,
    parse_session,
)


# --- Test data helpers ---


def make_user_entry(content: str, timestamp: str = "2026-01-15T10:00:00Z") -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "timestamp": timestamp,
        "sessionId": "test-session-id",
        "cwd": "/Users/test/project",
        "gitBranch": "main",
    }


def make_assistant_entry(
    text: str = "",
    tool_calls: list[dict] | None = None,
    timestamp: str = "2026-01-15T10:01:00Z",
    model: str = "claude-opus-4-6",
) -> dict:
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if tool_calls:
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.get("id", "tool-1"),
                "name": tc.get("name", "Read"),
                "input": tc.get("input", {}),
            })
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": model,
            "content": content,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
            },
        },
        "timestamp": timestamp,
    }


def make_tool_result_entry(
    tool_use_id: str,
    output: str = "result text",
    is_error: bool = False,
) -> dict:
    return {
        "type": "tool_result",
        "message": {
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": output,
                    "is_error": is_error,
                }
            ],
            "is_error": is_error,
        },
    }


def write_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# --- Path encoding tests ---


class TestPathEncoding:
    def test_encode_project_path(self):
        assert encode_project_path("/Users/foo/bar") == "-Users-foo-bar"

    def test_encode_strips_leading_slash(self):
        assert encode_project_path("/tmp/test") == "-tmp-test"

    def test_decode_project_path(self):
        assert decode_project_path("-Users-foo-bar") == "/Users/foo/bar"

    def test_roundtrip_simple(self):
        """Roundtrip works for paths without hyphens."""
        original = "/Users/akash/projects/myapp"
        assert decode_project_path(encode_project_path(original)) == original

    def test_encode_matches_claude_code(self):
        """Encoding matches Claude Code's actual format."""
        assert encode_project_path("/Users/akash/cursor-projects/betty-2") == (
            "-Users-akash-cursor-projects-betty-2"
        )


# --- Session parsing tests ---


class TestParseSession:
    def test_parse_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        session = parse_session(path)
        assert session.session_id == "empty"
        assert session.turns == []

    def test_parse_user_message(self, tmp_path):
        path = tmp_path / "session.jsonl"
        write_jsonl(path, [
            make_user_entry("Hello, help me with my code"),
        ])
        session = parse_session(path)
        assert len(session.turns) == 1
        assert session.turns[0].role == "user"
        assert session.turns[0].content == "Hello, help me with my code"

    def test_parse_assistant_text(self, tmp_path):
        path = tmp_path / "session.jsonl"
        write_jsonl(path, [
            make_user_entry("Hello"),
            make_assistant_entry(text="I can help with that!"),
        ])
        session = parse_session(path)
        assert len(session.turns) == 2
        assert session.turns[1].role == "assistant"
        assert session.turns[1].content == "I can help with that!"
        assert session.turns[1].model == "claude-opus-4-6"

    def test_parse_tool_calls(self, tmp_path):
        path = tmp_path / "session.jsonl"
        write_jsonl(path, [
            make_user_entry("Read my file"),
            make_assistant_entry(
                text="Let me read that file.",
                tool_calls=[{
                    "id": "tc-1",
                    "name": "Read",
                    "input": {"file_path": "/src/main.py"},
                }],
            ),
            make_tool_result_entry("tc-1", output="print('hello')"),
        ])
        session = parse_session(path)
        assert len(session.turns) == 2
        assistant_turn = session.turns[1]
        assert len(assistant_turn.tool_calls) == 1

        tc = assistant_turn.tool_calls[0]
        assert tc.tool_name == "Read"
        assert tc.input == {"file_path": "/src/main.py"}
        assert tc.output == "print('hello')"
        assert tc.success is True
        assert tc.file_path == "/src/main.py"

    def test_parse_tool_error(self, tmp_path):
        path = tmp_path / "session.jsonl"
        write_jsonl(path, [
            make_user_entry("Run a command"),
            make_assistant_entry(
                tool_calls=[{
                    "id": "tc-2",
                    "name": "Bash",
                    "input": {"command": "cat nonexistent.txt"},
                }],
            ),
            make_tool_result_entry("tc-2", output="No such file", is_error=True),
        ])
        session = parse_session(path)
        tc = session.turns[1].tool_calls[0]
        assert tc.success is False
        assert tc.command == "cat nonexistent.txt"

    def test_parse_token_usage(self, tmp_path):
        path = tmp_path / "session.jsonl"
        write_jsonl(path, [
            make_user_entry("Hello"),
            make_assistant_entry(text="Hi there"),
        ])
        session = parse_session(path)
        turn = session.turns[1]
        assert turn.input_tokens == 100
        assert turn.output_tokens == 50
        assert turn.cache_creation_tokens == 10
        assert turn.cache_read_tokens == 20

    def test_session_metadata(self, tmp_path):
        project_dir = tmp_path / "-Users-test-project"
        project_dir.mkdir()
        path = project_dir / "abc-123.jsonl"
        write_jsonl(path, [
            make_user_entry("Hello", timestamp="2026-01-15T10:00:00Z"),
            make_assistant_entry(text="Hi", model="claude-opus-4-6"),
        ])
        session = parse_session(path)
        # session_id comes from the JSONL entry's sessionId field
        assert session.session_id == "test-session-id"
        assert session.model == "claude-opus-4-6"
        assert session.project_path == "-Users-test-project"
        assert session.project_dir == "/Users/test/project"
        assert session.cwd == "/Users/test/project"
        assert session.git_branch == "main"

    def test_session_id_from_filename_when_no_entry(self, tmp_path):
        """When entries lack sessionId, fall back to file stem."""
        path = tmp_path / "my-session.jsonl"
        write_jsonl(path, [
            {"type": "user", "message": {"role": "user", "content": "Hi"}, "timestamp": "2026-01-15T10:00:00Z"},
        ])
        session = parse_session(path)
        assert session.session_id == "my-session"

    def test_session_goal(self, tmp_path):
        path = tmp_path / "session.jsonl"
        write_jsonl(path, [
            make_user_entry("Fix the authentication bug"),
            make_assistant_entry(text="I'll look into it."),
        ])
        session = parse_session(path)
        assert session.goal == "Fix the authentication bug"

    def test_all_tool_calls(self, tmp_path):
        path = tmp_path / "session.jsonl"
        write_jsonl(path, [
            make_user_entry("Do stuff"),
            make_assistant_entry(tool_calls=[
                {"id": "t1", "name": "Read", "input": {"file_path": "a.py"}},
                {"id": "t2", "name": "Grep", "input": {"pattern": "foo"}},
            ]),
            make_assistant_entry(tool_calls=[
                {"id": "t3", "name": "Edit", "input": {"file_path": "a.py"}},
            ]),
        ])
        session = parse_session(path)
        assert len(session.all_tool_calls) == 3
        names = [tc.tool_name for tc in session.all_tool_calls]
        assert names == ["Read", "Grep", "Edit"]

    def test_turn_numbering(self, tmp_path):
        path = tmp_path / "session.jsonl"
        write_jsonl(path, [
            make_user_entry("First"),
            make_assistant_entry(text="Response 1"),
            make_user_entry("Second"),
            make_assistant_entry(text="Response 2"),
        ])
        session = parse_session(path)
        assert [t.turn_number for t in session.turns] == [1, 2, 3, 4]

    def test_parse_list_content_user(self, tmp_path):
        """User messages can have list content (slash command expansions)."""
        path = tmp_path / "session.jsonl"
        entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            },
            "timestamp": "2026-01-15T10:00:00Z",
        }
        write_jsonl(path, [entry])
        session = parse_session(path)
        assert len(session.turns) == 1
        assert session.turns[0].content == "Part 1\nPart 2"

    def test_skips_malformed_json(self, tmp_path):
        path = tmp_path / "session.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(make_user_entry("Hello")) + "\n")
            f.write("this is not valid json\n")
            f.write(json.dumps(make_assistant_entry(text="Hi")) + "\n")
        session = parse_session(path)
        assert len(session.turns) == 2

    def test_skips_non_conversation_entries(self, tmp_path):
        """Progress and file-history entries should be ignored."""
        path = tmp_path / "session.jsonl"
        entries = [
            {"type": "progress", "data": {"type": "hook_progress"}},
            {"type": "file-history-snapshot", "snapshot": {}},
            make_user_entry("Hello"),
            make_assistant_entry(text="Hi"),
        ]
        write_jsonl(path, entries)
        session = parse_session(path)
        assert len(session.turns) == 2

    def test_parse_thinking_blocks(self, tmp_path):
        path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [
                    {"type": "thinking", "thinking": "Let me think about this..."},
                    {"type": "text", "text": "Here's my answer."},
                ],
                "usage": {"input_tokens": 50, "output_tokens": 25},
            },
            "timestamp": "2026-01-15T10:01:00Z",
        }
        write_jsonl(path, [entry])
        session = parse_session(path)
        assert len(session.turns) == 1
        assert session.turns[0].thinking == "Let me think about this..."
        assert session.turns[0].content == "Here's my answer."


# --- Session discovery tests ---


class TestDiscoverSessions:
    def test_discover_for_project(self, tmp_path, monkeypatch):
        project_dir = tmp_path / "-Users-test-myproject"
        project_dir.mkdir()
        (project_dir / "session-1.jsonl").write_text('{"type":"user"}\n')
        (project_dir / "session-2.jsonl").write_text('{"type":"user"}\n')
        (project_dir / "empty.jsonl").write_text("")  # Should be skipped

        monkeypatch.setattr(
            "betty.session_reader.CLAUDE_PROJECTS_DIR", tmp_path
        )

        results = discover_sessions(project_dir="/Users/test/myproject")
        session_ids = [sid for sid, _ in results]
        assert "session-1" in session_ids
        assert "session-2" in session_ids
        assert "empty" not in session_ids

    def test_discover_global(self, tmp_path, monkeypatch):
        proj1 = tmp_path / "-Users-test-proj1"
        proj1.mkdir()
        (proj1 / "s1.jsonl").write_text('{"type":"user"}\n')

        proj2 = tmp_path / "-Users-test-proj2"
        proj2.mkdir()
        (proj2 / "s2.jsonl").write_text('{"type":"user"}\n')

        monkeypatch.setattr(
            "betty.session_reader.CLAUDE_PROJECTS_DIR", tmp_path
        )

        results = discover_sessions()
        session_ids = [sid for sid, _ in results]
        assert "s1" in session_ids
        assert "s2" in session_ids

    def test_discover_with_limit(self, tmp_path, monkeypatch):
        project_dir = tmp_path / "-Users-test-project"
        project_dir.mkdir()
        for i in range(5):
            (project_dir / f"s{i}.jsonl").write_text('{"type":"user"}\n')

        monkeypatch.setattr(
            "betty.session_reader.CLAUDE_PROJECTS_DIR", tmp_path
        )

        results = discover_sessions(project_dir="/Users/test/project", limit=3)
        assert len(results) == 3


# --- Session watcher tests ---


class TestSessionWatcher:
    @pytest.mark.asyncio
    async def test_watch_new_turns(self, tmp_path):
        path = tmp_path / "active.jsonl"
        # Start with empty file
        path.write_text("")

        watcher = SessionWatcher(path, poll_interval=0.1, start_position=0)
        turns_received: list[Turn] = []

        async def collect_turns():
            async for turn in watcher.watch():
                turns_received.append(turn)
                if len(turns_received) >= 2:
                    watcher.stop()
                    return

        async def add_entries():
            await asyncio.sleep(0.15)
            with open(path, "a") as f:
                f.write(json.dumps(make_user_entry("Hello")) + "\n")
            await asyncio.sleep(0.15)
            with open(path, "a") as f:
                f.write(json.dumps(make_assistant_entry(text="Hi there")) + "\n")

        # Use create_task instead of gather+wait_for for cleaner cleanup
        writer = asyncio.create_task(add_entries())
        try:
            await asyncio.wait_for(collect_turns(), timeout=3.0)
        finally:
            watcher.stop()
            await writer

        assert len(turns_received) == 2
        assert turns_received[0].role == "user"
        assert turns_received[1].role == "assistant"
        assert turns_received[1].content == "Hi there"

    @pytest.mark.asyncio
    async def test_watch_links_tool_results(self, tmp_path):
        path = tmp_path / "active.jsonl"
        path.write_text("")

        watcher = SessionWatcher(path, poll_interval=0.1, start_position=0)
        turns_received: list[Turn] = []

        async def collect_turns():
            async for turn in watcher.watch():
                turns_received.append(turn)
                if len(turns_received) >= 1:
                    await asyncio.sleep(0.3)
                    watcher.stop()

        async def write_entries():
            await asyncio.sleep(0.1)
            with open(path, "a") as f:
                f.write(json.dumps(make_assistant_entry(
                    text="Reading file.",
                    tool_calls=[{"id": "tc-99", "name": "Read", "input": {"file_path": "test.py"}}],
                )) + "\n")
            await asyncio.sleep(0.1)
            with open(path, "a") as f:
                f.write(json.dumps(make_tool_result_entry("tc-99", "file contents")) + "\n")

        await asyncio.gather(
            asyncio.wait_for(collect_turns(), timeout=3.0),
            write_entries(),
        )

        assert len(turns_received) >= 1
        tc = turns_received[0].tool_calls[0]
        assert tc.tool_name == "Read"
        assert tc.tool_id == "tc-99"


# --- Model property tests ---


class TestModels:
    def test_tool_call_file_path(self):
        tc = ToolCall(tool_id="1", tool_name="Read", input={"file_path": "/a/b.py"})
        assert tc.file_path == "/a/b.py"

    def test_tool_call_command(self):
        tc = ToolCall(tool_id="1", tool_name="Bash", input={"command": "ls -la"})
        assert tc.command == "ls -la"

    def test_tool_call_no_command_for_non_bash(self):
        tc = ToolCall(tool_id="1", tool_name="Read", input={"file_path": "/a.py"})
        assert tc.command is None

    def test_turn_word_count(self):
        turn = Turn(role="user", content="hello world foo bar")
        assert turn.word_count == 4

    def test_turn_tool_names(self):
        turn = Turn(role="assistant", content="", tool_calls=[
            ToolCall(tool_id="1", tool_name="Read", input={}),
            ToolCall(tool_id="2", tool_name="Edit", input={}),
        ])
        assert turn.tool_names == ["Read", "Edit"]

    def test_session_duration(self):
        s = Session(session_id="test")
        s.started_at = datetime(2026, 1, 15, 10, 0, 0)
        s.turns = [
            Turn(role="user", content="hello", timestamp=datetime(2026, 1, 15, 10, 0, 0)),
            Turn(role="assistant", content="hi", timestamp=datetime(2026, 1, 15, 10, 5, 0)),
        ]
        assert s.duration_seconds == 300.0

    def test_session_tool_approval_stats(self):
        s = Session(session_id="test")
        s.turns = [Turn(role="assistant", content="", tool_calls=[
            ToolCall(tool_id="1", tool_name="Read", input={}, approval=ToolApproval.ACCEPTED),
            ToolCall(tool_id="2", tool_name="Write", input={}, approval=ToolApproval.ACCEPTED),
            ToolCall(tool_id="3", tool_name="Bash", input={}, approval=ToolApproval.REJECTED),
        ])]
        stats = s.tool_approval_stats()
        assert stats["accepted"] == 2
        assert stats["rejected"] == 1
