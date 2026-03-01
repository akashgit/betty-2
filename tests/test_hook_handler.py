"""Tests for betty.hook_handler — bridge between Claude Code and Betty daemon."""

from __future__ import annotations

import json
from io import StringIO
from unittest import mock

import pytest

from betty.hook_handler import (
    _format_pre_tool_use,
    _format_prompt_submit,
    _post_to_daemon,
    _read_stdin,
    handle_hook,
)


# ── _read_stdin ────────────────────────────────────────────────────────────


class TestReadStdin:
    def test_valid_json(self):
        payload = {"type": "UserPromptSubmit", "prompt": "hello"}
        with mock.patch("sys.stdin", StringIO(json.dumps(payload))):
            result = _read_stdin()
        assert result == payload

    def test_empty_stdin(self):
        with mock.patch("sys.stdin", StringIO("")):
            result = _read_stdin()
        assert result == {}

    def test_invalid_json(self):
        with mock.patch("sys.stdin", StringIO("not json")):
            result = _read_stdin()
        assert result == {}


# ── _post_to_daemon ────────────────────────────────────────────────────────


class TestPostToDaemon:
    def test_success(self):
        response_body = json.dumps({"action": "allow"}).encode("utf-8")
        mock_response = mock.MagicMock()
        mock_response.read.return_value = response_body
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response) as m:
            result = _post_to_daemon("PreToolUse", {"tool": "Bash"})

        assert result == {"action": "allow"}
        call_args = m.call_args
        req = call_args[0][0]
        assert "/hooks/PreToolUse" in req.full_url

    def test_daemon_unreachable(self):
        import urllib.error

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = _post_to_daemon("PreToolUse", {"tool": "Bash"})

        assert result is None

    def test_empty_response(self):
        mock_response = mock.MagicMock()
        mock_response.read.return_value = b""
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            result = _post_to_daemon("PostToolUse", {"tool": "Write"})

        assert result is None


# ── _format_prompt_submit ──────────────────────────────────────────────────


class TestFormatPromptSubmit:
    def test_with_questions(self):
        response = {"questions": ["What framework?", "Include tests?"]}
        output = _format_prompt_submit(response)
        assert output is not None
        ctx = output["additionalContext"]
        assert "What framework?" in ctx
        assert "Include tests?" in ctx

    def test_with_plan(self):
        response = {"predicted_plan": "Implement JWT auth with middleware"}
        output = _format_prompt_submit(response)
        assert "Predicted intent" in output["additionalContext"]
        assert "JWT auth" in output["additionalContext"]

    def test_with_policies(self):
        response = {"applicable_policies": [{"type": "security", "rule": "No hardcoded secrets"}]}
        output = _format_prompt_submit(response)
        assert "No hardcoded secrets" in output["additionalContext"]

    def test_with_similar_sessions(self):
        response = {"similar_sessions": [
            {"session_id": "abc12345", "goal": "Add login page", "relevance": 0.5}
        ]}
        output = _format_prompt_submit(response)
        assert "Add login page" in output["additionalContext"]

    def test_empty_response(self):
        output = _format_prompt_submit({"proceed": True})
        assert output is None

    def test_low_relevance_sessions_filtered(self):
        response = {"similar_sessions": [
            {"session_id": "abc", "goal": "Something", "relevance": 0.05}
        ]}
        output = _format_prompt_submit(response)
        assert output is None


# ── _format_pre_tool_use ───────────────────────────────────────────────────


class TestFormatPreToolUse:
    def test_allow(self):
        output = _format_pre_tool_use({"decision": "allow", "reason": "read-only"})
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert output["hookSpecificOutput"]["permissionDecisionReason"] == "read-only"

    def test_block_maps_to_deny(self):
        output = _format_pre_tool_use({"decision": "block", "reason": "destructive"})
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_ask(self):
        output = _format_pre_tool_use({"decision": "ask", "reason": "unknown pattern"})
        assert output["hookSpecificOutput"]["permissionDecision"] == "ask"

    def test_default_allow(self):
        output = _format_pre_tool_use({})
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"


# ── handle_hook ────────────────────────────────────────────────────────────


class TestHandleHook:
    def test_pre_tool_use_writes_hook_specific_output(self):
        payload = {"tool": "Bash", "input": {"command": "ls"}}
        response = {"decision": "allow", "reason": "safe"}

        with (
            mock.patch("betty.hook_handler._read_stdin", return_value=payload),
            mock.patch("betty.hook_handler._post_to_daemon", return_value=response),
            mock.patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            handle_hook("PreToolUse")

        output = json.loads(mock_stdout.getvalue())
        assert "hookSpecificOutput" in output
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_prompt_submit_writes_additional_context(self):
        payload = {"prompt": "add auth"}
        response = {"questions": ["Include tests?"]}

        with (
            mock.patch("betty.hook_handler._read_stdin", return_value=payload),
            mock.patch("betty.hook_handler._post_to_daemon", return_value=response),
            mock.patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            handle_hook("UserPromptSubmit")

        output = json.loads(mock_stdout.getvalue())
        assert "additionalContext" in output
        assert "Include tests?" in output["additionalContext"]

    def test_post_tool_use_no_output(self):
        payload = {"tool": "Write"}
        response = {"acknowledged": True}

        with (
            mock.patch("betty.hook_handler._read_stdin", return_value=payload),
            mock.patch("betty.hook_handler._post_to_daemon", return_value=response),
            mock.patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            handle_hook("PostToolUse")

        assert mock_stdout.getvalue() == ""

    def test_no_output_when_daemon_returns_none(self):
        payload = {"tool": "Bash"}

        with (
            mock.patch("betty.hook_handler._read_stdin", return_value=payload),
            mock.patch("betty.hook_handler._post_to_daemon", return_value=None),
            mock.patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            handle_hook("PostToolUse")

        assert mock_stdout.getvalue() == ""

    def test_empty_payload_skips_daemon_call(self):
        with (
            mock.patch("betty.hook_handler._read_stdin", return_value={}),
            mock.patch("betty.hook_handler._post_to_daemon") as mock_post,
        ):
            handle_hook("UserPromptSubmit")

        mock_post.assert_not_called()
