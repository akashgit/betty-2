"""Tests for betty.hook_handler — bridge between Claude Code and Betty daemon."""

from __future__ import annotations

import json
from io import StringIO
from unittest import mock

import pytest

from betty.hook_handler import _post_to_daemon, _read_stdin, handle_hook


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


# ── handle_hook ────────────────────────────────────────────────────────────


class TestHandleHook:
    def test_writes_response_to_stdout(self):
        payload = {"tool": "Bash", "input": {"command": "ls"}}
        response = {"decision": "allow"}

        with (
            mock.patch("betty.hook_handler._read_stdin", return_value=payload),
            mock.patch("betty.hook_handler._post_to_daemon", return_value=response),
            mock.patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            handle_hook("PreToolUse")

        output = mock_stdout.getvalue()
        assert json.loads(output) == response

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
