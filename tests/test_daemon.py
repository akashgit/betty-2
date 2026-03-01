"""Tests for betty.daemon — main process orchestrator."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from betty.daemon import (
    _read_pid,
    _remove_pid,
    _write_pid,
    is_running,
    stop_daemon,
)


@pytest.fixture(autouse=True)
def _use_tmp_pid(tmp_path: Path, monkeypatch):
    """Redirect PID file and log file to a temp directory."""
    monkeypatch.setattr("betty.daemon.PID_FILE", tmp_path / "betty.pid")
    monkeypatch.setattr("betty.daemon.LOG_FILE", tmp_path / "betty.log")
    monkeypatch.setattr("betty.daemon.BETTY_DIR", tmp_path)


# ── PID file management ──────────────────────────────────────────────────


class TestPidFile:
    def test_write_and_read(self):
        _write_pid()
        pid = _read_pid()
        assert pid == os.getpid()

    def test_read_no_file(self):
        assert _read_pid() is None

    def test_remove(self):
        _write_pid()
        _remove_pid()
        assert _read_pid() is None

    def test_remove_no_file(self):
        # Should not raise.
        _remove_pid()

    def test_stale_pid_cleaned(self, tmp_path: Path):
        # Write a PID that doesn't exist.
        pid_file = tmp_path / "betty.pid"
        pid_file.write_text("999999999")
        assert _read_pid() is None
        assert not pid_file.exists()


# ── is_running ───────────────────────────────────────────────────────────


class TestIsRunning:
    def test_not_running_by_default(self):
        assert not is_running()

    def test_running_after_write(self):
        _write_pid()
        assert is_running()

    def test_not_running_after_remove(self):
        _write_pid()
        _remove_pid()
        assert not is_running()


# ── stop_daemon ──────────────────────────────────────────────────────────


class TestStopDaemon:
    def test_stop_no_daemon(self):
        assert stop_daemon() is False

    def test_stop_sends_signal(self):
        _write_pid()
        with mock.patch("os.kill") as mock_kill:
            result = stop_daemon()
        assert result is True
        # os.kill called twice: once with signal 0 (alive check in _read_pid),
        # then with SIGTERM (the actual stop).
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(os.getpid(), 0)
        mock_kill.assert_any_call(os.getpid(), 15)

    def test_stop_stale_pid(self, tmp_path: Path):
        pid_file = tmp_path / "betty.pid"
        pid_file.write_text("999999999")
        assert stop_daemon() is False
