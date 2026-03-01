"""Tests for betty.hooks — install, uninstall, and status of Claude Code hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from betty.hooks import (
    BETTY_HOOK_MARKER,
    HOOK_TYPES,
    _has_betty_hook,
    _remove_betty_entries,
    hooks_status,
    install_hooks,
    uninstall_hooks,
)


@pytest.fixture()
def settings_file(tmp_path: Path) -> Path:
    """Return the path to a temporary settings.json file."""
    return tmp_path / "settings.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


# ── install_hooks ──────────────────────────────────────────────────────────


class TestInstallHooks:
    def test_creates_settings_file_when_absent(self, settings_file: Path):
        installed = install_hooks(settings_path=settings_file)
        assert set(installed) == set(HOOK_TYPES)
        assert settings_file.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "settings.json"
        install_hooks(settings_path=deep)
        assert deep.exists()

    def test_all_hook_types_installed(self, settings_file: Path):
        install_hooks(settings_path=settings_file)
        data = _read(settings_file)
        for ht in HOOK_TYPES:
            assert ht in data["hooks"]
            entries = data["hooks"][ht]
            assert len(entries) == 1
            cmd = entries[0]["hooks"][0]["command"]
            assert BETTY_HOOK_MARKER in cmd
            assert ht in cmd

    def test_idempotent(self, settings_file: Path):
        install_hooks(settings_path=settings_file)
        installed2 = install_hooks(settings_path=settings_file)
        assert installed2 == []
        data = _read(settings_file)
        for ht in HOOK_TYPES:
            assert len(data["hooks"][ht]) == 1

    def test_preserves_existing_hooks(self, settings_file: Path):
        existing = {
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            {"type": "command", "command": "echo existing"}
                        ]
                    }
                ]
            },
            "env": {"SOME_VAR": "1"},
        }
        settings_file.write_text(json.dumps(existing))

        install_hooks(settings_path=settings_file)
        data = _read(settings_file)

        assert data["env"]["SOME_VAR"] == "1"
        user_entries = data["hooks"]["UserPromptSubmit"]
        assert len(user_entries) == 2
        assert user_entries[0]["hooks"][0]["command"] == "echo existing"

    def test_preserves_existing_non_hook_keys(self, settings_file: Path):
        existing = {"model": "claude-opus-4-6", "skipDangerousModePermissionPrompt": True}
        settings_file.write_text(json.dumps(existing))

        install_hooks(settings_path=settings_file)
        data = _read(settings_file)
        assert data["model"] == "claude-opus-4-6"
        assert data["skipDangerousModePermissionPrompt"] is True


# ── uninstall_hooks ────────────────────────────────────────────────────────


class TestUninstallHooks:
    def test_removes_betty_hooks(self, settings_file: Path):
        install_hooks(settings_path=settings_file)
        removed = uninstall_hooks(settings_path=settings_file)
        assert set(removed) == set(HOOK_TYPES)

        data = _read(settings_file)
        assert "hooks" not in data or data["hooks"] == {}

    def test_noop_when_not_installed(self, settings_file: Path):
        settings_file.write_text(json.dumps({}))
        removed = uninstall_hooks(settings_path=settings_file)
        assert removed == []

    def test_preserves_other_hooks(self, settings_file: Path):
        existing = {
            "hooks": {
                "UserPromptSubmit": [
                    {"hooks": [{"type": "command", "command": "echo other"}]},
                ]
            }
        }
        settings_file.write_text(json.dumps(existing))
        install_hooks(settings_path=settings_file)

        uninstall_hooks(settings_path=settings_file)
        data = _read(settings_file)

        assert len(data["hooks"]["UserPromptSubmit"]) == 1
        assert data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] == "echo other"

    def test_file_not_present(self, settings_file: Path):
        removed = uninstall_hooks(settings_path=settings_file)
        assert removed == []


# ── hooks_status ───────────────────────────────────────────────────────────


class TestHooksStatus:
    def test_all_false_when_not_installed(self, settings_file: Path):
        settings_file.write_text(json.dumps({}))
        status = hooks_status(settings_path=settings_file)
        assert all(v is False for v in status.values())
        assert set(status.keys()) == set(HOOK_TYPES)

    def test_all_true_after_install(self, settings_file: Path):
        install_hooks(settings_path=settings_file)
        status = hooks_status(settings_path=settings_file)
        assert all(v is True for v in status.values())

    def test_file_missing(self, settings_file: Path):
        status = hooks_status(settings_path=settings_file)
        assert all(v is False for v in status.values())


# ── helpers ────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_has_betty_hook_positive(self):
        entries = [
            {"hooks": [{"type": "command", "command": f"betty hook-handler X {BETTY_HOOK_MARKER}"}]}
        ]
        assert _has_betty_hook(entries) is True

    def test_has_betty_hook_negative(self):
        entries = [
            {"hooks": [{"type": "command", "command": "echo hello"}]}
        ]
        assert _has_betty_hook(entries) is False

    def test_remove_betty_entries(self):
        entries = [
            {"hooks": [{"type": "command", "command": "echo keep"}]},
            {"hooks": [{"type": "command", "command": f"betty hook-handler X {BETTY_HOOK_MARKER}"}]},
        ]
        cleaned = _remove_betty_entries(entries)
        assert len(cleaned) == 1
        assert cleaned[0]["hooks"][0]["command"] == "echo keep"
