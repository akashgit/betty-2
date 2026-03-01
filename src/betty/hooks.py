"""Claude Code hooks management.

Install, uninstall, and query Betty hook entries in Claude Code's
``~/.claude/settings.json``.  Each hook calls
``betty hook-handler <hook-type>`` which forwards the event to the
running Betty daemon via HTTP.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Where Claude Code stores its global settings.
CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# Marker used to identify Betty-managed hook entries.
BETTY_HOOK_MARKER = "# betty-hook"

# Hook types Betty installs.
HOOK_TYPES = ("UserPromptSubmit", "PreToolUse", "PostToolUse")


def _make_hook_command(hook_type: str) -> str:
    """Build the shell command for a single hook type.

    The command pipes stdin JSON to ``betty hook-handler`` and silently
    succeeds when the daemon is unreachable so Claude Code is never
    blocked.
    """
    return f"betty hook-handler {hook_type} {BETTY_HOOK_MARKER}"


def _load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load Claude Code settings, returning ``{}`` if absent or invalid."""
    path = path or CLAUDE_SETTINGS_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read Claude settings: %s", exc)
        return {}


def _save_settings(data: dict[str, Any], path: Path | None = None) -> None:
    """Write settings back, creating parent directories if needed."""
    path = path or CLAUDE_SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


def _has_betty_hook(hook_entries: list[dict[str, Any]]) -> bool:
    """Return True if any entry in *hook_entries* was installed by Betty."""
    for entry in hook_entries:
        for hook in entry.get("hooks", []):
            cmd = hook.get("command", "")
            if BETTY_HOOK_MARKER in cmd:
                return True
    return False


def _remove_betty_entries(hook_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return *hook_entries* without any Betty-managed entries."""
    cleaned: list[dict[str, Any]] = []
    for entry in hook_entries:
        hooks = entry.get("hooks", [])
        filtered = [
            h for h in hooks
            if BETTY_HOOK_MARKER not in h.get("command", "")
        ]
        if filtered:
            cleaned.append({**entry, "hooks": filtered})
    return cleaned


def install_hooks(settings_path: Path | None = None) -> list[str]:
    """Install Betty hooks into Claude Code settings.

    Non-destructive: existing hooks are preserved.

    Returns:
        List of hook types that were newly installed.
    """
    data = _load_settings(settings_path)
    hooks_section: dict[str, Any] = data.setdefault("hooks", {})

    installed: list[str] = []
    for hook_type in HOOK_TYPES:
        entries: list[dict[str, Any]] = hooks_section.setdefault(hook_type, [])

        if _has_betty_hook(entries):
            logger.debug("Hook %s already installed, skipping", hook_type)
            continue

        betty_entry: dict[str, Any] = {
            "hooks": [
                {
                    "type": "command",
                    "command": _make_hook_command(hook_type),
                }
            ]
        }
        entries.append(betty_entry)
        installed.append(hook_type)

    _save_settings(data, settings_path)
    return installed


def uninstall_hooks(settings_path: Path | None = None) -> list[str]:
    """Remove all Betty hooks from Claude Code settings.

    Returns:
        List of hook types that were removed.
    """
    data = _load_settings(settings_path)
    hooks_section: dict[str, Any] = data.get("hooks", {})

    removed: list[str] = []
    for hook_type in HOOK_TYPES:
        entries: list[dict[str, Any]] = hooks_section.get(hook_type, [])
        if not _has_betty_hook(entries):
            continue

        cleaned = _remove_betty_entries(entries)
        if cleaned:
            hooks_section[hook_type] = cleaned
        else:
            hooks_section.pop(hook_type, None)
        removed.append(hook_type)

    # Clean up empty hooks section.
    if not hooks_section:
        data.pop("hooks", None)

    _save_settings(data, settings_path)
    return removed


def hooks_status(settings_path: Path | None = None) -> dict[str, bool]:
    """Check which Betty hooks are currently installed.

    Returns:
        Mapping of hook type to installed boolean.
    """
    data = _load_settings(settings_path)
    hooks_section: dict[str, Any] = data.get("hooks", {})

    status: dict[str, bool] = {}
    for hook_type in HOOK_TYPES:
        entries = hooks_section.get(hook_type, [])
        status[hook_type] = _has_betty_hook(entries)
    return status
