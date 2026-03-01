"""Hook handler — bridge between Claude Code hooks and the Betty daemon.

When Claude Code fires a hook it runs::

    betty hook-handler <hook-type>

This module reads the hook payload from stdin, forwards it to the
Betty daemon via HTTP, and prints any response JSON that Claude Code
should act on.  If the daemon is unreachable the handler exits
silently so Claude Code is never blocked.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

# Default timeout for daemon communication.  Hooks are latency-
# sensitive (Claude Code blocks on them), so keep this short.
DAEMON_TIMEOUT_SECS = 2

# Port where the Betty daemon listens.
DAEMON_PORT = 7832
DAEMON_BASE_URL = f"http://localhost:{DAEMON_PORT}"


def _read_stdin() -> dict[str, Any]:
    """Read and parse the JSON payload that Claude Code sends on stdin."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return {}


def _post_to_daemon(
    hook_type: str,
    payload: dict[str, Any],
    *,
    timeout: float = DAEMON_TIMEOUT_SECS,
) -> dict[str, Any] | None:
    """POST the hook payload to the daemon and return its response.

    Returns ``None`` when the daemon is not reachable or returns an
    error — the caller should treat this as "no action".
    """
    url = f"{DAEMON_BASE_URL}/hooks/{hook_type}"
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read().decode("utf-8")
            if resp_body.strip():
                return json.loads(resp_body)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        # Daemon not running or returned garbage — fail silently.
        pass

    return None


def handle_hook(hook_type: str) -> None:
    """Main entry point: read stdin, call daemon, write response.

    This function is designed to be called from the CLI and always
    exits cleanly.  Any output it prints to stdout is interpreted by
    Claude Code as the hook's response.
    """
    payload = _read_stdin()
    if not payload:
        return

    response = _post_to_daemon(hook_type, payload)

    if response:
        # Print the JSON response so Claude Code picks it up.
        json.dump(response, sys.stdout)
        sys.stdout.flush()
