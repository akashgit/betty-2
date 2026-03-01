"""Hook handler — bridge between Claude Code hooks and the Betty daemon.

When Claude Code fires a hook it runs::

    betty hook-handler <hook-type>

This module reads the hook payload from stdin, forwards it to the
Betty daemon via HTTP, and translates the response into the format
Claude Code expects.  If the daemon is unreachable the handler exits
silently so Claude Code is never blocked.

Claude Code hook response formats:
- UserPromptSubmit: {"additionalContext": "..."} or empty
- PreToolUse: {"hookSpecificOutput": {"permissionDecision": "allow|deny|ask", ...}}
- PostToolUse: empty (fire-and-forget)
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

# Default timeout for daemon communication.  Hooks are latency-
# sensitive (Claude Code blocks on them), so keep this short.
# UserPromptSubmit gets more time because the intent engine may call LLM.
DAEMON_TIMEOUT_SECS = 2

_HOOK_TIMEOUTS: dict[str, float] = {
    "UserPromptSubmit": 5,
    "PreToolUse": 2,
    "PostToolUse": 2,
}

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


def _format_prompt_submit(response: dict[str, Any]) -> dict[str, Any] | None:
    """Translate Betty's UserPromptSubmit response to Claude Code format.

    Claude Code expects:
    - {"additionalContext": "..."} to inject context into the conversation
    - {"decision": "block", "reason": "..."} to block the prompt
    - empty/no output to proceed normally
    """
    parts: list[str] = []

    # Clarifying questions.
    questions = response.get("questions", [])
    if questions:
        parts.append("Betty suggests asking these clarifying questions:")
        for q in questions:
            parts.append(f"  - {q}")

    # Predicted plan.
    plan = response.get("predicted_plan")
    if plan:
        parts.append(f"\nPredicted intent: {plan}")

    # Similar sessions.
    similar = response.get("similar_sessions", [])
    if similar:
        relevant = [s for s in similar if s.get("goal") and s.get("relevance", 0) > 0.15]
        if relevant:
            parts.append("\nRelevant past sessions:")
            for s in relevant[:3]:
                parts.append(f"  - {s['goal']} (session {s['session_id'][:8]})")

    # Applicable policies.
    policies = response.get("applicable_policies", [])
    if policies:
        parts.append("\nApplicable policies:")
        for p in policies:
            parts.append(f"  - [{p.get('type', 'policy')}] {p.get('rule', '')}")

    # Suggested context.
    ctx = response.get("suggested_context")
    if ctx:
        parts.append(f"\n{ctx}")

    if not parts:
        return None

    return {"additionalContext": "\n".join(parts)}


def _format_pre_tool_use(response: dict[str, Any]) -> dict[str, Any] | None:
    """Translate Betty's PreToolUse response to Claude Code format.

    Claude Code expects:
    {
        "hookSpecificOutput": {
            "permissionDecision": "allow" | "deny" | "ask",
            "permissionDecisionReason": "...",
        }
    }
    """
    decision = response.get("decision", "allow")
    reason = response.get("reason", "")

    # Map Betty's internal decision names to Claude Code's format.
    decision_map = {"allow": "allow", "block": "deny", "ask": "ask"}
    permission = decision_map.get(decision, "allow")

    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": permission,
            "permissionDecisionReason": reason,
        }
    }


def handle_hook(hook_type: str) -> None:
    """Main entry point: read stdin, call daemon, write response.

    This function is designed to be called from the CLI and always
    exits cleanly.  Any output it prints to stdout is interpreted by
    Claude Code as the hook's response.
    """
    payload = _read_stdin()
    if not payload:
        return

    timeout = _HOOK_TIMEOUTS.get(hook_type, DAEMON_TIMEOUT_SECS)
    response = _post_to_daemon(hook_type, payload, timeout=timeout)

    if not response:
        return

    # Translate to Claude Code's expected format.
    if hook_type == "UserPromptSubmit":
        output = _format_prompt_submit(response)
    elif hook_type == "PreToolUse":
        output = _format_pre_tool_use(response)
    else:
        # PostToolUse is fire-and-forget, no output needed.
        return

    if output:
        json.dump(output, sys.stdout)
        sys.stdout.flush()
