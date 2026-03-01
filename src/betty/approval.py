"""Approval model for Betty 2.0.

Learns and predicts tool approval decisions based on past behavior.
Used by the PreToolUse hook to auto-approve or flag tool calls.

Safety tiers:
- ALWAYS_SAFE: Read-only tools (Read, Grep, Glob, WebSearch, WebFetch)
- LEARNABLE: Tools that can be auto-approved in familiar contexts (Write, Edit, Bash non-destructive)
- ALWAYS_ASK: Destructive or high-risk operations (force push, rm -rf, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any


class SafetyTier(Enum):
    """Safety classification for tool calls."""

    ALWAYS_SAFE = "always_safe"
    LEARNABLE = "learnable"
    ALWAYS_ASK = "always_ask"


class ApprovalDecision(Enum):
    """Possible approval decisions."""

    APPROVE = "approve"
    REJECT = "reject"
    ASK = "ask"


# Tools that are always safe (read-only or non-destructive)
ALWAYS_SAFE_TOOLS = frozenset({
    "Read", "Grep", "Glob", "WebSearch", "WebFetch",
    "TaskList", "TaskGet", "TaskCreate", "TaskUpdate",
    "Agent", "AskUserQuestion", "EnterPlanMode", "ExitPlanMode",
    "Skill", "SendMessage", "TeamCreate", "TeamDelete",
    "EnterWorktree",
})

# Tools that can be learned
LEARNABLE_TOOLS = frozenset({
    "Write", "Edit", "Bash", "NotebookEdit",
})

# Destructive bash patterns that should always require approval
DESTRUCTIVE_BASH_PATTERNS = [
    re.compile(r"\brm\s+-rf\b"),
    re.compile(r"\bgit\s+push\s+.*--force\b"),
    re.compile(r"\bgit\s+push\s+-f\b"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\s+-f"),
    re.compile(r"\bgit\s+branch\s+-D\b"),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+database\b", re.IGNORECASE),
    re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\s+777\b"),
    re.compile(r"\bcurl\b.*\|\s*(?:bash|sh)\b"),
]


@dataclass
class ApprovalPrediction:
    """Result of an approval prediction."""

    decision: ApprovalDecision
    confidence: float  # 0.0 to 1.0
    reason: str
    safety_tier: SafetyTier
    pattern_count: int = 0  # How many times this pattern has been seen


@dataclass
class ApprovalRecord:
    """A recorded approval decision for pattern matching."""

    tool_name: str
    action_pattern: str  # Normalized pattern (e.g., "edit:src/**/*.py")
    decision: str  # "accepted" or "rejected"
    count: int = 1
    project_scope: str | None = None


def classify_safety_tier(tool_name: str, tool_input: dict[str, Any]) -> SafetyTier:
    """Classify a tool call into a safety tier.

    Args:
        tool_name: Name of the tool (Read, Write, Bash, etc.)
        tool_input: Tool input parameters

    Returns:
        SafetyTier classification
    """
    if tool_name in ALWAYS_SAFE_TOOLS:
        return SafetyTier.ALWAYS_SAFE

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if _is_destructive_command(command):
            return SafetyTier.ALWAYS_ASK
        return SafetyTier.LEARNABLE

    if tool_name in LEARNABLE_TOOLS:
        return SafetyTier.LEARNABLE

    # Unknown tools default to always ask
    return SafetyTier.ALWAYS_ASK


def _is_destructive_command(command: str) -> bool:
    """Check if a bash command matches known destructive patterns."""
    return any(pat.search(command) for pat in DESTRUCTIVE_BASH_PATTERNS)


def _normalize_path_pattern(file_path: str) -> str:
    """Normalize a file path into a pattern for matching.

    Examples:
        /Users/foo/proj/src/main.py -> src/main.py  (relative to common prefix)
        /Users/foo/proj/src/utils/helper.py -> src/**/*.py (glob pattern)

    For simplicity, we use the directory + extension as the pattern.
    """
    path = PurePosixPath(file_path)
    # Use parent dir name + extension as pattern
    parent = path.parent.name or "root"
    suffix = path.suffix or ".unknown"
    return f"{parent}/*{suffix}"


def make_action_pattern(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Create a normalized action pattern for a tool call.

    This is used as the key for looking up approval history.
    """
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        # Extract the base command (first word)
        parts = command.strip().split()
        if parts:
            base_cmd = parts[0]
            # For git, include subcommand
            if base_cmd == "git" and len(parts) > 1:
                return f"bash:git-{parts[1]}"
            return f"bash:{base_cmd}"
        return "bash:unknown"

    if tool_name in ("Read", "Write", "Edit"):
        file_path = tool_input.get("file_path", "")
        if file_path:
            return f"{tool_name.lower()}:{_normalize_path_pattern(file_path)}"
        return f"{tool_name.lower()}:unknown"

    if tool_name == "Grep":
        return "grep:search"

    if tool_name == "Glob":
        return "glob:search"

    if tool_name == "NotebookEdit":
        notebook = tool_input.get("notebook_path", "")
        if notebook:
            return f"notebook:{_normalize_path_pattern(notebook)}"
        return "notebook:unknown"

    return f"{tool_name.lower()}:unknown"


class ApprovalModel:
    """Learns and predicts tool approval decisions.

    Uses a combination of safety tiers, learned patterns, and delegation
    level to decide whether to auto-approve, ask, or reject tool calls.
    """

    def __init__(self, autonomy_level: int = 1, confidence_threshold: float = 0.8):
        """Initialize the approval model.

        Args:
            autonomy_level: 0=observe, 1=suggest, 2=semi-auto, 3=full-auto
            confidence_threshold: Minimum confidence for auto-approval
        """
        self.autonomy_level = autonomy_level
        self.confidence_threshold = confidence_threshold
        self._patterns: dict[str, ApprovalRecord] = {}

    def load_patterns(self, records: list[dict[str, Any]]) -> None:
        """Load approval patterns from database records.

        Args:
            records: List of dicts with tool_name, action_pattern, decision, count
        """
        for rec in records:
            key = self._pattern_key(
                rec.get("tool_name", ""),
                rec.get("action_pattern", ""),
                rec.get("project_scope"),
            )
            self._patterns[key] = ApprovalRecord(
                tool_name=rec.get("tool_name", ""),
                action_pattern=rec.get("action_pattern", ""),
                decision=rec.get("decision", "unknown"),
                count=rec.get("count", 1),
                project_scope=rec.get("project_scope"),
            )

    def record(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        decision: str,
        project_scope: str | None = None,
    ) -> ApprovalRecord:
        """Record an approval decision for future predictions.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            decision: "accepted" or "rejected"
            project_scope: Project directory for scoping

        Returns:
            The ApprovalRecord that was created/updated
        """
        action_pattern = make_action_pattern(tool_name, tool_input)
        key = self._pattern_key(tool_name, action_pattern, project_scope)

        if key in self._patterns:
            self._patterns[key].decision = decision
            self._patterns[key].count += 1
        else:
            self._patterns[key] = ApprovalRecord(
                tool_name=tool_name,
                action_pattern=action_pattern,
                decision=decision,
                count=1,
                project_scope=project_scope,
            )

        return self._patterns[key]

    def predict(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        project_scope: str | None = None,
    ) -> ApprovalPrediction:
        """Predict whether a tool call should be approved.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            project_scope: Project directory for scoping

        Returns:
            ApprovalPrediction with decision, confidence, and reason
        """
        tier = classify_safety_tier(tool_name, tool_input)

        # Level 0: observe only, never auto-approve
        if self.autonomy_level == 0:
            return ApprovalPrediction(
                decision=ApprovalDecision.ASK,
                confidence=1.0,
                reason="Autonomy level 0 (observer mode)",
                safety_tier=tier,
            )

        # Always-safe tools: auto-approve at level >= 1
        if tier == SafetyTier.ALWAYS_SAFE:
            return ApprovalPrediction(
                decision=ApprovalDecision.APPROVE,
                confidence=1.0,
                reason=f"{tool_name} is a read-only tool",
                safety_tier=tier,
            )

        # Always-ask tools: never auto-approve
        if tier == SafetyTier.ALWAYS_ASK:
            command = tool_input.get("command", "")
            return ApprovalPrediction(
                decision=ApprovalDecision.ASK,
                confidence=1.0,
                reason=f"Destructive operation detected: {command[:80]}",
                safety_tier=tier,
            )

        # Learnable tools: check history and autonomy level
        action_pattern = make_action_pattern(tool_name, tool_input)

        # Check project-scoped pattern first, then global
        record = self._get_pattern(tool_name, action_pattern, project_scope)
        if record is None and project_scope:
            record = self._get_pattern(tool_name, action_pattern, None)

        if record is not None:
            if record.decision == "accepted":
                confidence = min(1.0, 0.5 + record.count * 0.1)
                # Level 3 (full-auto): approve any previously accepted pattern.
                # Level 2 (semi-auto): approve only if confidence meets threshold.
                if self.autonomy_level >= 3 or (
                    self.autonomy_level >= 2 and confidence >= self.confidence_threshold
                ):
                    return ApprovalPrediction(
                        decision=ApprovalDecision.APPROVE,
                        confidence=confidence,
                        reason=f"Pattern '{action_pattern}' approved {record.count} time(s)",
                        safety_tier=tier,
                        pattern_count=record.count,
                    )
                return ApprovalPrediction(
                    decision=ApprovalDecision.ASK,
                    confidence=confidence,
                    reason=f"Pattern seen {record.count} time(s) but autonomy level requires confirmation",
                    safety_tier=tier,
                    pattern_count=record.count,
                )
            else:
                # Previously rejected
                return ApprovalPrediction(
                    decision=ApprovalDecision.ASK,
                    confidence=0.8,
                    reason=f"Pattern '{action_pattern}' was previously rejected",
                    safety_tier=tier,
                    pattern_count=record.count,
                )

        # No history: ask unless full-auto
        if self.autonomy_level >= 3:
            return ApprovalPrediction(
                decision=ApprovalDecision.APPROVE,
                confidence=0.5,
                reason="Full-auto mode, no prior history",
                safety_tier=tier,
            )

        return ApprovalPrediction(
            decision=ApprovalDecision.ASK,
            confidence=0.0,
            reason=f"No approval history for pattern '{action_pattern}'",
            safety_tier=tier,
        )

    def get_auto_approve_rules(self) -> list[dict[str, Any]]:
        """Get patterns that qualify for auto-approval.

        Returns patterns that have been approved enough times
        to exceed the confidence threshold.
        """
        rules = []
        for record in self._patterns.values():
            if record.decision != "accepted":
                continue
            confidence = min(1.0, 0.5 + record.count * 0.1)
            if confidence >= self.confidence_threshold:
                rules.append({
                    "tool_name": record.tool_name,
                    "action_pattern": record.action_pattern,
                    "confidence": confidence,
                    "count": record.count,
                    "project_scope": record.project_scope,
                })
        return rules

    def _pattern_key(
        self,
        tool_name: str,
        action_pattern: str,
        project_scope: str | None,
    ) -> str:
        """Create a lookup key for a pattern."""
        scope = project_scope or "__global__"
        return f"{tool_name}|{action_pattern}|{scope}"

    def _get_pattern(
        self,
        tool_name: str,
        action_pattern: str,
        project_scope: str | None,
    ) -> ApprovalRecord | None:
        """Look up a pattern record."""
        key = self._pattern_key(tool_name, action_pattern, project_scope)
        return self._patterns.get(key)
