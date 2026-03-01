"""Core data models for Betty 2.0."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ToolApproval(Enum):
    """Tool approval decision by the user."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AUTO = "auto"  # Auto-approved (e.g., always-allow mode)
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    """A single tool invocation within an assistant turn."""

    tool_id: str  # tool_use block ID
    tool_name: str  # Read, Write, Edit, Bash, Grep, Glob, etc.
    input: dict[str, Any]  # Full tool input dict
    output: str | None = None  # Tool result text (from tool_result entry)
    success: bool | None = None  # Whether the tool call succeeded
    approval: ToolApproval = ToolApproval.UNKNOWN

    @property
    def file_path(self) -> str | None:
        """Extract file path from tool input, if applicable."""
        return (
            self.input.get("file_path")
            or self.input.get("path")
            or None
        )

    @property
    def command(self) -> str | None:
        """Extract command from Bash tool input."""
        if self.tool_name == "Bash":
            return self.input.get("command")
        return None


@dataclass
class Turn:
    """A single turn in a conversation (user message or assistant response)."""

    role: str  # "user" | "assistant"
    content: str  # Text content of the turn
    timestamp: datetime = field(default_factory=datetime.now)
    turn_number: int = 0

    # Assistant-specific fields
    model: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None

    # Thinking content (extended thinking)
    thinking: str | None = None

    @property
    def word_count(self) -> int:
        """Word count of the text content."""
        return len(self.content.split()) if self.content else 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def tool_names(self) -> list[str]:
        """List of tool names used in this turn."""
        return [tc.tool_name for tc in self.tool_calls]


@dataclass
class Session:
    """A Claude Code session parsed from a JSONL transcript."""

    session_id: str
    project_dir: str = ""  # Decoded project directory (e.g., /Users/foo/bar)
    project_path: str = ""  # Encoded project path (e.g., -Users-foo-bar)
    model: str = "unknown"
    cwd: str = ""
    git_branch: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    turns: list[Turn] = field(default_factory=list)
    active: bool = False

    @property
    def ended_at(self) -> datetime:
        """Timestamp of the last turn, or started_at if no turns."""
        if self.turns:
            return self.turns[-1].timestamp
        return self.started_at

    @property
    def duration_seconds(self) -> float:
        """Session duration in seconds."""
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        """All tool calls across all turns."""
        calls = []
        for turn in self.turns:
            calls.extend(turn.tool_calls)
        return calls

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens or 0 for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens or 0 for t in self.turns)

    @property
    def goal(self) -> str | None:
        """First user message as a proxy for session goal."""
        for turn in self.turns:
            if turn.role == "user" and turn.content.strip():
                return turn.content.strip()
        return None

    def tool_approval_stats(self) -> dict[str, int]:
        """Count of tool approvals by decision type."""
        stats: dict[str, int] = {}
        for tc in self.all_tool_calls:
            key = tc.approval.value
            stats[key] = stats.get(key, 0) + 1
        return stats
