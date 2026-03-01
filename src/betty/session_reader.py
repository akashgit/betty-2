"""Session reader: parse Claude Code JSONL transcripts.

Reads and parses Claude Code session transcript files (.jsonl) from
~/.claude/projects/<encoded-cwd>/. Supports batch reading of completed
sessions and async polling of active sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from .models import Session, ToolCall, Turn

logger = logging.getLogger(__name__)

# ~/.claude/projects/ is where Claude Code stores session transcripts
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


# --- Path encoding/decoding ---


def encode_project_path(path: str) -> str:
    """Encode a project directory path for Claude Code's storage format.

    /Users/foo/bar -> -Users-foo-bar
    """
    return "-" + path.lstrip("/").replace("/", "-")


def decode_project_path(encoded: str) -> str:
    """Decode a Claude Code encoded project path.

    -Users-foo-bar -> /Users/foo/bar
    """
    if encoded.startswith("-"):
        return "/" + encoded[1:].replace("-", "/")
    return encoded


# --- JSONL Parsing ---


def _parse_timestamp(ts: str | None) -> datetime:
    """Parse ISO 8601 timestamp to local datetime."""
    if not ts:
        return datetime.now()
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return datetime.now()


def _parse_user_entry(entry: dict) -> Turn | None:
    """Parse a user-type JSONL entry into a Turn."""
    message = entry.get("message", {})
    content = message.get("content", "")

    # Content can be a string or a list of blocks
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
        content = "\n".join(text_parts)

    if not isinstance(content, str) or not content.strip():
        return None

    return Turn(
        role="user",
        content=content,
        timestamp=_parse_timestamp(entry.get("timestamp")),
    )


def _parse_assistant_entry(entry: dict) -> Turn | None:
    """Parse an assistant-type JSONL entry into a Turn with tool calls."""
    message = entry.get("message", {})
    content_blocks = message.get("content", [])

    if not isinstance(content_blocks, list):
        return None

    # Extract usage and model info
    usage = message.get("usage", {})
    model = message.get("model")

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking: str | None = None

    for block in content_blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "")

        if block_type == "text":
            text = block.get("text", "")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)

        elif block_type == "tool_use":
            tool_calls.append(ToolCall(
                tool_id=block.get("id", ""),
                tool_name=block.get("name", ""),
                input=block.get("input", {}),
            ))

        elif block_type == "thinking":
            thinking_text = block.get("thinking", "")
            if isinstance(thinking_text, str) and thinking_text.strip():
                thinking = thinking_text

    content = "\n".join(text_parts)

    # Skip entries with no meaningful content
    if not content and not tool_calls and not thinking:
        return None

    return Turn(
        role="assistant",
        content=content,
        timestamp=_parse_timestamp(entry.get("timestamp")),
        model=model,
        tool_calls=tool_calls,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_creation_tokens=usage.get("cache_creation_input_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        thinking=thinking,
    )


def _parse_tool_result_entry(entry: dict) -> tuple[str, str, bool] | None:
    """Parse a tool_result entry. Returns (tool_use_id, output_text, is_error)."""
    message = entry.get("message", {})
    content = message.get("content", [])
    tool_use_id = None
    output_parts: list[str] = []
    is_error = message.get("is_error", False)

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    output_parts.append(result_content)
                elif isinstance(result_content, list):
                    for sub in result_content:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            output_parts.append(sub.get("text", ""))
                is_error = is_error or block.get("is_error", False)

    if tool_use_id is None:
        return None

    return tool_use_id, "\n".join(output_parts), is_error


def parse_session(transcript_path: Path) -> Session:
    """Parse a complete JSONL transcript file into a Session.

    Reads the entire file and returns a structured Session with all turns,
    tool calls, and metadata.
    """
    session = Session(session_id=transcript_path.stem)
    turns: list[Turn] = []

    # Map tool_use_id -> ToolCall for linking results
    pending_tool_calls: dict[str, ToolCall] = {}

    # Track metadata from first entry
    metadata_set = False
    turn_number = 0

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type", "")

                # Extract session metadata from first relevant entry
                if not metadata_set:
                    if entry.get("sessionId"):
                        session.session_id = entry["sessionId"]
                    if entry.get("cwd"):
                        session.cwd = entry["cwd"]
                    if entry.get("gitBranch"):
                        session.git_branch = entry["gitBranch"]
                    # Derive project info from the file path
                    project_encoded = transcript_path.parent.name
                    session.project_path = project_encoded
                    session.project_dir = decode_project_path(project_encoded)
                    metadata_set = True

                if entry_type == "user":
                    turn = _parse_user_entry(entry)
                    if turn:
                        turn_number += 1
                        turn.turn_number = turn_number
                        turns.append(turn)

                elif entry_type == "assistant":
                    turn = _parse_assistant_entry(entry)
                    if turn:
                        turn_number += 1
                        turn.turn_number = turn_number
                        turns.append(turn)

                        # Track model from first assistant turn
                        if session.model == "unknown" and turn.model:
                            session.model = turn.model

                        # Index tool calls for result matching
                        for tc in turn.tool_calls:
                            if tc.tool_id:
                                pending_tool_calls[tc.tool_id] = tc

                elif entry_type == "tool_result":
                    result = _parse_tool_result_entry(entry)
                    if result:
                        tool_use_id, output, is_error = result
                        tc = pending_tool_calls.get(tool_use_id)
                        if tc:
                            tc.output = output
                            tc.success = not is_error

    except (IOError, OSError) as e:
        logger.error(f"Failed to read transcript: {transcript_path}: {e}")

    session.turns = turns
    if turns:
        session.started_at = turns[0].timestamp

    return session


# --- Session Discovery ---


def discover_sessions(
    project_dir: str | None = None,
    limit: int | None = None,
) -> list[tuple[str, Path]]:
    """Discover session transcript files.

    Args:
        project_dir: If set, only discover sessions for this project directory.
            If None, discovers all sessions globally.
        limit: Maximum number of sessions to return (most recent first).

    Returns:
        List of (session_id, transcript_path) tuples, sorted by modification
        time (most recent first).
    """
    if not CLAUDE_PROJECTS_DIR.exists():
        return []

    sessions: list[tuple[str, Path, float]] = []

    if project_dir:
        # Single project
        encoded = encode_project_path(project_dir)
        project_path = CLAUDE_PROJECTS_DIR / encoded
        if project_path.exists():
            _collect_sessions(project_path, sessions)
    else:
        # Global: scan all project directories
        try:
            for subdir in CLAUDE_PROJECTS_DIR.iterdir():
                if subdir.is_dir() and subdir.name.startswith("-"):
                    _collect_sessions(subdir, sessions)
        except (PermissionError, OSError):
            pass

    # Sort by modification time (most recent first)
    sessions.sort(key=lambda x: x[2], reverse=True)

    if limit:
        sessions = sessions[:limit]

    return [(sid, path) for sid, path, _ in sessions]


def _collect_sessions(
    project_path: Path,
    results: list[tuple[str, Path, float]],
) -> None:
    """Collect session files from a project directory."""
    try:
        for f in project_path.iterdir():
            if f.suffix == ".jsonl" and f.is_file():
                try:
                    stat = f.stat()
                    if stat.st_size > 0:
                        results.append((f.stem, f, stat.st_mtime))
                except (PermissionError, OSError):
                    pass
    except (PermissionError, OSError):
        pass


def read_sessions(
    project_dir: str | None = None,
    limit: int | None = None,
) -> list[Session]:
    """Read and parse multiple sessions.

    Args:
        project_dir: If set, only read sessions for this project.
            If None, reads all sessions globally.
        limit: Maximum number of sessions to read.

    Returns:
        List of Session objects, sorted by start time (most recent first).
    """
    discovered = discover_sessions(project_dir=project_dir, limit=limit)
    sessions = []
    for _session_id, path in discovered:
        session = parse_session(path)
        sessions.append(session)
    return sessions


# --- Active Session Watcher (async) ---


class SessionWatcher:
    """Watch an active session transcript for new turns using asyncio.

    Polls the transcript file for new content and yields new turns
    as they appear.
    """

    def __init__(
        self,
        transcript_path: Path,
        poll_interval: float = 0.5,
        start_position: int | None = None,
    ):
        self._path = transcript_path
        self._poll_interval = poll_interval
        self._position = start_position or 0
        self._turn_number = 0
        self._pending_tool_calls: dict[str, ToolCall] = {}
        self._stop_event: asyncio.Event | None = None

    async def watch(self) -> AsyncIterator[Turn]:
        """Async generator that yields new turns as they appear.

        Usage:
            watcher = SessionWatcher(path)
            async for turn in watcher.watch():
                process(turn)
        """
        self._stop_event = asyncio.Event()

        # Wait for file to exist
        for _ in range(40):  # Up to 2 seconds
            if self._path.exists():
                break
            await asyncio.sleep(0.05)

        if not self._path.exists():
            logger.warning(f"Transcript file not found: {self._path}")
            return

        # If no start position given, start from end of file
        if self._position == 0:
            self._position = self._path.stat().st_size

        while not self._stop_event.is_set():
            async for turn in self._check_for_updates():
                yield turn
                if self._stop_event.is_set():
                    return
            # Wait for poll interval or stop signal
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval,
                )
                return  # Stop was requested
            except asyncio.TimeoutError:
                pass  # Normal poll interval elapsed

    def stop(self) -> None:
        """Stop the watcher."""
        if self._stop_event:
            self._stop_event.set()

    async def _check_for_updates(self) -> AsyncIterator[Turn]:
        """Check for new content and yield any new turns."""
        if not self._path.exists():
            return

        try:
            current_size = self._path.stat().st_size
        except OSError:
            return

        if current_size <= self._position:
            return

        last_good_position = self._position

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                f.seek(self._position)

                while True:
                    raw_line = f.readline()
                    if not raw_line:
                        break

                    has_newline = raw_line.endswith("\n")
                    line = raw_line.strip()

                    if not line:
                        last_good_position = f.tell()
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        if has_newline:
                            # Complete but malformed — skip
                            last_good_position = f.tell()
                        else:
                            # Incomplete line (still being written) — retry later
                            break
                        continue

                    entry_type = entry.get("type", "")
                    turn: Turn | None = None

                    if entry_type == "user":
                        turn = _parse_user_entry(entry)
                    elif entry_type == "assistant":
                        turn = _parse_assistant_entry(entry)
                        if turn:
                            for tc in turn.tool_calls:
                                if tc.tool_id:
                                    self._pending_tool_calls[tc.tool_id] = tc
                    elif entry_type == "tool_result":
                        result = _parse_tool_result_entry(entry)
                        if result:
                            tool_use_id, output, is_error = result
                            tc = self._pending_tool_calls.get(tool_use_id)
                            if tc:
                                tc.output = output
                                tc.success = not is_error

                    if turn:
                        self._turn_number += 1
                        turn.turn_number = self._turn_number
                        yield turn

                    last_good_position = f.tell()

        except (IOError, OSError) as e:
            logger.error(f"Error reading transcript: {e}")

        self._position = last_good_position
