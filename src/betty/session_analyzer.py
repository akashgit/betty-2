"""Session analyzer: extract patterns from completed sessions.

Analyzes Claude Code session transcripts to extract:
- Decision patterns (tool approvals, interventions, edits)
- Coding style signals (testing, naming, error handling, libraries)
- Workflow patterns (plan-first vs dive-in, session length preferences)
- Project conventions (build/test commands, directory structure)

Supports both heuristic extraction (no LLM needed) and LLM-augmented
analysis for nuanced pattern detection. Persists results to SQLite
with confidence scores that increase across sessions.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .models import Session, ToolCall, Turn

logger = logging.getLogger(__name__)

# --- Analysis Result Models ---


@dataclass
class Pattern:
    """A single observed pattern with confidence."""

    category: str  # e.g. "coding_style", "workflow", "tool_preference"
    key: str  # e.g. "test_framework", "naming_convention"
    value: str  # e.g. "pytest", "snake_case"
    confidence: float = 0.1  # starts low, grows with evidence
    evidence_count: int = 1
    evidence: list[str] = field(default_factory=list)  # brief supporting evidence


@dataclass
class SessionAnalysis:
    """Structured analysis of a single session."""

    session_id: str
    project_dir: str = ""

    # Extracted patterns by category
    patterns: list[Pattern] = field(default_factory=list)

    # Session-level summary
    goal: str | None = None
    outcome: str | None = None
    tools_used: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    decisions_made: list[dict[str, Any]] = field(default_factory=list)

    # Stats
    turn_count: int = 0
    tool_call_count: int = 0
    duration_seconds: float = 0.0

    def patterns_by_category(self, category: str) -> list[Pattern]:
        return [p for p in self.patterns if p.category == category]


# --- Heuristic Extractors ---

# Common test frameworks and their indicators
_TEST_FRAMEWORKS = {
    "pytest": ["pytest", "conftest", "@pytest.fixture", "pytest.mark"],
    "unittest": ["unittest", "TestCase", "self.assert"],
    "jest": ["jest", "describe(", "it(", "expect("],
    "mocha": ["mocha", "describe(", "it(", "chai"],
    "vitest": ["vitest", "describe(", "it(", "expect("],
}

# Build/test command patterns
_BUILD_COMMANDS = re.compile(
    r"(?:npm|yarn|pnpm|bun)\s+(?:run\s+)?(?:build|test|lint|dev|start)"
    r"|(?:make|cargo|go)\s+\w+"
    r"|pytest\b"
    r"|python\s+-m\s+\w+"
    r"|pip\s+install"
    r"|uv\s+(?:run|pip|sync)"
)


def _extract_tool_preferences(session: Session) -> list[Pattern]:
    """Extract patterns about which tools the user prefers."""
    patterns: list[Pattern] = []
    tool_counts: Counter[str] = Counter()

    for tc in session.all_tool_calls:
        tool_counts[tc.tool_name] += 1

    if not tool_counts:
        return patterns

    total = sum(tool_counts.values())

    # Dominant tools (>20% of total usage)
    for tool, count in tool_counts.most_common():
        ratio = count / total
        if ratio >= 0.2:
            patterns.append(Pattern(
                category="tool_preference",
                key=f"frequent_tool",
                value=tool,
                confidence=min(0.3, ratio),
                evidence=[f"Used {count}/{total} times ({ratio:.0%})"],
            ))

    return patterns


def _extract_file_patterns(session: Session) -> list[Pattern]:
    """Extract patterns about file organization and types worked on."""
    patterns: list[Pattern] = []
    file_paths: list[str] = []

    for tc in session.all_tool_calls:
        path = tc.file_path
        if path:
            file_paths.append(path)

    if not file_paths:
        return patterns

    # Detect language from file extensions
    ext_counts: Counter[str] = Counter()
    for fp in file_paths:
        if "." in fp:
            ext = fp.rsplit(".", 1)[-1].lower()
            ext_counts[ext] += 1

    ext_to_lang = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "tsx": "typescript-react", "jsx": "javascript-react",
        "rs": "rust", "go": "go", "rb": "ruby", "java": "java",
        "kt": "kotlin", "swift": "swift", "c": "c", "cpp": "cpp",
        "cs": "csharp",
    }

    for ext, count in ext_counts.most_common(3):
        if ext in ext_to_lang:
            patterns.append(Pattern(
                category="coding_style",
                key="primary_language",
                value=ext_to_lang[ext],
                confidence=0.2,
                evidence=[f"{count} files with .{ext} extension"],
            ))

    # Detect test file patterns
    test_files = [fp for fp in file_paths if "test" in fp.lower() or "spec" in fp.lower()]
    if test_files:
        # Detect test location pattern
        if any("/tests/" in fp for fp in test_files):
            patterns.append(Pattern(
                category="coding_style",
                key="test_location",
                value="separate_tests_dir",
                confidence=0.2,
                evidence=[f"{len(test_files)} test files in /tests/"],
            ))
        elif any("__tests__" in fp for fp in test_files):
            patterns.append(Pattern(
                category="coding_style",
                key="test_location",
                value="__tests___dir",
                confidence=0.2,
                evidence=[f"{len(test_files)} test files in __tests__/"],
            ))

    # Detect src layout
    src_files = [fp for fp in file_paths if "/src/" in fp]
    if src_files:
        patterns.append(Pattern(
            category="project_convention",
            key="source_layout",
            value="src_layout",
            confidence=0.15,
            evidence=[f"{len(src_files)} files under /src/"],
        ))

    return patterns


def _extract_bash_patterns(session: Session) -> list[Pattern]:
    """Extract patterns from Bash commands (build, test, toolchain)."""
    patterns: list[Pattern] = []
    commands: list[str] = []

    for tc in session.all_tool_calls:
        cmd = tc.command
        if cmd:
            commands.append(cmd)

    if not commands:
        return patterns

    # Detect test framework from commands
    for framework, indicators in _TEST_FRAMEWORKS.items():
        for cmd in commands:
            if any(ind in cmd for ind in indicators):
                patterns.append(Pattern(
                    category="coding_style",
                    key="test_framework",
                    value=framework,
                    confidence=0.3,
                    evidence=[f"Command: {cmd[:80]}"],
                ))
                break

    # Detect package manager
    pkg_managers = {
        "npm": 0, "yarn": 0, "pnpm": 0, "bun": 0,
        "pip": 0, "uv": 0, "poetry": 0,
        "cargo": 0, "go": 0, "make": 0,
    }
    for cmd in commands:
        first_word = cmd.strip().split()[0] if cmd.strip() else ""
        if first_word in pkg_managers:
            pkg_managers[first_word] += 1

    for mgr, count in sorted(pkg_managers.items(), key=lambda x: -x[1]):
        if count > 0:
            patterns.append(Pattern(
                category="project_convention",
                key="package_manager",
                value=mgr,
                confidence=min(0.3, 0.1 * count),
                evidence=[f"Used {count} times"],
            ))
            break  # only record the most-used

    # Detect build/test commands
    build_cmds: list[str] = []
    for cmd in commands:
        matches = _BUILD_COMMANDS.findall(cmd)
        build_cmds.extend(matches)

    for cmd in set(build_cmds):
        patterns.append(Pattern(
            category="project_convention",
            key="build_command",
            value=cmd,
            confidence=0.2,
            evidence=[f"Used in session"],
        ))

    return patterns


def _extract_workflow_patterns(session: Session) -> list[Pattern]:
    """Extract workflow patterns: plan-first vs dive-in, session style."""
    patterns: list[Pattern] = []

    if not session.turns:
        return patterns

    # Check if user starts with a planning message (long first prompt)
    first_user = None
    for turn in session.turns:
        if turn.role == "user":
            first_user = turn
            break

    if first_user:
        word_count = first_user.word_count
        if word_count > 100:
            patterns.append(Pattern(
                category="workflow",
                key="planning_style",
                value="detailed_upfront",
                confidence=0.15,
                evidence=[f"First prompt: {word_count} words"],
            ))
        elif word_count < 20:
            patterns.append(Pattern(
                category="workflow",
                key="planning_style",
                value="terse_incremental",
                confidence=0.15,
                evidence=[f"First prompt: {word_count} words"],
            ))

    # Session length preference
    user_turn_count = len(session.user_turns)
    if user_turn_count <= 3:
        patterns.append(Pattern(
            category="workflow",
            key="session_style",
            value="short_focused",
            confidence=0.1,
            evidence=[f"{user_turn_count} user turns"],
        ))
    elif user_turn_count > 10:
        patterns.append(Pattern(
            category="workflow",
            key="session_style",
            value="long_iterative",
            confidence=0.1,
            evidence=[f"{user_turn_count} user turns"],
        ))

    # Check if user provides corrections (suggests hands-on style)
    correction_keywords = ["no", "wrong", "actually", "instead", "not what", "try again"]
    correction_count = 0
    for turn in session.user_turns:
        content_lower = turn.content.lower()
        if any(kw in content_lower for kw in correction_keywords):
            correction_count += 1

    if correction_count >= 2:
        patterns.append(Pattern(
            category="workflow",
            key="feedback_style",
            value="frequent_corrections",
            confidence=0.15,
            evidence=[f"{correction_count} correction-like messages"],
        ))

    return patterns


def _extract_decision_patterns(session: Session) -> list[dict[str, Any]]:
    """Extract key decisions made during the session.

    Returns a list of decision dicts suitable for session_summaries storage.
    """
    decisions: list[dict[str, Any]] = []

    for turn in session.assistant_turns:
        for tc in turn.tool_calls:
            # File writes are decisions
            if tc.tool_name in ("Write", "Edit") and tc.file_path:
                decisions.append({
                    "type": "file_modification",
                    "tool": tc.tool_name,
                    "file": tc.file_path,
                })
            # Bash commands that change state
            elif tc.tool_name == "Bash" and tc.command:
                cmd = tc.command.strip()
                if any(cmd.startswith(prefix) for prefix in [
                    "git commit", "git push", "npm install", "pip install",
                    "rm ", "mkdir", "mv ", "cp ",
                ]):
                    decisions.append({
                        "type": "system_command",
                        "command": cmd[:100],
                    })

    return decisions


# --- LLM-Augmented Analysis ---

_LLM_ANALYSIS_PROMPT = """Analyze this Claude Code session transcript and extract user patterns.

Session goal: {goal}
Project directory: {project_dir}
Turn count: {turn_count}
Tools used: {tools_summary}

Here are the key turns (user messages and assistant actions):

{turns_text}

Extract the following as JSON:

{{
  "coding_style": {{
    "naming_convention": "snake_case | camelCase | PascalCase | null",
    "error_handling": "try_except | result_type | assertions | null",
    "documentation_style": "docstrings | comments | minimal | null",
    "type_annotations": "strict | optional | none | null"
  }},
  "workflow": {{
    "approach": "plan_first | dive_in | iterative",
    "testing": "tdd | test_after | minimal_testing | no_testing",
    "commit_style": "frequent_small | large_batched | null"
  }},
  "preferences": [
    {{"key": "preference_name", "value": "preference_value", "reason": "brief evidence"}}
  ]
}}

Only include fields where you have clear evidence. Set uncertain fields to null."""


async def _llm_analyze(
    session: Session,
    llm_complete_json: Any,
) -> list[Pattern]:
    """Use LLM to extract nuanced patterns from a session."""
    patterns: list[Pattern] = []

    # Build a compact summary of the session turns
    turns_text_parts: list[str] = []
    for turn in session.turns[:30]:  # Cap at 30 turns to fit context
        if turn.role == "user":
            content = turn.content[:200]
            turns_text_parts.append(f"USER: {content}")
        elif turn.role == "assistant":
            content = turn.content[:150] if turn.content else ""
            tools = ", ".join(turn.tool_names) if turn.tool_calls else "none"
            turns_text_parts.append(f"ASSISTANT: {content}\n  Tools: {tools}")

    turns_text = "\n".join(turns_text_parts)

    tool_counts: Counter[str] = Counter()
    for tc in session.all_tool_calls:
        tool_counts[tc.tool_name] += 1
    tools_summary = ", ".join(f"{name}({count})" for name, count in tool_counts.most_common(10))

    prompt = _LLM_ANALYSIS_PROMPT.format(
        goal=session.goal or "unknown",
        project_dir=session.project_dir,
        turn_count=len(session.turns),
        tools_summary=tools_summary or "none",
        turns_text=turns_text,
    )

    try:
        result = await llm_complete_json(
            prompt=prompt,
            system="You are a pattern analysis engine. Extract user preferences from coding sessions.",
            temperature=0.0,
        )
    except Exception as e:
        logger.warning("LLM analysis failed: %s", e)
        return patterns

    # Parse coding style
    coding_style = result.get("coding_style", {})
    for key, value in coding_style.items():
        if value and value != "null":
            patterns.append(Pattern(
                category="coding_style",
                key=key,
                value=str(value),
                confidence=0.2,
                evidence=["LLM-extracted from session"],
            ))

    # Parse workflow
    workflow = result.get("workflow", {})
    for key, value in workflow.items():
        if value and value != "null":
            patterns.append(Pattern(
                category="workflow",
                key=key,
                value=str(value),
                confidence=0.2,
                evidence=["LLM-extracted from session"],
            ))

    # Parse additional preferences
    preferences = result.get("preferences", [])
    for pref in preferences:
        if isinstance(pref, dict) and pref.get("key") and pref.get("value"):
            patterns.append(Pattern(
                category="preference",
                key=str(pref["key"]),
                value=str(pref["value"]),
                confidence=0.15,
                evidence=[str(pref.get("reason", "LLM-extracted"))],
            ))

    return patterns


# --- Main Analyzer ---


class SessionAnalyzer:
    """Analyze sessions to extract user patterns and preferences.

    Works in two modes:
    1. Heuristic-only: fast, no LLM needed, extracts structural patterns
    2. LLM-augmented: uses LLM for nuanced coding style and preference detection

    Usage:
        analyzer = SessionAnalyzer()
        analysis = analyzer.analyze(session)

        # With LLM:
        analyzer = SessionAnalyzer(llm=llm_service)
        analysis = await analyzer.analyze_with_llm(session)

        # Persist to DB:
        await analyzer.persist(analysis, db)
    """

    def __init__(self, llm: Any | None = None):
        """Initialize analyzer.

        Args:
            llm: Optional LLMService instance. If provided, analyze_with_llm()
                uses it for deeper pattern extraction.
        """
        self._llm = llm

    def analyze(self, session: Session) -> SessionAnalysis:
        """Analyze a session using heuristics only.

        Fast, deterministic, no external calls. Good for batch processing.
        """
        analysis = SessionAnalysis(
            session_id=session.session_id,
            project_dir=session.project_dir,
            goal=session.goal,
            turn_count=len(session.turns),
            tool_call_count=len(session.all_tool_calls),
            duration_seconds=session.duration_seconds,
        )

        # Collect all file paths touched
        seen_files: set[str] = set()
        for tc in session.all_tool_calls:
            if tc.file_path:
                seen_files.add(tc.file_path)
        analysis.files_touched = sorted(seen_files)

        # Collect tool names
        tool_names: set[str] = set()
        for tc in session.all_tool_calls:
            tool_names.add(tc.tool_name)
        analysis.tools_used = sorted(tool_names)

        # Extract decisions
        analysis.decisions_made = _extract_decision_patterns(session)

        # Run heuristic extractors
        analysis.patterns.extend(_extract_tool_preferences(session))
        analysis.patterns.extend(_extract_file_patterns(session))
        analysis.patterns.extend(_extract_bash_patterns(session))
        analysis.patterns.extend(_extract_workflow_patterns(session))

        return analysis

    async def analyze_with_llm(self, session: Session) -> SessionAnalysis:
        """Analyze a session using both heuristics and LLM.

        Falls back to heuristic-only if LLM is unavailable.
        """
        # Start with heuristic analysis
        analysis = self.analyze(session)

        # Add LLM patterns if available
        if self._llm is not None:
            try:
                llm_patterns = await _llm_analyze(
                    session,
                    self._llm.complete_json,
                )
                # Merge LLM patterns, avoiding duplicates
                existing_keys = {(p.category, p.key, p.value) for p in analysis.patterns}
                for p in llm_patterns:
                    if (p.category, p.key, p.value) not in existing_keys:
                        analysis.patterns.append(p)
            except Exception as e:
                logger.warning("LLM analysis failed, using heuristics only: %s", e)

        return analysis

    async def persist(
        self,
        analysis: SessionAnalysis,
        db: Any,
        project_scope: str | None = None,
        started_at: str | None = None,
    ) -> None:
        """Persist analysis results to the database.

        Saves:
        - Session summary to session_summaries table
        - Patterns to user_preferences table with confidence scores

        Args:
            analysis: The analysis to persist.
            db: A UserModelDB instance.
            project_scope: Optional project scope for preferences.
                Defaults to analysis.project_dir.
            started_at: ISO timestamp for session start. Falls back to
                current time if not provided.
        """
        scope = project_scope or analysis.project_dir or None

        if not started_at:
            from datetime import datetime, timezone
            started_at = datetime.now(timezone.utc).isoformat()

        # Save session summary
        await db.save_session(
            session_id=analysis.session_id,
            project_dir=analysis.project_dir,
            started_at=started_at,
            goal=analysis.goal,
            tools_used=analysis.tools_used,
            files_touched=analysis.files_touched,
            decisions_made=analysis.decisions_made,
            patterns_observed=[
                f"{p.category}:{p.key}={p.value}" for p in analysis.patterns
            ],
        )

        # Persist each pattern as a user preference
        for pattern in analysis.patterns:
            await db.set_preference(
                category=pattern.category,
                key=pattern.key,
                value=pattern.value,
                confidence=pattern.confidence,
                project_scope=scope,
            )

    def analyze_batch(self, sessions: list[Session]) -> list[SessionAnalysis]:
        """Analyze multiple sessions. Returns list of analyses."""
        return [self.analyze(session) for session in sessions]
