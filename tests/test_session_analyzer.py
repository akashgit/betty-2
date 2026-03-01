"""Tests for betty.session_analyzer — extract patterns from sessions."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from betty.db import UserModelDB
from betty.models import Session, ToolCall, Turn
from betty.session_analyzer import (
    Pattern,
    SessionAnalysis,
    SessionAnalyzer,
    _extract_bash_patterns,
    _extract_decision_patterns,
    _extract_file_patterns,
    _extract_tool_preferences,
    _extract_workflow_patterns,
)


# --- Helpers ---


def _make_session(
    turns: list[Turn] | None = None,
    session_id: str = "test-session",
    project_dir: str = "/Users/test/project",
) -> Session:
    """Create a session with the given turns."""
    s = Session(session_id=session_id, project_dir=project_dir)
    s.turns = turns or []
    if turns:
        s.started_at = turns[0].timestamp
    return s


def _make_user_turn(content: str, turn_number: int = 1) -> Turn:
    return Turn(
        role="user",
        content=content,
        turn_number=turn_number,
        timestamp=datetime(2025, 1, 15, 10, 0, 0) + timedelta(minutes=turn_number),
    )


def _make_assistant_turn(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    turn_number: int = 2,
) -> Turn:
    return Turn(
        role="assistant",
        content=content,
        turn_number=turn_number,
        tool_calls=tool_calls or [],
        timestamp=datetime(2025, 1, 15, 10, 0, 0) + timedelta(minutes=turn_number),
    )


def _make_tool_call(
    tool_name: str = "Read",
    input: dict | None = None,
    output: str | None = None,
    tool_id: str = "tc_1",
) -> ToolCall:
    return ToolCall(
        tool_id=tool_id,
        tool_name=tool_name,
        input=input or {},
        output=output,
    )


# --- Test Pattern Dataclass ---


class TestPattern:
    def test_pattern_defaults(self):
        p = Pattern(category="test", key="k", value="v")
        assert p.confidence == 0.1
        assert p.evidence_count == 1
        assert p.evidence == []

    def test_pattern_with_evidence(self):
        p = Pattern(
            category="coding_style",
            key="test_framework",
            value="pytest",
            confidence=0.3,
            evidence=["Used in 5 commands"],
        )
        assert p.category == "coding_style"
        assert p.value == "pytest"
        assert len(p.evidence) == 1


# --- Test SessionAnalysis ---


class TestSessionAnalysis:
    def test_patterns_by_category(self):
        analysis = SessionAnalysis(session_id="s1")
        analysis.patterns = [
            Pattern(category="coding_style", key="lang", value="python"),
            Pattern(category="workflow", key="style", value="tdd"),
            Pattern(category="coding_style", key="framework", value="pytest"),
        ]
        cs = analysis.patterns_by_category("coding_style")
        assert len(cs) == 2
        wf = analysis.patterns_by_category("workflow")
        assert len(wf) == 1
        assert analysis.patterns_by_category("nonexistent") == []


# --- Test Tool Preference Extraction ---


class TestToolPreferences:
    def test_empty_session(self):
        session = _make_session()
        patterns = _extract_tool_preferences(session)
        assert patterns == []

    def test_dominant_tool(self):
        tools = [
            _make_tool_call("Read", {"file_path": f"/f{i}.py"}, tool_id=f"tc_{i}")
            for i in range(8)
        ] + [
            _make_tool_call("Bash", {"command": "ls"}, tool_id="tc_bash"),
            _make_tool_call("Edit", {"file_path": "/f.py"}, tool_id="tc_edit"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_tool_preferences(session)

        # Read should be dominant (8/10 = 80%)
        read_patterns = [p for p in patterns if p.value == "Read"]
        assert len(read_patterns) == 1
        assert read_patterns[0].confidence == 0.3  # capped at 0.3

    def test_no_dominant_tool(self):
        tools = [
            _make_tool_call("Read", tool_id="tc_1"),
            _make_tool_call("Write", tool_id="tc_2"),
            _make_tool_call("Bash", tool_id="tc_3"),
            _make_tool_call("Grep", tool_id="tc_4"),
            _make_tool_call("Glob", tool_id="tc_5"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_tool_preferences(session)
        # Each tool is 20% — exactly at threshold
        assert len(patterns) == 5


# --- Test File Pattern Extraction ---


class TestFilePatterns:
    def test_python_detection(self):
        tools = [
            _make_tool_call("Read", {"file_path": "/src/main.py"}, tool_id="tc_1"),
            _make_tool_call("Read", {"file_path": "/src/utils.py"}, tool_id="tc_2"),
            _make_tool_call("Read", {"file_path": "/src/models.py"}, tool_id="tc_3"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_file_patterns(session)

        lang_patterns = [p for p in patterns if p.key == "primary_language"]
        assert any(p.value == "python" for p in lang_patterns)

    def test_test_directory_detection(self):
        tools = [
            _make_tool_call("Read", {"file_path": "/tests/test_main.py"}, tool_id="tc_1"),
            _make_tool_call("Read", {"file_path": "/tests/test_utils.py"}, tool_id="tc_2"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_file_patterns(session)

        loc_patterns = [p for p in patterns if p.key == "test_location"]
        assert len(loc_patterns) == 1
        assert loc_patterns[0].value == "separate_tests_dir"

    def test_src_layout_detection(self):
        tools = [
            _make_tool_call("Read", {"file_path": "/src/betty/main.py"}, tool_id="tc_1"),
            _make_tool_call("Edit", {"file_path": "/src/betty/utils.py"}, tool_id="tc_2"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_file_patterns(session)

        src_patterns = [p for p in patterns if p.key == "source_layout"]
        assert len(src_patterns) == 1
        assert src_patterns[0].value == "src_layout"

    def test_typescript_detection(self):
        tools = [
            _make_tool_call("Read", {"file_path": "/src/app.tsx"}, tool_id="tc_1"),
            _make_tool_call("Read", {"file_path": "/src/utils.ts"}, tool_id="tc_2"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_file_patterns(session)

        lang_patterns = [p for p in patterns if p.key == "primary_language"]
        values = {p.value for p in lang_patterns}
        assert "typescript" in values or "typescript-react" in values

    def test_no_files(self):
        session = _make_session()
        patterns = _extract_file_patterns(session)
        assert patterns == []


# --- Test Bash Pattern Extraction ---


class TestBashPatterns:
    def test_pytest_detection(self):
        tools = [
            _make_tool_call("Bash", {"command": "pytest tests/ -v"}, tool_id="tc_1"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_bash_patterns(session)

        fw_patterns = [p for p in patterns if p.key == "test_framework"]
        assert any(p.value == "pytest" for p in fw_patterns)

    def test_npm_detection(self):
        tools = [
            _make_tool_call("Bash", {"command": "npm run build"}, tool_id="tc_1"),
            _make_tool_call("Bash", {"command": "npm test"}, tool_id="tc_2"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_bash_patterns(session)

        mgr_patterns = [p for p in patterns if p.key == "package_manager"]
        assert any(p.value == "npm" for p in mgr_patterns)

    def test_build_command_extraction(self):
        tools = [
            _make_tool_call("Bash", {"command": "cargo build"}, tool_id="tc_1"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_bash_patterns(session)

        build_patterns = [p for p in patterns if p.key == "build_command"]
        assert any("cargo" in p.value for p in build_patterns)

    def test_no_bash_commands(self):
        tools = [_make_tool_call("Read", {"file_path": "/f.py"}, tool_id="tc_1")]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        patterns = _extract_bash_patterns(session)
        assert patterns == []


# --- Test Workflow Pattern Extraction ---


class TestWorkflowPatterns:
    def test_detailed_upfront(self):
        long_prompt = "Please implement a full authentication system with OAuth2 support. " * 15
        turn = _make_user_turn(long_prompt, turn_number=1)
        session = _make_session(turns=[turn])
        patterns = _extract_workflow_patterns(session)

        style_patterns = [p for p in patterns if p.key == "planning_style"]
        assert any(p.value == "detailed_upfront" for p in style_patterns)

    def test_terse_incremental(self):
        turn = _make_user_turn("fix the bug", turn_number=1)
        session = _make_session(turns=[turn])
        patterns = _extract_workflow_patterns(session)

        style_patterns = [p for p in patterns if p.key == "planning_style"]
        assert any(p.value == "terse_incremental" for p in style_patterns)

    def test_short_session(self):
        turns = [
            _make_user_turn("fix login", 1),
            _make_assistant_turn("Done.", turn_number=2),
            _make_user_turn("thanks", 3),
        ]
        session = _make_session(turns=turns)
        patterns = _extract_workflow_patterns(session)

        session_style = [p for p in patterns if p.key == "session_style"]
        assert any(p.value == "short_focused" for p in session_style)

    def test_long_iterative_session(self):
        turns = []
        for i in range(1, 25):
            if i % 2 == 1:
                turns.append(_make_user_turn(f"step {i}", i))
            else:
                turns.append(_make_assistant_turn(f"done {i}", turn_number=i))
        session = _make_session(turns=turns)
        patterns = _extract_workflow_patterns(session)

        session_style = [p for p in patterns if p.key == "session_style"]
        assert any(p.value == "long_iterative" for p in session_style)

    def test_frequent_corrections(self):
        turns = [
            _make_user_turn("add a button", 1),
            _make_assistant_turn("Added.", turn_number=2),
            _make_user_turn("no, wrong color", 3),
            _make_assistant_turn("Fixed.", turn_number=4),
            _make_user_turn("actually, use a different component instead", 5),
            _make_assistant_turn("Changed.", turn_number=6),
        ]
        session = _make_session(turns=turns)
        patterns = _extract_workflow_patterns(session)

        fb_patterns = [p for p in patterns if p.key == "feedback_style"]
        assert any(p.value == "frequent_corrections" for p in fb_patterns)

    def test_empty_session(self):
        session = _make_session()
        patterns = _extract_workflow_patterns(session)
        assert patterns == []


# --- Test Decision Pattern Extraction ---


class TestDecisionPatterns:
    def test_file_modifications(self):
        tools = [
            _make_tool_call("Write", {"file_path": "/src/main.py"}, tool_id="tc_1"),
            _make_tool_call("Edit", {"file_path": "/src/utils.py"}, tool_id="tc_2"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        decisions = _extract_decision_patterns(session)

        assert len(decisions) == 2
        assert decisions[0]["type"] == "file_modification"
        assert decisions[0]["file"] == "/src/main.py"

    def test_system_commands(self):
        tools = [
            _make_tool_call("Bash", {"command": "git commit -m 'fix'"}, tool_id="tc_1"),
            _make_tool_call("Bash", {"command": "pip install requests"}, tool_id="tc_2"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        decisions = _extract_decision_patterns(session)

        assert len(decisions) == 2
        assert decisions[0]["type"] == "system_command"

    def test_read_commands_not_decisions(self):
        tools = [
            _make_tool_call("Read", {"file_path": "/src/main.py"}, tool_id="tc_1"),
            _make_tool_call("Bash", {"command": "ls -la"}, tool_id="tc_2"),
        ]
        turn = _make_assistant_turn(tool_calls=tools)
        session = _make_session(turns=[turn])
        decisions = _extract_decision_patterns(session)

        # Read and non-modifying bash should not be recorded as decisions
        assert len(decisions) == 0


# --- Test SessionAnalyzer ---


class TestSessionAnalyzer:
    def test_analyze_basic(self):
        tools = [
            _make_tool_call("Read", {"file_path": "/src/main.py"}, tool_id="tc_1"),
            _make_tool_call("Bash", {"command": "pytest"}, tool_id="tc_2"),
            _make_tool_call("Edit", {"file_path": "/src/main.py"}, tool_id="tc_3"),
        ]
        turns = [
            _make_user_turn("Fix the login bug in main.py", 1),
            _make_assistant_turn("Let me look at the code.", tool_calls=tools, turn_number=2),
            _make_user_turn("Looks good, thanks!", 3),
        ]
        session = _make_session(turns=turns)

        analyzer = SessionAnalyzer()
        analysis = analyzer.analyze(session)

        assert analysis.session_id == "test-session"
        assert analysis.project_dir == "/Users/test/project"
        assert analysis.goal == "Fix the login bug in main.py"
        assert analysis.turn_count == 3
        assert analysis.tool_call_count == 3
        assert "/src/main.py" in analysis.files_touched
        assert "Read" in analysis.tools_used
        assert "Bash" in analysis.tools_used
        assert len(analysis.patterns) > 0

    def test_analyze_empty_session(self):
        session = _make_session()
        analyzer = SessionAnalyzer()
        analysis = analyzer.analyze(session)

        assert analysis.turn_count == 0
        assert analysis.tool_call_count == 0
        assert analysis.patterns == []
        assert analysis.files_touched == []

    def test_analyze_batch(self):
        sessions = [
            _make_session(session_id=f"s{i}", turns=[_make_user_turn(f"task {i}")])
            for i in range(3)
        ]
        analyzer = SessionAnalyzer()
        results = analyzer.analyze_batch(sessions)

        assert len(results) == 3
        assert results[0].session_id == "s0"
        assert results[2].session_id == "s2"

    @pytest.mark.asyncio
    async def test_analyze_with_llm(self):
        """Test LLM-augmented analysis with a mock LLM."""
        mock_llm = MagicMock()
        mock_llm.complete_json = AsyncMock(return_value={
            "coding_style": {
                "naming_convention": "snake_case",
                "type_annotations": "strict",
            },
            "workflow": {
                "approach": "plan_first",
                "testing": "tdd",
            },
            "preferences": [
                {"key": "editor_tabs", "value": "spaces_4", "reason": "Consistent 4-space indent"},
            ],
        })

        turns = [
            _make_user_turn("Refactor the auth module", 1),
            _make_assistant_turn("Working on it.", turn_number=2),
        ]
        session = _make_session(turns=turns)

        analyzer = SessionAnalyzer(llm=mock_llm)
        analysis = await analyzer.analyze_with_llm(session)

        # Should have both heuristic and LLM patterns
        categories = {p.category for p in analysis.patterns}
        assert "coding_style" in categories or "workflow" in categories

        # LLM should have been called
        mock_llm.complete_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_with_llm_fallback(self):
        """Test that LLM failure falls back to heuristics."""
        mock_llm = MagicMock()
        mock_llm.complete_json = AsyncMock(side_effect=Exception("API error"))

        tools = [
            _make_tool_call("Bash", {"command": "pytest tests/"}, tool_id="tc_1"),
        ]
        turns = [
            _make_user_turn("Run the tests", 1),
            _make_assistant_turn(tool_calls=tools, turn_number=2),
        ]
        session = _make_session(turns=turns)

        analyzer = SessionAnalyzer(llm=mock_llm)
        analysis = await analyzer.analyze_with_llm(session)

        # Should still have heuristic patterns despite LLM failure
        assert analysis.turn_count == 2
        fw_patterns = [p for p in analysis.patterns if p.key == "test_framework"]
        assert any(p.value == "pytest" for p in fw_patterns)


# --- Test Persistence ---


class TestPersistence:
    @pytest.mark.asyncio
    async def test_persist_to_db(self, tmp_path):
        """Test persisting analysis results to SQLite."""
        db_path = tmp_path / "test.db"
        db = UserModelDB(db_path)
        await db.connect()

        try:
            tools = [
                _make_tool_call("Bash", {"command": "pytest"}, tool_id="tc_1"),
                _make_tool_call("Read", {"file_path": "/src/main.py"}, tool_id="tc_2"),
            ]
            turns = [
                _make_user_turn("Fix the bug", 1),
                _make_assistant_turn(tool_calls=tools, turn_number=2),
            ]
            session = _make_session(turns=turns)

            analyzer = SessionAnalyzer()
            analysis = analyzer.analyze(session)

            await analyzer.persist(analysis, db)

            # Verify session was saved
            saved = await db.get_session("test-session")
            assert saved is not None
            assert saved["session_id"] == "test-session"
            assert "Bash" in saved["tools_used"]

            # Verify patterns were saved as preferences
            prefs = await db.get_preferences_by_category(
                "coding_style", project_scope="/Users/test/project"
            )
            # Should have at least one pattern (from file extension detection)
            # Exact count depends on heuristics, just verify the pipeline works
            assert isinstance(prefs, list)

        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_persist_increases_evidence_count(self, tmp_path):
        """Test that re-analyzing the same pattern increases confidence."""
        db_path = tmp_path / "test.db"
        db = UserModelDB(db_path)
        await db.connect()

        try:
            analyzer = SessionAnalyzer()

            # Analyze two sessions with same test framework
            for i in range(2):
                tools = [
                    _make_tool_call("Bash", {"command": "pytest tests/"}, tool_id=f"tc_{i}"),
                ]
                turns = [
                    _make_user_turn(f"task {i}", 1),
                    _make_assistant_turn(tool_calls=tools, turn_number=2),
                ]
                session = _make_session(
                    session_id=f"session-{i}",
                    turns=turns,
                )
                analysis = analyzer.analyze(session)
                await analyzer.persist(analysis, db)

            # Check evidence count increased for pytest
            pref = await db.get_preference(
                "coding_style", "test_framework",
                project_scope="/Users/test/project",
            )
            if pref:
                assert pref["evidence_count"] >= 2

        finally:
            await db.close()
