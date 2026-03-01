"""Tests for betty.claude_md — CLAUDE.md auto-maintainer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from betty.claude_md import (
    ClaudeMdMaintainer,
    SuggestedUpdate,
    UpdateType,
    _content_already_present,
    _parse_sections,
    _suggest_permission_rules,
)


# -- Fake user model for testing ------------------------------------------


@dataclass
class FakePrediction:
    value: str
    confidence: float
    evidence_count: int = 3
    evidence: list[str] = field(default_factory=lambda: ["session-1"])


class FakeUserModel:
    """Minimal user model mock for testing."""

    def __init__(self, preferences: dict[str, list[FakePrediction]] | None = None):
        self._preferences = preferences or {}

    async def get_preferences(self, category: str, project_scope: str | None = None):
        return self._preferences.get(category, [])

    async def predict_preference(self, category: str, key: str, project_scope: str | None = None):
        prefs = self._preferences.get(category, [])
        for p in prefs:
            if p.value == key:
                return p
        return FakePrediction(value="", confidence=0.0, evidence=[])


# -- _parse_sections ------------------------------------------------------


class TestParseSections:
    def test_basic_sections(self):
        content = """# Title

## Commands

Run tests with pytest.

## Rules

Never use print().
"""
        sections = _parse_sections(content)
        assert "## Commands" in sections
        assert "## Rules" in sections
        assert "pytest" in sections["## Commands"]
        assert "print" in sections["## Rules"]

    def test_empty_content(self):
        assert _parse_sections("") == {}

    def test_no_sections(self):
        sections = _parse_sections("Just plain text\nno headings here")
        assert sections == {}

    def test_single_section(self):
        sections = _parse_sections("## Only One\n\nSome content")
        assert len(sections) == 1
        assert "Some content" in sections["## Only One"]


# -- _content_already_present ---------------------------------------------


class TestContentAlreadyPresent:
    def test_exact_match(self):
        assert _content_already_present("use pytest for tests", "use pytest for tests")

    def test_case_insensitive(self):
        assert _content_already_present("Use Pytest", "use pytest")

    def test_whitespace_normalized(self):
        assert _content_already_present("use  pytest   for tests", "use pytest for tests")

    def test_substring(self):
        assert _content_already_present(
            "Always use pytest for testing. Run with -x flag.",
            "use pytest for testing",
        )

    def test_not_present(self):
        assert not _content_already_present("use unittest", "use pytest")


# -- _suggest_permission_rules --------------------------------------------


class TestSuggestPermissionRules:
    def test_high_count_generates_suggestion(self):
        patterns = [
            {"tool_name": "Write", "action_pattern": "write:src/*.py", "count": 10},
        ]
        suggestions = _suggest_permission_rules(patterns)
        assert len(suggestions) == 1
        assert suggestions[0].target_file == ".claude/settings.json"
        assert suggestions[0].confidence >= 0.7

    def test_low_count_skipped(self):
        patterns = [
            {"tool_name": "Write", "action_pattern": "write:src/*.py", "count": 1},
        ]
        suggestions = _suggest_permission_rules(patterns)
        assert len(suggestions) == 0

    def test_medium_count_skipped(self):
        patterns = [
            {"tool_name": "Write", "action_pattern": "write:src/*.py", "count": 3},
        ]
        # count=3 => confidence=0.8, which is >= 0.7
        suggestions = _suggest_permission_rules(patterns)
        assert len(suggestions) == 1

    def test_empty_patterns(self):
        assert _suggest_permission_rules([]) == []


# -- SuggestedUpdate -------------------------------------------------------


class TestSuggestedUpdate:
    def test_should_auto_apply_high_confidence(self):
        update = SuggestedUpdate(
            update_type=UpdateType.BUILD_TEST,
            section="## Commands",
            content="- `uv run pytest -x`",
            confidence=0.9,
            evidence=["s1", "s2", "s3"],
        )
        assert update.should_auto_apply is True

    def test_should_not_auto_apply_low_confidence(self):
        update = SuggestedUpdate(
            update_type=UpdateType.BUILD_TEST,
            section="## Commands",
            content="- `uv run pytest -x`",
            confidence=0.5,
            evidence=["s1"],
        )
        assert update.should_auto_apply is False

    def test_should_not_auto_apply_few_evidence(self):
        update = SuggestedUpdate(
            update_type=UpdateType.BUILD_TEST,
            section="## Commands",
            content="- `uv run pytest -x`",
            confidence=0.9,
            evidence=["s1"],
        )
        assert update.should_auto_apply is False


# -- ClaudeMdMaintainer.suggest_updates ------------------------------------


class TestSuggestUpdates:
    @pytest.mark.asyncio
    async def test_suggests_new_preference(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n\n## Development Commands\n\nExisting stuff.\n")
        model = FakeUserModel({
            "workflow": [FakePrediction("uv run pytest -x", 0.8, evidence=["s1", "s2"])],
        })
        maintainer = ClaudeMdMaintainer()
        suggestions = await maintainer.suggest_updates(str(tmp_path), model)

        assert len(suggestions) == 1
        assert suggestions[0].update_type == UpdateType.BUILD_TEST
        assert "pytest -x" in suggestions[0].content

    @pytest.mark.asyncio
    async def test_skips_existing_content(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text(
            "# CLAUDE.md\n\n## Development Commands\n\n- `uv run pytest -x`\n"
        )
        model = FakeUserModel({
            "workflow": [FakePrediction("uv run pytest -x", 0.8)],
        })
        maintainer = ClaudeMdMaintainer()
        suggestions = await maintainer.suggest_updates(str(tmp_path), model)
        assert len(suggestions) == 0

    @pytest.mark.asyncio
    async def test_skips_low_confidence(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n")
        model = FakeUserModel({
            "workflow": [FakePrediction("some command", 0.1)],
        })
        maintainer = ClaudeMdMaintainer()
        suggestions = await maintainer.suggest_updates(str(tmp_path), model)
        assert len(suggestions) == 0

    @pytest.mark.asyncio
    async def test_no_claude_md_file(self, tmp_path: Path):
        model = FakeUserModel({
            "coding_style": [FakePrediction("prefer snake_case", 0.9)],
        })
        maintainer = ClaudeMdMaintainer()
        suggestions = await maintainer.suggest_updates(str(tmp_path), model)
        assert len(suggestions) == 1

    @pytest.mark.asyncio
    async def test_multiple_categories(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n")
        model = FakeUserModel({
            "workflow": [FakePrediction("uv run pytest", 0.8)],
            "coding_style": [FakePrediction("use dataclasses", 0.7)],
            "preference": [FakePrediction("never use print()", 0.9)],
        })
        maintainer = ClaudeMdMaintainer()
        suggestions = await maintainer.suggest_updates(str(tmp_path), model)
        assert len(suggestions) == 3

    @pytest.mark.asyncio
    async def test_sorted_by_confidence(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n")
        model = FakeUserModel({
            "workflow": [FakePrediction("low conf cmd", 0.4)],
            "coding_style": [FakePrediction("high conf style", 0.95)],
        })
        maintainer = ClaudeMdMaintainer()
        suggestions = await maintainer.suggest_updates(str(tmp_path), model)
        assert len(suggestions) == 2
        assert suggestions[0].confidence > suggestions[1].confidence

    @pytest.mark.asyncio
    async def test_includes_approval_patterns(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n")
        model = FakeUserModel()
        patterns = [
            {"tool_name": "Edit", "action_pattern": "edit:src/*.py", "count": 10},
        ]
        maintainer = ClaudeMdMaintainer()
        suggestions = await maintainer.suggest_updates(str(tmp_path), model, patterns)
        settings_suggestions = [s for s in suggestions if s.target_file == ".claude/settings.json"]
        assert len(settings_suggestions) == 1

    @pytest.mark.asyncio
    async def test_empty_value_skipped(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n")
        model = FakeUserModel({
            "workflow": [FakePrediction("", 0.8)],
        })
        maintainer = ClaudeMdMaintainer()
        suggestions = await maintainer.suggest_updates(str(tmp_path), model)
        assert len(suggestions) == 0


# -- ClaudeMdMaintainer.apply_updates -------------------------------------


class TestApplyUpdates:
    def test_apply_to_existing_section(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text(
            "# CLAUDE.md\n\n## Development Commands\n\nRun stuff.\n"
        )
        updates = [
            SuggestedUpdate(
                update_type=UpdateType.BUILD_TEST,
                section="## Development Commands",
                content="- `uv run pytest -x`",
                confidence=0.9,
                evidence=["s1", "s2", "s3"],
            ),
        ]
        maintainer = ClaudeMdMaintainer()
        applied = maintainer.apply_updates(str(tmp_path), updates)
        assert len(applied) == 1

        content = (tmp_path / "CLAUDE.md").read_text()
        assert "uv run pytest -x" in content
        assert "Run stuff." in content  # Preserved.

    def test_apply_new_section(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n\n## Existing\n\nStuff.\n")
        updates = [
            SuggestedUpdate(
                update_type=UpdateType.RULE,
                section="## Rules",
                content="- Never use print() for logging",
                confidence=0.9,
                evidence=["s1", "s2", "s3"],
            ),
        ]
        maintainer = ClaudeMdMaintainer()
        applied = maintainer.apply_updates(str(tmp_path), updates)
        assert len(applied) == 1

        content = (tmp_path / "CLAUDE.md").read_text()
        assert "## Rules" in content
        assert "Never use print()" in content

    def test_apply_creates_claude_md(self, tmp_path: Path):
        updates = [
            SuggestedUpdate(
                update_type=UpdateType.CONVENTION,
                section="## Coding Conventions",
                content="- Use snake_case for variables",
                confidence=0.8,
                evidence=["s1", "s2", "s3"],
            ),
        ]
        maintainer = ClaudeMdMaintainer()
        applied = maintainer.apply_updates(str(tmp_path), updates)
        assert len(applied) == 1
        assert (tmp_path / "CLAUDE.md").exists()

    def test_auto_mode_filters_low_confidence(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n")
        updates = [
            SuggestedUpdate(
                update_type=UpdateType.BUILD_TEST,
                section="## Development Commands",
                content="- `pytest`",
                confidence=0.5,
                evidence=["s1"],
            ),
        ]
        maintainer = ClaudeMdMaintainer()
        applied = maintainer.apply_updates(str(tmp_path), updates, auto=True)
        assert len(applied) == 0

    def test_auto_mode_applies_high_confidence(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n")
        updates = [
            SuggestedUpdate(
                update_type=UpdateType.BUILD_TEST,
                section="## Development Commands",
                content="- `pytest -x`",
                confidence=0.9,
                evidence=["s1", "s2", "s3"],
            ),
        ]
        maintainer = ClaudeMdMaintainer()
        applied = maintainer.apply_updates(str(tmp_path), updates, auto=True)
        assert len(applied) == 1

    def test_no_duplicate_application(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text(
            "# CLAUDE.md\n\n## Development Commands\n\n- `pytest -x`\n"
        )
        updates = [
            SuggestedUpdate(
                update_type=UpdateType.BUILD_TEST,
                section="## Development Commands",
                content="- `pytest -x`",
                confidence=0.9,
                evidence=["s1", "s2", "s3"],
            ),
        ]
        maintainer = ClaudeMdMaintainer()
        applied = maintainer.apply_updates(str(tmp_path), updates)
        assert len(applied) == 0

    def test_applied_count(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# CLAUDE.md\n")
        maintainer = ClaudeMdMaintainer()
        assert maintainer.applied_count == 0

        updates = [
            SuggestedUpdate(
                update_type=UpdateType.RULE,
                section="## Rules",
                content="- Use logger, not print",
                confidence=0.9,
                evidence=["s1", "s2", "s3"],
            ),
        ]
        maintainer.apply_updates(str(tmp_path), updates)
        assert maintainer.applied_count == 1


# -- Settings.json application --------------------------------------------


class TestApplySettingsJson:
    def test_creates_settings_file(self, tmp_path: Path):
        updates = [
            SuggestedUpdate(
                update_type=UpdateType.RULE,
                section="allowedTools",
                content=json.dumps({"tool": "Edit", "pattern": "edit:src/*.py", "auto_approve": True}),
                confidence=0.9,
                evidence=["seen 10 times"],
                target_file=".claude/settings.json",
            ),
        ]
        maintainer = ClaudeMdMaintainer()
        applied = maintainer.apply_updates(str(tmp_path), updates)
        assert len(applied) == 1

        settings_path = tmp_path / ".claude" / "settings.json"
        assert settings_path.exists()
        data = json.loads(settings_path.read_text())
        assert "Edit:edit:src/*.py" in data["permissions"]["allow"]

    def test_preserves_existing_settings(self, tmp_path: Path):
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings_path = settings_dir / "settings.json"
        settings_path.write_text(json.dumps({"model": "opus", "permissions": {"allow": ["Read:*"]}}))

        updates = [
            SuggestedUpdate(
                update_type=UpdateType.RULE,
                section="allowedTools",
                content=json.dumps({"tool": "Write", "pattern": "write:src/*.py", "auto_approve": True}),
                confidence=0.9,
                evidence=["seen 10 times"],
                target_file=".claude/settings.json",
            ),
        ]
        maintainer = ClaudeMdMaintainer()
        applied = maintainer.apply_updates(str(tmp_path), updates)
        assert len(applied) == 1

        data = json.loads(settings_path.read_text())
        assert data["model"] == "opus"  # Preserved.
        assert "Read:*" in data["permissions"]["allow"]  # Preserved.
        assert "Write:write:src/*.py" in data["permissions"]["allow"]  # Added.

    def test_no_duplicate_rules(self, tmp_path: Path):
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings_path = settings_dir / "settings.json"
        settings_path.write_text(json.dumps({
            "permissions": {"allow": ["Edit:edit:src/*.py"]}
        }))

        updates = [
            SuggestedUpdate(
                update_type=UpdateType.RULE,
                section="allowedTools",
                content=json.dumps({"tool": "Edit", "pattern": "edit:src/*.py", "auto_approve": True}),
                confidence=0.9,
                evidence=["seen 10 times"],
                target_file=".claude/settings.json",
            ),
        ]
        maintainer = ClaudeMdMaintainer()
        applied = maintainer.apply_updates(str(tmp_path), updates)
        assert len(applied) == 0


# -- Format suggestion ----------------------------------------------------


class TestFormatSuggestion:
    def test_command_formatted(self):
        result = ClaudeMdMaintainer._format_suggestion(
            UpdateType.BUILD_TEST, "uv run pytest -x"
        )
        assert result == "- `uv run pytest -x`"

    def test_non_command_formatted(self):
        result = ClaudeMdMaintainer._format_suggestion(
            UpdateType.BUILD_TEST, "Always run tests before committing"
        )
        assert result == "- Always run tests before committing"

    def test_rule_formatted(self):
        result = ClaudeMdMaintainer._format_suggestion(
            UpdateType.RULE, "Never use print() for logging"
        )
        assert result == "- Never use print() for logging"

    def test_convention_formatted(self):
        result = ClaudeMdMaintainer._format_suggestion(
            UpdateType.CONVENTION, "Use dataclasses over named tuples"
        )
        assert result == "- Use dataclasses over named tuples"
