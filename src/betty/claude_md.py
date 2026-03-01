"""CLAUDE.md auto-maintainer — update project docs with learned preferences.

Reads existing CLAUDE.md files, compares against the user model, and
suggests additions.  Never removes existing content (diff-based, additions
only).  Each suggestion carries a confidence score and evidence trail.

Also handles .claude/settings.json for learned permission rules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class UpdateType(Enum):
    """Category of CLAUDE.md update."""

    BUILD_TEST = "build_test"  # Build/test commands
    CONVENTION = "convention"  # Coding conventions
    ARCHITECTURE = "architecture"  # Architecture notes
    RULE = "rule"  # Do/don't rules


@dataclass
class SuggestedUpdate:
    """A single suggested addition to CLAUDE.md or settings.json."""

    update_type: UpdateType
    section: str  # Target section heading (e.g., "## Development Commands")
    content: str  # The text to add
    confidence: float  # 0.0 - 1.0
    evidence: list[str] = field(default_factory=list)
    target_file: str = "CLAUDE.md"  # "CLAUDE.md" or ".claude/settings.json"

    @property
    def should_auto_apply(self) -> bool:
        """Whether confidence is high enough for auto-apply (delegation >= 2)."""
        return self.confidence >= 0.7 and len(self.evidence) >= 3


# -- Section definitions for CLAUDE.md ------------------------------------

# Standard sections and what maps to them.
SECTION_MAP: dict[UpdateType, str] = {
    UpdateType.BUILD_TEST: "## Development Commands",
    UpdateType.CONVENTION: "## Coding Conventions",
    UpdateType.ARCHITECTURE: "## Architecture",
    UpdateType.RULE: "## Rules",
}

# Categories in the user model that map to update types.
CATEGORY_TO_TYPE: dict[str, UpdateType] = {
    "workflow": UpdateType.BUILD_TEST,
    "coding_style": UpdateType.CONVENTION,
    "project_convention": UpdateType.ARCHITECTURE,
    "preference": UpdateType.RULE,
    "tool_preference": UpdateType.BUILD_TEST,
}


# -- CLAUDE.md parsing ----------------------------------------------------


def _parse_sections(content: str) -> dict[str, str]:
    """Parse a CLAUDE.md into sections keyed by heading.

    Returns a dict mapping heading text (e.g., "## Development Commands")
    to the body content under that heading.
    """
    sections: dict[str, str] = {}
    current_heading = ""
    current_lines: list[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            if current_heading:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def _content_already_present(existing: str, new_content: str) -> bool:
    """Check if the suggested content is already in the existing text.

    Uses normalized comparison to avoid duplicating content that
    differs only in whitespace.
    """
    existing_norm = " ".join(existing.lower().split())
    new_norm = " ".join(new_content.lower().split())
    return new_norm in existing_norm


# -- Settings.json handling -----------------------------------------------


def _suggest_permission_rules(
    approval_patterns: list[dict[str, Any]],
) -> list[SuggestedUpdate]:
    """Suggest .claude/settings.json permission rules from learned approvals.

    Takes approval patterns that have been accepted enough times to be
    high-confidence and suggests adding them as allow rules.
    """
    suggestions: list[SuggestedUpdate] = []

    for pattern in approval_patterns:
        count = pattern.get("count", 0)
        if count < 3:
            continue

        tool_name = pattern.get("tool_name", "")
        action_pattern = pattern.get("action_pattern", "")
        confidence = min(1.0, 0.5 + count * 0.1)

        if confidence < 0.7:
            continue

        content = json.dumps({
            "tool": tool_name,
            "pattern": action_pattern,
            "auto_approve": True,
        }, indent=2)

        suggestions.append(SuggestedUpdate(
            update_type=UpdateType.RULE,
            section="allowedTools",
            content=content,
            confidence=confidence,
            evidence=[f"Approved {count} times"],
            target_file=".claude/settings.json",
        ))

    return suggestions


# -- Main class -----------------------------------------------------------


class ClaudeMdMaintainer:
    """Maintains CLAUDE.md and .claude/settings.json based on learned preferences.

    Usage:
        maintainer = ClaudeMdMaintainer()
        suggestions = await maintainer.suggest_updates("/path/to/project", user_model)
        maintainer.apply_updates("/path/to/project", approved_suggestions)
    """

    def __init__(self) -> None:
        self._applied_count = 0

    async def suggest_updates(
        self,
        project_dir: str,
        user_model: Any,
        approval_patterns: list[dict[str, Any]] | None = None,
    ) -> list[SuggestedUpdate]:
        """Analyze current CLAUDE.md against learned preferences.

        Args:
            project_dir: Path to the project directory.
            user_model: UserModel instance for preference queries.
            approval_patterns: Optional list of approval pattern dicts
                from ApprovalModel.get_auto_approve_rules().

        Returns:
            List of suggested additions, each with confidence and evidence.
        """
        project_path = Path(project_dir)
        suggestions: list[SuggestedUpdate] = []

        # Read existing CLAUDE.md.
        claude_md_path = project_path / "CLAUDE.md"
        existing_content = ""
        if claude_md_path.exists():
            existing_content = claude_md_path.read_text()
        existing_sections = _parse_sections(existing_content)

        # Query user model for preferences in each category.
        for category, update_type in CATEGORY_TO_TYPE.items():
            predictions = await user_model.get_preferences(
                category, project_scope=project_dir,
            )

            for pred in predictions:
                if pred.confidence < 0.3:
                    continue
                if not pred.value:
                    continue

                section = SECTION_MAP[update_type]
                section_content = existing_sections.get(section, "")

                if _content_already_present(section_content, pred.value):
                    continue
                if _content_already_present(existing_content, pred.value):
                    continue

                content = self._format_suggestion(update_type, pred.value)

                suggestions.append(SuggestedUpdate(
                    update_type=update_type,
                    section=section,
                    content=content,
                    confidence=pred.confidence,
                    evidence=pred.evidence[:5],
                ))

        # Settings.json permission rules from approval patterns.
        if approval_patterns:
            suggestions.extend(_suggest_permission_rules(approval_patterns))

        # Sort by confidence descending.
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return suggestions

    def apply_updates(
        self,
        project_dir: str,
        updates: list[SuggestedUpdate],
        auto: bool = False,
    ) -> list[SuggestedUpdate]:
        """Apply approved updates to CLAUDE.md and/or settings.json.

        Args:
            project_dir: Path to the project directory.
            updates: List of approved SuggestedUpdate items.
            auto: If True, only apply updates with should_auto_apply == True.

        Returns:
            List of updates that were actually applied.
        """
        project_path = Path(project_dir)
        applied: list[SuggestedUpdate] = []

        # Separate by target file.
        claude_md_updates = [u for u in updates if u.target_file == "CLAUDE.md"]
        settings_updates = [u for u in updates if u.target_file == ".claude/settings.json"]

        # Apply CLAUDE.md updates.
        if claude_md_updates:
            filtered = claude_md_updates if not auto else [
                u for u in claude_md_updates if u.should_auto_apply
            ]
            if filtered:
                result = self._apply_claude_md(project_path, filtered)
                applied.extend(result)

        # Apply settings.json updates.
        if settings_updates:
            filtered = settings_updates if not auto else [
                u for u in settings_updates if u.should_auto_apply
            ]
            if filtered:
                result = self._apply_settings_json(project_path, filtered)
                applied.extend(result)

        self._applied_count += len(applied)
        return applied

    @property
    def applied_count(self) -> int:
        """Total number of updates applied in this session."""
        return self._applied_count

    def _apply_claude_md(
        self,
        project_path: Path,
        updates: list[SuggestedUpdate],
    ) -> list[SuggestedUpdate]:
        """Apply updates to CLAUDE.md, adding to existing sections or appending new ones."""
        claude_md_path = project_path / "CLAUDE.md"

        if claude_md_path.exists():
            content = claude_md_path.read_text()
        else:
            content = "# CLAUDE.md\n\nThis file provides guidance to Claude Code.\n"

        existing_sections = _parse_sections(content)
        applied: list[SuggestedUpdate] = []

        # Group updates by section.
        by_section: dict[str, list[SuggestedUpdate]] = {}
        for update in updates:
            by_section.setdefault(update.section, []).append(update)

        for section_heading, section_updates in by_section.items():
            additions = []
            for update in section_updates:
                section_body = existing_sections.get(section_heading, "")
                if not _content_already_present(section_body, update.content):
                    additions.append(update.content)
                    applied.append(update)

            if not additions:
                continue

            addition_block = "\n".join(additions)

            if section_heading in existing_sections:
                # Append to existing section.
                # Find the section in the content and append after it.
                idx = content.find(section_heading)
                if idx != -1:
                    # Find the end of the section (next ## or end of file).
                    next_section = content.find("\n## ", idx + len(section_heading))
                    if next_section == -1:
                        # Append at end.
                        content = content.rstrip() + "\n" + addition_block + "\n"
                    else:
                        # Insert before the next section.
                        content = (
                            content[:next_section].rstrip()
                            + "\n"
                            + addition_block
                            + "\n\n"
                            + content[next_section + 1:]  # skip the leading \n
                        )
            else:
                # Add new section at end.
                content = content.rstrip() + "\n\n" + section_heading + "\n\n" + addition_block + "\n"

        if applied:
            claude_md_path.write_text(content)
            logger.info("Applied %d updates to %s", len(applied), claude_md_path)

        return applied

    def _apply_settings_json(
        self,
        project_path: Path,
        updates: list[SuggestedUpdate],
    ) -> list[SuggestedUpdate]:
        """Apply permission rules to .claude/settings.json."""
        settings_dir = project_path / ".claude"
        settings_path = settings_dir / "settings.json"

        if settings_path.exists():
            data = json.loads(settings_path.read_text())
        else:
            data = {}

        applied: list[SuggestedUpdate] = []

        for update in updates:
            try:
                rule = json.loads(update.content)
            except json.JSONDecodeError:
                continue

            # Add to permissions.allow list.
            allow_list = data.setdefault("permissions", {}).setdefault("allow", [])

            # Check for duplicates.
            tool = rule.get("tool", "")
            pattern = rule.get("pattern", "")
            entry = f"{tool}:{pattern}"

            if entry not in allow_list:
                allow_list.append(entry)
                applied.append(update)

        if applied:
            settings_dir.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(json.dumps(data, indent=2) + "\n")
            logger.info("Applied %d permission rules to %s", len(applied), settings_path)

        return applied

    @staticmethod
    def _format_suggestion(update_type: UpdateType, value: str) -> str:
        """Format a preference value as CLAUDE.md content."""
        if update_type == UpdateType.BUILD_TEST:
            # Wrap commands in code blocks if they look like commands.
            if value.startswith(("uv ", "npm ", "pip ", "python ", "pytest ", "make ")):
                return f"- `{value}`"
            return f"- {value}"
        if update_type == UpdateType.RULE:
            return f"- {value}"
        if update_type == UpdateType.CONVENTION:
            return f"- {value}"
        if update_type == UpdateType.ARCHITECTURE:
            return f"- {value}"
        return f"- {value}"
