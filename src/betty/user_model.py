"""User model: preference aggregation and prediction engine.

Sits on top of UserModelDB to aggregate patterns from multiple sessions
into coherent predictions. Uses frequency-based statistics with confidence
scoring (not ML for MVP). Supports scoped preferences: global > org > project.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Confidence thresholds
CONFIDENCE_ASK = 0.3  # Below this: don't use, ask the user
CONFIDENCE_SUGGEST = 0.7  # Between ASK and SUGGEST: suggest, let them confirm
# >= SUGGEST: act on behalf of the user (at delegation level 2+)

# Time decay: halve confidence every this many days
DECAY_HALF_LIFE_DAYS = 90

# Minimum evidence count to consider a preference reliable
MIN_EVIDENCE_FOR_ACTION = 3


@dataclass
class Prediction:
    """A predicted user preference with confidence and evidence."""

    value: str
    confidence: float
    evidence_count: int = 0
    evidence: list[str] = field(default_factory=list)

    @property
    def should_ask(self) -> bool:
        """Whether confidence is too low and we should ask the user."""
        return self.confidence < CONFIDENCE_ASK

    @property
    def should_suggest(self) -> bool:
        """Whether we should suggest this to the user for confirmation."""
        return CONFIDENCE_ASK <= self.confidence < CONFIDENCE_SUGGEST

    @property
    def should_act(self) -> bool:
        """Whether confidence is high enough to act autonomously."""
        return self.confidence >= CONFIDENCE_SUGGEST and self.evidence_count >= MIN_EVIDENCE_FOR_ACTION


def _time_decay_factor(last_seen_iso: str) -> float:
    """Calculate time-based decay factor.

    Returns a multiplier between 0 and 1 based on how long ago
    the preference was last observed. Uses exponential decay with
    a 90-day half-life.
    """
    try:
        last_seen = datetime.fromisoformat(last_seen_iso)
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_ago = (now - last_seen).total_seconds() / 86400
        if days_ago <= 0:
            return 1.0
        return math.pow(0.5, days_ago / DECAY_HALF_LIFE_DAYS)
    except (ValueError, TypeError):
        return 0.5  # Unknown age, moderate decay


class UserModel:
    """Preference aggregation and prediction engine.

    Wraps UserModelDB to provide higher-level preference prediction,
    profile summaries, and context generation for LLM injection.

    Usage:
        model = UserModel(db)
        prediction = await model.predict_preference("coding_style", "test_framework")
        context = await model.get_relevant_context("Add a new endpoint", "/Users/me/project")
    """

    def __init__(self, db: Any):
        """Initialize with a connected UserModelDB instance."""
        self._db = db

    async def predict_preference(
        self,
        category: str,
        key: str,
        project_scope: str | None = None,
    ) -> Prediction:
        """Predict a user preference for a given category and key.

        Checks project-scoped preference first, then falls back to global.
        Applies time decay to confidence.
        """
        # Try project-scoped first (more specific = higher priority)
        if project_scope:
            pref = await self._db.get_preference(category, key, project_scope=project_scope)
            if pref:
                decay = _time_decay_factor(pref["last_seen"])
                return Prediction(
                    value=pref["value"],
                    confidence=pref["confidence"] * decay,
                    evidence_count=pref["evidence_count"],
                    evidence=[f"project:{project_scope}"],
                )

        # Fall back to global
        pref = await self._db.get_preference(category, key, project_scope=None)
        if pref:
            decay = _time_decay_factor(pref["last_seen"])
            return Prediction(
                value=pref["value"],
                confidence=pref["confidence"] * decay,
                evidence_count=pref["evidence_count"],
                evidence=["global"],
            )

        # No data
        return Prediction(value="", confidence=0.0)

    async def confidence_for(
        self,
        category: str,
        key: str,
        project_scope: str | None = None,
    ) -> float:
        """Get confidence level for a specific preference."""
        prediction = await self.predict_preference(category, key, project_scope)
        return prediction.confidence

    async def get_preferences(
        self,
        category: str,
        project_scope: str | None = None,
    ) -> list[Prediction]:
        """Get all predictions for a category, merging project and global scopes."""
        predictions: dict[str, Prediction] = {}

        # Get global preferences
        global_prefs = await self._db.get_preferences_by_category(category, project_scope=None)
        for pref in global_prefs:
            decay = _time_decay_factor(pref["last_seen"])
            predictions[pref["key"]] = Prediction(
                value=pref["value"],
                confidence=pref["confidence"] * decay,
                evidence_count=pref["evidence_count"],
                evidence=["global"],
            )

        # Override with project-scoped preferences (higher priority)
        if project_scope:
            project_prefs = await self._db.get_preferences_by_category(category, project_scope=project_scope)
            for pref in project_prefs:
                if pref.get("project_scope") == project_scope:
                    decay = _time_decay_factor(pref["last_seen"])
                    predictions[pref["key"]] = Prediction(
                        value=pref["value"],
                        confidence=pref["confidence"] * decay,
                        evidence_count=pref["evidence_count"],
                        evidence=[f"project:{project_scope}"],
                    )

        return list(predictions.values())

    async def get_profile_summary(
        self,
        project_scope: str | None = None,
    ) -> str:
        """Generate a human-readable summary of known preferences.

        Suitable for injecting into LLM context. Only includes
        preferences with confidence >= CONFIDENCE_ASK.
        """
        categories = [
            "coding_style", "workflow", "tool_preference",
            "project_convention", "preference",
        ]

        lines: list[str] = []
        for category in categories:
            prefs = await self.get_preferences(category, project_scope)
            # Filter to meaningful confidence and sort by confidence
            meaningful = [p for p in prefs if p.confidence >= CONFIDENCE_ASK]
            meaningful.sort(key=lambda p: p.confidence, reverse=True)

            if meaningful:
                label = category.replace("_", " ").title()
                lines.append(f"\n{label}:")
                for p in meaningful:
                    conf_label = "high" if p.should_act else "medium"
                    lines.append(f"  - {p.value} (confidence: {conf_label}, seen {p.evidence_count}x)")

        if not lines:
            return "No preferences learned yet."

        return "Known user preferences:" + "\n".join(lines)

    async def get_relevant_context(
        self,
        prompt: str,
        project_scope: str | None = None,
    ) -> str:
        """Generate context string for a new user prompt.

        Returns preferences and past decisions relevant to the current task.
        This is the key interface for intent amplification -- the returned
        text gets injected into Claude Code sessions.
        """
        parts: list[str] = []

        # Profile summary
        profile = await self.get_profile_summary(project_scope)
        if profile != "No preferences learned yet.":
            parts.append(profile)

        # Recent sessions for this project
        if project_scope:
            sessions = await self._db.get_sessions_for_project(project_scope, limit=5)
            if sessions:
                parts.append("\nRecent sessions in this project:")
                for s in sessions[:5]:
                    goal = s.get("goal", "unknown")
                    outcome = s.get("outcome", "")
                    if goal:
                        line = f"  - {goal}"
                        if outcome:
                            line += f" ({outcome})"
                        parts.append(line)

        # Relevant approval patterns
        if prompt:
            # Extract tool names from prompt if mentioned
            tool_keywords = ["read", "write", "edit", "bash", "grep", "glob"]
            mentioned_tools = [kw for kw in tool_keywords if kw in prompt.lower()]
            for tool in mentioned_tools:
                pattern = await self._db.get_approval_pattern(
                    tool.capitalize(), "*", project_scope
                )
                if pattern:
                    parts.append(
                        f"\nTool approval: {tool.capitalize()} — "
                        f"usually {pattern['decision']} ({pattern['count']}x)"
                    )

        if not parts:
            return ""

        return "\n".join(parts)

    async def decay_stale_preferences(
        self,
        project_scope: str | None = None,
    ) -> int:
        """Reduce confidence of stale preferences.

        Reads all preferences and updates their confidence based on
        time decay. Returns the number of preferences updated.

        This should be called periodically (e.g., daily) to ensure
        old preferences don't dominate.
        """
        categories = [
            "coding_style", "workflow", "tool_preference",
            "project_convention", "preference",
        ]

        updated = 0
        for category in categories:
            prefs = await self._db.get_preferences_by_category(category, project_scope=None)
            if project_scope:
                prefs.extend(
                    await self._db.get_preferences_by_category(category, project_scope=project_scope)
                )

            for pref in prefs:
                decay = _time_decay_factor(pref["last_seen"])
                new_confidence = pref["confidence"] * decay

                # Only update if decay is significant (> 5% change)
                if abs(new_confidence - pref["confidence"]) > 0.05:
                    await self._db.set_preference(
                        category=pref["category"],
                        key=pref["key"],
                        value=pref["value"],
                        confidence=new_confidence,
                        project_scope=pref.get("project_scope"),
                    )
                    updated += 1

        return updated
