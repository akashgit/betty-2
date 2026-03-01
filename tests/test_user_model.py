"""Tests for betty.user_model — preference aggregation and prediction."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from betty.db import UserModelDB
from betty.user_model import (
    CONFIDENCE_ASK,
    CONFIDENCE_SUGGEST,
    MIN_EVIDENCE_FOR_ACTION,
    Prediction,
    UserModel,
    _time_decay_factor,
)


# --- Test Prediction dataclass ---


class TestPrediction:
    def test_should_ask_low_confidence(self):
        p = Prediction(value="v", confidence=0.1)
        assert p.should_ask is True
        assert p.should_suggest is False
        assert p.should_act is False

    def test_should_suggest_medium_confidence(self):
        p = Prediction(value="v", confidence=0.5, evidence_count=5)
        assert p.should_ask is False
        assert p.should_suggest is True
        assert p.should_act is False

    def test_should_act_high_confidence(self):
        p = Prediction(value="v", confidence=0.8, evidence_count=5)
        assert p.should_ask is False
        assert p.should_suggest is False
        assert p.should_act is True

    def test_should_act_requires_evidence(self):
        p = Prediction(value="v", confidence=0.9, evidence_count=1)
        assert p.should_act is False  # Too few observations

    def test_should_act_with_enough_evidence(self):
        p = Prediction(
            value="v",
            confidence=0.8,
            evidence_count=MIN_EVIDENCE_FOR_ACTION,
        )
        assert p.should_act is True

    def test_boundary_ask(self):
        p = Prediction(value="v", confidence=CONFIDENCE_ASK)
        assert p.should_ask is False
        assert p.should_suggest is True

    def test_boundary_suggest(self):
        p = Prediction(value="v", confidence=CONFIDENCE_SUGGEST, evidence_count=10)
        assert p.should_suggest is False
        assert p.should_act is True


# --- Test time decay ---


class TestTimeDecay:
    def test_recent_no_decay(self):
        now = datetime.now(timezone.utc).isoformat()
        factor = _time_decay_factor(now)
        assert factor > 0.99

    def test_90_days_halved(self):
        ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        factor = _time_decay_factor(ninety_days_ago)
        assert 0.45 < factor < 0.55  # Should be ~0.5

    def test_180_days_quarter(self):
        old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        factor = _time_decay_factor(old)
        assert 0.20 < factor < 0.30  # Should be ~0.25

    def test_invalid_timestamp(self):
        factor = _time_decay_factor("not-a-date")
        assert factor == 0.5  # Default for unknown

    def test_future_timestamp(self):
        future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        factor = _time_decay_factor(future)
        assert factor == 1.0  # No decay for future dates


# --- Test UserModel with real DB ---


@pytest.fixture
async def model(tmp_path):
    """Create a UserModel with an in-memory-like SQLite DB."""
    db_path = tmp_path / "test_model.db"
    db = UserModelDB(db_path)
    await db.connect()
    yield UserModel(db)
    await db.close()


@pytest.fixture
async def db_and_model(tmp_path):
    """Return both DB and model for tests that need direct DB access."""
    db_path = tmp_path / "test_model.db"
    db = UserModelDB(db_path)
    await db.connect()
    yield db, UserModel(db)
    await db.close()


class TestPredictPreference:
    @pytest.mark.asyncio
    async def test_no_data(self, model):
        prediction = await model.predict_preference("coding_style", "test_framework")
        assert prediction.value == ""
        assert prediction.confidence == 0.0
        assert prediction.should_ask is True

    @pytest.mark.asyncio
    async def test_global_preference(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("coding_style", "test_framework", "pytest", confidence=0.6)

        prediction = await model.predict_preference("coding_style", "test_framework")
        assert prediction.value == "pytest"
        # Confidence should be close to 0.6 (minimal time decay for just-set value)
        assert prediction.confidence > 0.55

    @pytest.mark.asyncio
    async def test_project_overrides_global(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("coding_style", "test_framework", "pytest", confidence=0.6)
        await db.set_preference(
            "coding_style", "test_framework", "jest",
            confidence=0.8, project_scope="/my/project",
        )

        # Global query returns pytest
        prediction = await model.predict_preference("coding_style", "test_framework")
        assert prediction.value == "pytest"

        # Project query returns jest
        prediction = await model.predict_preference(
            "coding_style", "test_framework",
            project_scope="/my/project",
        )
        assert prediction.value == "jest"
        assert prediction.confidence > 0.7

    @pytest.mark.asyncio
    async def test_project_falls_back_to_global(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("coding_style", "naming", "snake_case", confidence=0.5)

        # Query with project scope but no project-level pref
        prediction = await model.predict_preference(
            "coding_style", "naming",
            project_scope="/other/project",
        )
        assert prediction.value == "snake_case"


class TestConfidenceFor:
    @pytest.mark.asyncio
    async def test_returns_zero_for_unknown(self, model):
        conf = await model.confidence_for("nonexistent", "key")
        assert conf == 0.0

    @pytest.mark.asyncio
    async def test_returns_confidence(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("workflow", "style", "tdd", confidence=0.7)

        conf = await model.confidence_for("workflow", "style")
        assert conf > 0.65


class TestGetPreferences:
    @pytest.mark.asyncio
    async def test_merges_scopes(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("coding_style", "lang", "python", confidence=0.5)
        await db.set_preference("coding_style", "tabs", "spaces", confidence=0.4)
        await db.set_preference(
            "coding_style", "lang", "typescript",
            confidence=0.8, project_scope="/web-project",
        )

        # Global only
        prefs = await model.get_preferences("coding_style")
        assert len(prefs) == 2

        # With project scope, lang should be overridden
        prefs = await model.get_preferences("coding_style", project_scope="/web-project")
        values = {p.value for p in prefs}
        assert "typescript" in values  # Project override
        assert "spaces" in values  # Global fallback


class TestGetProfileSummary:
    @pytest.mark.asyncio
    async def test_no_preferences(self, model):
        summary = await model.get_profile_summary()
        assert summary == "No preferences learned yet."

    @pytest.mark.asyncio
    async def test_with_preferences(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("coding_style", "lang", "python", confidence=0.5)
        await db.set_preference("workflow", "approach", "tdd", confidence=0.8)

        summary = await model.get_profile_summary()
        assert "python" in summary
        assert "tdd" in summary
        assert "Known user preferences:" in summary

    @pytest.mark.asyncio
    async def test_excludes_low_confidence(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("coding_style", "obscure", "value", confidence=0.1)

        summary = await model.get_profile_summary()
        assert summary == "No preferences learned yet."


class TestGetRelevantContext:
    @pytest.mark.asyncio
    async def test_empty_context(self, model):
        context = await model.get_relevant_context("fix the bug")
        assert context == ""

    @pytest.mark.asyncio
    async def test_includes_profile(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("coding_style", "lang", "python", confidence=0.6)

        context = await model.get_relevant_context("add a new feature")
        assert "python" in context

    @pytest.mark.asyncio
    async def test_includes_recent_sessions(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("coding_style", "lang", "python", confidence=0.5)
        await db.save_session(
            session_id="s1",
            project_dir="/my/project",
            started_at="2025-01-15T10:00:00Z",
            goal="Fix login bug",
            outcome="Fixed",
        )

        context = await model.get_relevant_context(
            "add auth feature", project_scope="/my/project"
        )
        assert "Fix login bug" in context


class TestDecayStalePreferences:
    @pytest.mark.asyncio
    async def test_decay_old_preferences(self, db_and_model):
        db, model = db_and_model

        # Insert a preference with an old timestamp manually
        old_time = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        await db.db.execute(
            """
            INSERT INTO user_preferences (category, key, value, confidence, evidence_count, last_seen, project_scope)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("coding_style", "old_pref", "value", 0.8, 5, old_time, ""),
        )
        await db.db.commit()

        updated = await model.decay_stale_preferences()
        # Should have updated the old preference
        assert updated >= 1

    @pytest.mark.asyncio
    async def test_no_decay_for_recent(self, db_and_model):
        db, model = db_and_model
        await db.set_preference("coding_style", "fresh", "value", confidence=0.5)

        updated = await model.decay_stale_preferences()
        # Recent preference shouldn't need updating
        assert updated == 0
