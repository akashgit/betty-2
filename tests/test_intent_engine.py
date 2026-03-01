"""Tests for betty.intent_engine — analyze prompts and surface questions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from betty.intent_engine import (
    IntentAnalysis,
    IntentEngine,
    Priority,
    Question,
)


# --- Test Question dataclass ---


class TestQuestion:
    def test_defaults(self):
        q = Question(text="What framework?", reason="No preference set")
        assert q.priority == Priority.MEDIUM
        assert q.options == []

    def test_with_options(self):
        q = Question(
            text="Which auth method?",
            reason="Multiple options available",
            priority=Priority.HIGH,
            options=["JWT", "OAuth2", "Session"],
        )
        assert q.priority == Priority.HIGH
        assert len(q.options) == 3


# --- Test IntentAnalysis ---


class TestIntentAnalysis:
    def test_has_questions(self):
        a = IntentAnalysis()
        assert a.has_questions is False

        a.questions.append(Question(text="Q1", reason="R1"))
        assert a.has_questions is True

    def test_high_priority_questions(self):
        a = IntentAnalysis(questions=[
            Question(text="Q1", reason="R1", priority=Priority.HIGH),
            Question(text="Q2", reason="R2", priority=Priority.MEDIUM),
            Question(text="Q3", reason="R3", priority=Priority.HIGH),
        ])
        high = a.high_priority_questions
        assert len(high) == 2

    def test_build_context_injection_empty(self):
        a = IntentAnalysis()
        assert a.build_context_injection() == ""

    def test_build_context_injection_with_content(self):
        a = IntentAnalysis(
            suggested_context="User prefers pytest",
            questions=[Question(text="Add tests?", reason="No tests mentioned")],
            applicable_policies=[{"rule": "require_tests", "description": "All PRs need tests"}],
        )
        ctx = a.build_context_injection()
        assert "pytest" in ctx
        assert "Add tests?" in ctx
        assert "require_tests" in ctx


# --- Test IntentEngine ---


def _make_mock_user_model(context: str = "User prefers Python and pytest"):
    """Create a mock UserModel."""
    model = MagicMock()
    model.get_relevant_context = AsyncMock(return_value=context)
    return model


def _make_mock_searcher(sessions=None):
    """Create a mock SessionSearcher."""
    searcher = MagicMock()
    if sessions is None:
        sessions = []

    # Create mock SimilarSession objects
    mock_sessions = []
    for s in sessions:
        ms = MagicMock()
        ms.session_id = s.get("session_id", "s1")
        ms.goal = s.get("goal", "test goal")
        ms.outcome = s.get("outcome", "completed")
        ms.relevance_score = s.get("relevance", 0.5)
        mock_sessions.append(ms)

    searcher.find_similar = AsyncMock(return_value=mock_sessions)
    return searcher


def _make_mock_db(policies=None):
    """Create a mock UserModelDB."""
    db = MagicMock()
    db.get_policies = AsyncMock(return_value=policies or [])
    return db


def _make_mock_llm(response=None):
    """Create a mock LLMService."""
    llm = MagicMock()
    if response is None:
        response = {
            "questions": [
                {
                    "text": "Which auth method should be used?",
                    "reason": "Your last auth implementation used JWT but org policy requires OAuth2",
                    "priority": "high",
                    "options": ["JWT", "OAuth2"],
                }
            ],
            "predicted_plan": "Implement OAuth2 authentication with rate limiting",
            "confidence": 0.8,
        }
    llm.complete_json = AsyncMock(return_value=response)
    return llm


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        engine = IntentEngine()
        analysis = await engine.analyze("")
        assert not analysis.has_questions
        assert analysis.confidence == 0.0

    @pytest.mark.asyncio
    async def test_with_llm(self):
        engine = IntentEngine(
            user_model=_make_mock_user_model(),
            searcher=_make_mock_searcher(),
            db=_make_mock_db(),
            llm=_make_mock_llm(),
        )
        analysis = await engine.analyze("add user authentication", "/my/project")

        assert analysis.has_questions
        assert analysis.questions[0].text == "Which auth method should be used?"
        assert analysis.questions[0].priority == Priority.HIGH
        assert analysis.confidence == 0.8
        assert analysis.predicted_plan == "Implement OAuth2 authentication with rate limiting"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_heuristics(self):
        llm = MagicMock()
        llm.complete_json = AsyncMock(side_effect=Exception("API error"))

        engine = IntentEngine(
            user_model=_make_mock_user_model(),
            llm=llm,
        )
        analysis = await engine.analyze("add a new feature")

        # Should have heuristic questions about testing
        assert analysis.has_questions
        assert analysis.confidence == 0.3  # Heuristic confidence

    @pytest.mark.asyncio
    async def test_no_llm_heuristic_only(self):
        engine = IntentEngine(
            user_model=_make_mock_user_model(),
        )
        analysis = await engine.analyze("add a new feature")

        # Heuristic should suggest tests
        test_questions = [q for q in analysis.questions if "test" in q.text.lower()]
        assert len(test_questions) > 0

    @pytest.mark.asyncio
    async def test_bug_fix_heuristic(self):
        engine = IntentEngine()
        analysis = await engine.analyze("fix the login bug")

        # Should suggest regression test
        regression_qs = [q for q in analysis.questions if "regression" in q.text.lower()]
        assert len(regression_qs) > 0

    @pytest.mark.asyncio
    async def test_includes_similar_sessions(self):
        searcher = _make_mock_searcher([
            {"session_id": "s1", "goal": "Add auth module", "outcome": "completed", "relevance": 0.8},
            {"session_id": "s2", "goal": "Fix auth bug", "outcome": "fixed", "relevance": 0.6},
        ])

        engine = IntentEngine(searcher=searcher)
        analysis = await engine.analyze("add authentication", "/my/project")

        assert len(analysis.similar_sessions) == 2
        assert analysis.similar_sessions[0]["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_includes_policies(self):
        db = _make_mock_db(policies=[
            {"rule": "require_tests", "description": "All changes need tests"},
        ])

        engine = IntentEngine(db=db)
        analysis = await engine.analyze("add a feature", "/my/project")

        assert len(analysis.applicable_policies) == 1

    @pytest.mark.asyncio
    async def test_questions_sorted_by_priority(self):
        llm = _make_mock_llm({
            "questions": [
                {"text": "Q1", "reason": "R1", "priority": "low"},
                {"text": "Q2", "reason": "R2", "priority": "high"},
                {"text": "Q3", "reason": "R3", "priority": "medium"},
            ],
            "predicted_plan": "plan",
            "confidence": 0.7,
        })

        engine = IntentEngine(llm=llm)
        analysis = await engine.analyze("do something")

        priorities = [q.priority for q in analysis.questions]
        assert priorities == [Priority.HIGH, Priority.MEDIUM, Priority.LOW]

    @pytest.mark.asyncio
    async def test_user_model_failure_graceful(self):
        user_model = MagicMock()
        user_model.get_relevant_context = AsyncMock(side_effect=Exception("DB error"))

        engine = IntentEngine(user_model=user_model)
        analysis = await engine.analyze("add a feature")

        # Should still work, just without user context
        assert analysis is not None

    @pytest.mark.asyncio
    async def test_context_injection_output(self):
        engine = IntentEngine(
            user_model=_make_mock_user_model("Prefers Python"),
            db=_make_mock_db([{"rule": "use_pytest", "description": "Use pytest for tests"}]),
            llm=_make_mock_llm({
                "questions": [{"text": "Add tests?", "reason": "Good practice", "priority": "medium"}],
                "predicted_plan": "plan",
                "confidence": 0.7,
            }),
        )
        analysis = await engine.analyze("add feature", "/my/project")

        ctx = analysis.build_context_injection()
        assert "Prefers Python" in ctx
        assert "Add tests?" in ctx
        assert "use_pytest" in ctx


class TestHeuristicQuestions:
    @pytest.mark.asyncio
    async def test_add_without_tests(self):
        engine = IntentEngine()
        analysis = await engine.analyze("add a new endpoint")
        test_qs = [q for q in analysis.questions if "test" in q.text.lower()]
        assert len(test_qs) > 0

    @pytest.mark.asyncio
    async def test_add_with_tests_mentioned(self):
        engine = IntentEngine()
        analysis = await engine.analyze("add a new endpoint with tests")
        # Should not ask about tests since user already mentioned them
        test_qs = [q for q in analysis.questions if "Should tests be included" in q.text]
        assert len(test_qs) == 0

    @pytest.mark.asyncio
    async def test_max_three_questions(self):
        engine = IntentEngine()
        # Even with many triggers, cap at 3
        analysis = await engine.analyze("add and fix something")
        assert len(analysis.questions) <= 3

    @pytest.mark.asyncio
    async def test_with_policies_present(self):
        db = _make_mock_db(policies=[
            {"rule": "r1", "description": "d1"},
            {"rule": "r2", "description": "d2"},
        ])
        engine = IntentEngine(db=db)
        analysis = await engine.analyze("implement something")
        policy_qs = [q for q in analysis.questions if "policies" in q.text.lower()]
        assert len(policy_qs) > 0
