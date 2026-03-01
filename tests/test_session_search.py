"""Tests for betty.session_search — find relevant past sessions."""

from __future__ import annotations

import pytest

from betty.db import UserModelDB
from betty.session_search import (
    SessionSearcher,
    SimilarSession,
    _keyword_similarity,
    _tokenize,
)


# --- Test Tokenizer ---


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Fix the login bug")
        assert "fix" in tokens
        assert "login" in tokens
        assert "bug" in tokens
        assert "the" not in tokens  # stopword

    def test_removes_short_words(self):
        tokens = _tokenize("a b c fix")
        assert "fix" in tokens
        assert "a" not in tokens

    def test_handles_special_chars(self):
        tokens = _tokenize("src/main.py: TypeError")
        assert "src" in tokens
        assert "main" in tokens
        assert "py" in tokens
        assert "typeerror" in tokens

    def test_empty(self):
        assert _tokenize("") == []
        assert _tokenize("the a an") == []


# --- Test Keyword Similarity ---


class TestKeywordSimilarity:
    def test_identical(self):
        tokens = _tokenize("fix login bug")
        score = _keyword_similarity(tokens, tokens)
        assert score > 0.8

    def test_partial_overlap(self):
        a = _tokenize("fix login bug")
        b = _tokenize("fix authentication error")
        score = _keyword_similarity(a, b)
        assert 0.1 < score < 0.6

    def test_no_overlap(self):
        a = _tokenize("fix login bug")
        b = _tokenize("deploy production server")
        score = _keyword_similarity(a, b)
        assert score < 0.1

    def test_empty(self):
        assert _keyword_similarity([], ["fix"]) == 0.0
        assert _keyword_similarity(["fix"], []) == 0.0


# --- Test SimilarSession ---


class TestSimilarSession:
    def test_basic(self):
        s = SimilarSession(
            session_id="s1",
            goal="Fix login",
            relevance_score=0.85,
        )
        assert s.session_id == "s1"
        assert s.relevance_score == 0.85
        assert s.tools_used == []


# --- Test SessionSearcher with DB ---


@pytest.fixture
async def searcher(tmp_path):
    """Create a SessionSearcher with a real DB and some sessions."""
    db_path = tmp_path / "test_search.db"
    db = UserModelDB(db_path)
    await db.connect()

    # Seed with some sessions
    await db.save_session(
        session_id="s1",
        project_dir="/my/project",
        started_at="2025-01-15T10:00:00Z",
        goal="Fix the login authentication bug in auth.py",
        outcome="Fixed",
        tools_used=["Read", "Edit", "Bash"],
        files_touched=["src/auth/login.py", "tests/test_auth.py"],
    )
    await db.save_session(
        session_id="s2",
        project_dir="/my/project",
        started_at="2025-01-14T10:00:00Z",
        goal="Add new REST API endpoint for user profiles",
        outcome="Completed",
        tools_used=["Read", "Write", "Bash"],
        files_touched=["src/api/users.py", "tests/test_api.py"],
    )
    await db.save_session(
        session_id="s3",
        project_dir="/my/project",
        started_at="2025-01-13T10:00:00Z",
        goal="Refactor database connection pooling",
        outcome="Completed",
        tools_used=["Read", "Edit"],
        files_touched=["src/db/pool.py"],
    )
    await db.save_session(
        session_id="s4",
        project_dir="/other/project",
        started_at="2025-01-12T10:00:00Z",
        goal="Setup CI/CD pipeline with GitHub Actions",
        outcome="Completed",
        tools_used=["Write", "Bash"],
        files_touched=[".github/workflows/ci.yml"],
    )

    yield SessionSearcher(db)
    await db.close()


class TestFindSimilar:
    @pytest.mark.asyncio
    async def test_finds_relevant_session(self, searcher):
        results = await searcher.find_similar(
            "fix the authentication bug",
            project_dir="/my/project",
        )
        assert len(results) > 0
        # The login auth session should be the best match
        assert results[0].session_id == "s1"
        assert results[0].relevance_score > 0.1

    @pytest.mark.asyncio
    async def test_ranks_by_relevance(self, searcher):
        results = await searcher.find_similar(
            "add a new API endpoint",
            project_dir="/my/project",
        )
        assert len(results) > 0
        # Should be sorted by relevance
        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score

    @pytest.mark.asyncio
    async def test_respects_limit(self, searcher):
        results = await searcher.find_similar(
            "fix something",
            project_dir="/my/project",
            limit=2,
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_empty_prompt(self, searcher):
        results = await searcher.find_similar("", project_dir="/my/project")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_matching_sessions(self, searcher):
        results = await searcher.find_similar(
            "quantum computing simulation",
            project_dir="/my/project",
        )
        # May have some low-relevance results or none
        for r in results:
            assert r.relevance_score < 0.5

    @pytest.mark.asyncio
    async def test_project_scoped(self, searcher):
        results = await searcher.find_similar(
            "database connection",
            project_dir="/my/project",
        )
        # Should find the database session
        session_ids = {r.session_id for r in results}
        assert "s3" in session_ids

    @pytest.mark.asyncio
    async def test_result_includes_metadata(self, searcher):
        results = await searcher.find_similar(
            "fix login auth",
            project_dir="/my/project",
        )
        if results:
            r = results[0]
            assert r.goal
            assert isinstance(r.tools_used, list)
            assert isinstance(r.files_touched, list)
