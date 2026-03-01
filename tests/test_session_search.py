"""Tests for betty.session_search — find relevant past sessions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from betty.db import UserModelDB
from betty.session_search import (
    SessionSearcher,
    SimilarSession,
    _cosine_similarity,
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


# --- Test Cosine Similarity ---


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0


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

    def test_went_well_positive_outcome(self):
        s = SimilarSession(
            session_id="s1",
            goal="Fix bug",
            relevance_score=0.8,
            outcome="Successfully completed the fix",
        )
        assert len(s.went_well) > 0

    def test_went_well_no_outcome(self):
        s = SimilarSession(
            session_id="s1",
            goal="Fix bug",
            relevance_score=0.8,
        )
        assert s.went_well == []

    def test_went_wrong_negative_outcome(self):
        s = SimilarSession(
            session_id="s1",
            goal="Fix bug",
            relevance_score=0.3,
            outcome="Failed due to missing dependency",
        )
        assert len(s.went_wrong) > 0

    def test_went_wrong_no_issues(self):
        s = SimilarSession(
            session_id="s1",
            goal="Fix bug",
            relevance_score=0.8,
            outcome="All good",
        )
        assert s.went_wrong == []

    def test_key_decisions_field(self):
        s = SimilarSession(
            session_id="s1",
            goal="Fix bug",
            relevance_score=0.8,
            key_decisions=["Edit src/auth.py", "git commit -m fix"],
        )
        assert len(s.key_decisions) == 2


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
        decisions_made=[
            {"type": "file_modification", "tool": "Edit", "file": "src/auth/login.py"}
        ],
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

    @pytest.mark.asyncio
    async def test_global_search_includes_other_projects(self, searcher):
        """Global search finds sessions from other projects."""
        results = await searcher.find_similar(
            "CI/CD pipeline GitHub Actions",
            project_dir="/my/project",
        )
        session_ids = {r.session_id for r in results}
        # s4 is from /other/project and should be found via global search
        assert "s4" in session_ids

    @pytest.mark.asyncio
    async def test_no_duplicates(self, searcher):
        """No duplicate session IDs in results."""
        results = await searcher.find_similar(
            "authentication login",
            project_dir="/my/project",
        )
        session_ids = [r.session_id for r in results]
        assert len(session_ids) == len(set(session_ids))

    @pytest.mark.asyncio
    async def test_key_decisions_populated(self, searcher):
        """Key decisions are extracted from session data."""
        results = await searcher.find_similar(
            "login authentication",
            project_dir="/my/project",
        )
        login_result = next(
            (r for r in results if r.session_id == "s1"), None
        )
        if login_result:
            assert len(login_result.key_decisions) > 0
            assert any("login.py" in d for d in login_result.key_decisions)

    @pytest.mark.asyncio
    async def test_score_capped_at_one(self, searcher):
        """Relevance scores never exceed 1.0."""
        results = await searcher.find_similar(
            "Fix the login authentication bug in auth.py",
            project_dir="/my/project",
        )
        for r in results:
            assert r.relevance_score <= 1.0


# --- Test embedding-based search ---


class TestEmbeddingSearch:
    @pytest.mark.asyncio
    async def test_embedding_search_with_mock_llm(self, tmp_path):
        """Embedding search works with a mock LLM."""
        db = UserModelDB(tmp_path / "test.db")
        await db.connect()

        await db.save_session(
            session_id="s1",
            project_dir="/proj",
            started_at="2025-01-15T10:00:00Z",
            goal="Fix login authentication",
            tools_used=["Edit"],
            files_touched=["src/auth.py"],
        )
        await db.save_session(
            session_id="s2",
            project_dir="/proj",
            started_at="2025-01-14T10:00:00Z",
            goal="Deploy to production server",
            tools_used=["Bash"],
            files_touched=["deploy.sh"],
        )

        mock_llm = MagicMock()

        async def mock_embed(text):
            if "login" in text.lower() or "auth" in text.lower():
                return [1.0, 0.5, 0.0]
            elif "deploy" in text.lower():
                return [0.0, 0.5, 1.0]
            else:
                return [0.3, 0.3, 0.3]

        mock_llm.embed = mock_embed

        searcher = SessionSearcher(db, llm=mock_llm)
        results = await searcher.find_similar(
            "fix auth login",
            project_dir="/proj",
        )

        assert len(results) > 0
        assert results[0].session_id == "s1"

        await db.close()

    @pytest.mark.asyncio
    async def test_embedding_fallback_to_keywords(self, tmp_path):
        """Falls back to keyword search when embedding fails."""
        db = UserModelDB(tmp_path / "test.db")
        await db.connect()

        await db.save_session(
            session_id="s1",
            project_dir="/proj",
            started_at="2025-01-15T10:00:00Z",
            goal="Fix login bug",
            tools_used=["Edit"],
        )

        mock_llm = MagicMock()
        mock_llm.embed = AsyncMock(side_effect=Exception("Service down"))

        searcher = SessionSearcher(db, llm=mock_llm)
        results = await searcher.find_similar(
            "fix login bug",
            project_dir="/proj",
        )

        assert len(results) > 0
        await db.close()


# --- Test transcript-based search ---


class TestTranscriptSearch:
    @pytest.mark.asyncio
    async def test_find_similar_from_transcripts(self, tmp_path):
        """Transcript-based search works with JSONL files."""
        project_dir = "/test/myproject"
        encoded = "-test-myproject"
        session_dir = tmp_path / ".claude" / "projects" / encoded
        session_dir.mkdir(parents=True)

        transcript = session_dir / "session_abc.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"content": "Fix the database connection timeout"},
                "timestamp": "2025-01-20T10:00:00Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "I'll fix the database timeout."},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Edit",
                            "input": {"file_path": "src/db.py"},
                        },
                    ],
                    "model": "claude-sonnet",
                },
                "timestamp": "2025-01-20T10:00:05Z",
            },
        ]
        with open(transcript, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        import betty.session_reader as sr
        original_dir = sr.CLAUDE_PROJECTS_DIR
        sr.CLAUDE_PROJECTS_DIR = tmp_path / ".claude" / "projects"

        try:
            searcher = SessionSearcher(db=MagicMock())
            results = await searcher.find_similar_from_transcripts(
                "database connection issue",
                project_dir=project_dir,
            )

            assert len(results) > 0
            assert results[0].session_id == "session_abc"
            assert results[0].relevance_score > 0.0
        finally:
            sr.CLAUDE_PROJECTS_DIR = original_dir

    @pytest.mark.asyncio
    async def test_transcript_search_empty_prompt(self):
        """Empty prompt returns empty results."""
        searcher = SessionSearcher(db=MagicMock())
        results = await searcher.find_similar_from_transcripts("")
        assert results == []
