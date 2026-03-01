"""Tests for betty.db module."""

import pytest
import pytest_asyncio

from betty.db import UserModelDB


@pytest_asyncio.fixture
async def db(tmp_path):
    db = UserModelDB(tmp_path / "test.db")
    await db.connect()
    yield db
    await db.close()


class TestDatabaseSetup:
    @pytest.mark.asyncio
    async def test_creates_database(self, tmp_path):
        db_path = tmp_path / "new.db"
        db = UserModelDB(db_path)
        await db.connect()
        assert db_path.exists()
        await db.close()

    @pytest.mark.asyncio
    async def test_wal_mode(self, db):
        async with db.db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
            assert row[0] == "wal"

    @pytest.mark.asyncio
    async def test_schema_version(self, db):
        async with db.db.execute("SELECT MAX(version) FROM schema_version") as cursor:
            row = await cursor.fetchone()
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path):
        async with UserModelDB(tmp_path / "ctx.db") as db:
            await db.set_preference("test", "key", "value")
            pref = await db.get_preference("test", "key")
            assert pref["value"] == "value"

    @pytest.mark.asyncio
    async def test_idempotent_migration(self, tmp_path):
        db_path = tmp_path / "idem.db"
        db1 = UserModelDB(db_path)
        await db1.connect()
        await db1.close()
        db2 = UserModelDB(db_path)
        await db2.connect()
        async with db2.db.execute("SELECT MAX(version) FROM schema_version") as cursor:
            row = await cursor.fetchone()
            assert row[0] == 1
        await db2.close()


class TestPreferences:
    @pytest.mark.asyncio
    async def test_set_and_get(self, db):
        await db.set_preference("editor", "theme", "dark", confidence=0.9)
        pref = await db.get_preference("editor", "theme")
        assert pref is not None
        assert pref["value"] == "dark"
        assert pref["confidence"] == 0.9
        assert pref["evidence_count"] == 1

    @pytest.mark.asyncio
    async def test_upsert_increments_evidence(self, db):
        await db.set_preference("editor", "theme", "dark")
        await db.set_preference("editor", "theme", "dark")
        pref = await db.get_preference("editor", "theme")
        assert pref["evidence_count"] == 2

    @pytest.mark.asyncio
    async def test_upsert_updates_value(self, db):
        await db.set_preference("editor", "theme", "dark")
        await db.set_preference("editor", "theme", "light", confidence=0.95)
        pref = await db.get_preference("editor", "theme")
        assert pref["value"] == "light"
        assert pref["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, db):
        pref = await db.get_preference("nope", "nope")
        assert pref is None

    @pytest.mark.asyncio
    async def test_project_scope(self, db):
        await db.set_preference("git", "branch", "main", project_scope="/proj/a")
        await db.set_preference("git", "branch", "develop", project_scope="/proj/b")
        pref_a = await db.get_preference("git", "branch", project_scope="/proj/a")
        pref_b = await db.get_preference("git", "branch", project_scope="/proj/b")
        assert pref_a["value"] == "main"
        assert pref_b["value"] == "develop"

    @pytest.mark.asyncio
    async def test_get_by_category(self, db):
        await db.set_preference("editor", "theme", "dark")
        await db.set_preference("editor", "font", "mono")
        await db.set_preference("git", "branch", "main")
        prefs = await db.get_preferences_by_category("editor")
        assert len(prefs) == 2


class TestSessions:
    @pytest.mark.asyncio
    async def test_save_and_get(self, db):
        await db.save_session(session_id="s1", project_dir="/projects/test", started_at="2025-01-01T00:00:00Z", goal="Fix bug", tools_used=["Read", "Edit"], files_touched=["src/main.py"])
        session = await db.get_session("s1")
        assert session is not None
        assert session["goal"] == "Fix bug"
        assert session["tools_used"] == ["Read", "Edit"]

    @pytest.mark.asyncio
    async def test_upsert_session(self, db):
        await db.save_session("s1", "/proj", "2025-01-01T00:00:00Z", goal="start")
        await db.save_session("s1", "/proj", "2025-01-01T00:00:00Z", goal="updated")
        session = await db.get_session("s1")
        assert session["goal"] == "updated"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, db):
        assert await db.get_session("nope") is None

    @pytest.mark.asyncio
    async def test_sessions_for_project(self, db):
        await db.save_session("s1", "/proj/a", "2025-01-01T00:00:00Z")
        await db.save_session("s2", "/proj/a", "2025-01-02T00:00:00Z")
        await db.save_session("s3", "/proj/b", "2025-01-03T00:00:00Z")
        sessions = await db.get_sessions_for_project("/proj/a")
        assert len(sessions) == 2


class TestApprovalPatterns:
    @pytest.mark.asyncio
    async def test_record_and_get(self, db):
        await db.record_approval("Bash", "git push", "deny")
        pattern = await db.get_approval_pattern("Bash", "git push")
        assert pattern is not None
        assert pattern["decision"] == "deny"
        assert pattern["count"] == 1

    @pytest.mark.asyncio
    async def test_count_increments(self, db):
        await db.record_approval("Read", "*", "approve")
        await db.record_approval("Read", "*", "approve")
        await db.record_approval("Read", "*", "approve")
        pattern = await db.get_approval_pattern("Read", "*")
        assert pattern["count"] == 3

    @pytest.mark.asyncio
    async def test_nonexistent_pattern(self, db):
        assert await db.get_approval_pattern("nope", "nope") is None


class TestPolicies:
    @pytest.mark.asyncio
    async def test_add_and_get(self, db):
        pid = await db.add_policy("security", "no-force-push", "Never force push to main")
        assert pid is not None
        policies = await db.get_policies("security")
        assert len(policies) == 1
        assert policies[0]["rule"] == "no-force-push"

    @pytest.mark.asyncio
    async def test_filter_by_type(self, db):
        await db.add_policy("security", "rule1")
        await db.add_policy("style", "rule2")
        security = await db.get_policies("security")
        assert len(security) == 1
        all_policies = await db.get_policies()
        assert len(all_policies) == 2


class TestEscalationLog:
    @pytest.mark.asyncio
    async def test_log_and_retrieve(self, db):
        await db.log_escalation("s1", "Deploy to prod?", "telegram")
        logs = await db.get_escalation_log("s1")
        assert len(logs) == 1
        assert logs[0]["question"] == "Deploy to prod?"

    @pytest.mark.asyncio
    async def test_multiple_entries(self, db):
        await db.log_escalation("s1", "Question 1", "telegram")
        await db.log_escalation("s1", "Question 2", "queue")
        logs = await db.get_escalation_log("s1")
        assert len(logs) == 2

    @pytest.mark.asyncio
    async def test_all_logs(self, db):
        await db.log_escalation("s1", "Q1", "telegram")
        await db.log_escalation("s2", "Q2", "queue")
        logs = await db.get_escalation_log()
        assert len(logs) == 2
