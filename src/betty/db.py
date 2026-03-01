"""SQLite database for Betty's user model store.

Uses aiosqlite for async access. WAL mode for concurrent reads.
Auto-creates database and runs migrations on first connection.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from betty.config import DB_FILE

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 3

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT NOT NULL,
    project_scope TEXT NOT NULL DEFAULT '',
    UNIQUE(category, key, project_scope)
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    project_dir TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    goal TEXT,
    outcome TEXT,
    tools_used TEXT NOT NULL DEFAULT '[]',
    files_touched TEXT NOT NULL DEFAULT '[]',
    decisions_made TEXT NOT NULL DEFAULT '[]',
    patterns_observed TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS approval_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    action_pattern TEXT NOT NULL,
    decision TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT NOT NULL,
    project_scope TEXT NOT NULL DEFAULT '',
    UNIQUE(tool_name, action_pattern, project_scope)
);

CREATE TABLE IF NOT EXISTS org_policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_type TEXT NOT NULL,
    rule TEXT NOT NULL,
    description TEXT,
    source TEXT,
    project_scope TEXT
);

CREATE TABLE IF NOT EXISTS escalation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    response TEXT,
    channel TEXT NOT NULL,
    response_time_seconds REAL,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_prefs_category ON user_preferences(category);
CREATE INDEX IF NOT EXISTS idx_user_prefs_project ON user_preferences(project_scope);
CREATE INDEX IF NOT EXISTS idx_session_project ON session_summaries(project_dir);
CREATE INDEX IF NOT EXISTS idx_approval_tool ON approval_patterns(tool_name);
CREATE INDEX IF NOT EXISTS idx_escalation_session ON escalation_log(session_id);
"""

_GLOBAL_SCOPE = ""


def _scope(project_scope: str | None) -> str:
    return project_scope if project_scope is not None else _GLOBAL_SCOPE


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class UserModelDB:
    """Async SQLite database for Betty's user model."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_FILE
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._migrate()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    async def _migrate(self) -> None:
        try:
            async with self.db.execute("SELECT MAX(version) FROM schema_version") as cursor:
                row = await cursor.fetchone()
                current = row[0] if row and row[0] else 0
        except aiosqlite.OperationalError:
            current = 0

        if current < 1:
            await self.db.executescript(SCHEMA_V1)
            await self.db.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (1, _now_iso()),
            )
            await self.db.commit()
            logger.info("Applied schema v1")

        if current < 2:
            await self.db.execute(
                "ALTER TABLE approval_patterns ADD COLUMN auto_approve INTEGER NOT NULL DEFAULT 0"
            )
            await self.db.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (2, _now_iso()),
            )
            await self.db.commit()
            logger.info("Applied schema v2: added auto_approve column")

        if current < 3:
            # Fix confidence values that were stuck due to ON CONFLICT
            # resetting confidence to excluded.confidence instead of growing.
            # Recalculate: confidence = 1 - 0.7 * 0.85^(evidence_count - 1)
            async with self.db.execute(
                "SELECT id, evidence_count FROM user_preferences WHERE evidence_count > 1"
            ) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                ec = row["evidence_count"]
                new_conf = min(0.95, 1.0 - 0.7 * (0.85 ** (ec - 1)))
                await self.db.execute(
                    "UPDATE user_preferences SET confidence = ? WHERE id = ?",
                    (new_conf, row["id"]),
                )
            await self.db.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (3, _now_iso()),
            )
            await self.db.commit()
            logger.info("Applied schema v3: recalculated confidence from evidence_count (%d rows)", len(rows))

    async def set_preference(self, category: str, key: str, value: str, confidence: float = 0.5, project_scope: str | None = None) -> None:
        now = _now_iso()
        await self.db.execute(
            """INSERT INTO user_preferences (category, key, value, confidence, evidence_count, last_seen, project_scope)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(category, key, project_scope) DO UPDATE SET
                value = excluded.value,
                confidence = MIN(0.95, user_preferences.confidence + (1.0 - user_preferences.confidence) * 0.15),
                evidence_count = evidence_count + 1, last_seen = excluded.last_seen""",
            (category, key, value, confidence, now, _scope(project_scope)),
        )
        await self.db.commit()

    async def get_preference(self, category: str, key: str, project_scope: str | None = None) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT value, confidence, evidence_count, last_seen FROM user_preferences WHERE category = ? AND key = ? AND project_scope = ?",
            (category, key, _scope(project_scope)),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {"value": row["value"], "confidence": row["confidence"], "evidence_count": row["evidence_count"], "last_seen": row["last_seen"]}

    async def get_preferences_by_category(self, category: str, project_scope: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM user_preferences WHERE category = ?"
        params: list[Any] = [category]
        if project_scope is not None:
            query += " AND (project_scope = ? OR project_scope = ?)"
            params.extend([_GLOBAL_SCOPE, project_scope])
        else:
            query += " AND project_scope = ?"
            params.append(_GLOBAL_SCOPE)
        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def save_session(self, session_id: str, project_dir: str, started_at: str, ended_at: str | None = None, goal: str | None = None, outcome: str | None = None, tools_used: list[str] | None = None, files_touched: list[str] | None = None, decisions_made: list[dict] | None = None, patterns_observed: list[str] | None = None) -> None:
        await self.db.execute(
            """INSERT INTO session_summaries (session_id, project_dir, started_at, ended_at, goal, outcome, tools_used, files_touched, decisions_made, patterns_observed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET ended_at = excluded.ended_at, goal = excluded.goal, outcome = excluded.outcome,
                tools_used = excluded.tools_used, files_touched = excluded.files_touched, decisions_made = excluded.decisions_made, patterns_observed = excluded.patterns_observed""",
            (session_id, project_dir, started_at, ended_at, goal, outcome, json.dumps(tools_used or []), json.dumps(files_touched or []), json.dumps(decisions_made or []), json.dumps(patterns_observed or [])),
        )
        await self.db.commit()

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async with self.db.execute("SELECT * FROM session_summaries WHERE session_id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            for field in ("tools_used", "files_touched", "decisions_made", "patterns_observed"):
                result[field] = json.loads(result[field])
            return result

    async def get_sessions_for_project(self, project_dir: str, limit: int = 50) -> list[dict[str, Any]]:
        async with self.db.execute("SELECT * FROM session_summaries WHERE project_dir = ? ORDER BY started_at DESC LIMIT ?", (project_dir, limit)) as cursor:
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                r = dict(row)
                for field in ("tools_used", "files_touched", "decisions_made", "patterns_observed"):
                    r[field] = json.loads(r[field])
                results.append(r)
            return results

    async def record_approval(self, tool_name: str, action_pattern: str, decision: str, project_scope: str | None = None) -> None:
        now = _now_iso()
        await self.db.execute(
            """INSERT INTO approval_patterns (tool_name, action_pattern, decision, count, last_seen, project_scope)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(tool_name, action_pattern, project_scope) DO UPDATE SET
                decision = excluded.decision, count = count + 1, last_seen = excluded.last_seen""",
            (tool_name, action_pattern, decision, now, _scope(project_scope)),
        )
        await self.db.commit()

    async def get_approval_pattern(self, tool_name: str, action_pattern: str, project_scope: str | None = None) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT decision, count, last_seen FROM approval_patterns WHERE tool_name = ? AND action_pattern = ? AND project_scope = ?",
            (tool_name, action_pattern, _scope(project_scope)),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return dict(row)

    async def add_policy(self, policy_type: str, rule: str, description: str | None = None, source: str | None = None, project_scope: str | None = None) -> int:
        cursor = await self.db.execute(
            "INSERT INTO org_policies (policy_type, rule, description, source, project_scope) VALUES (?, ?, ?, ?, ?)",
            (policy_type, rule, description, source, project_scope),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_policies(self, policy_type: str | None = None, project_scope: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM org_policies WHERE 1=1"
        params: list[Any] = []
        if policy_type:
            query += " AND policy_type = ?"
            params.append(policy_type)
        if project_scope is not None:
            query += " AND (project_scope IS NULL OR project_scope = ?)"
            params.append(project_scope)
        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def log_escalation(self, session_id: str, question: str, channel: str, response: str | None = None, response_time_seconds: float | None = None) -> None:
        await self.db.execute(
            "INSERT INTO escalation_log (session_id, question, response, channel, response_time_seconds, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, question, response, channel, response_time_seconds, _now_iso()),
        )
        await self.db.commit()

    async def get_escalation_log(self, session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if session_id:
            query = "SELECT * FROM escalation_log WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?"
            params: list[Any] = [session_id, limit]
        else:
            query = "SELECT * FROM escalation_log ORDER BY timestamp DESC LIMIT ?"
            params = [limit]
        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
