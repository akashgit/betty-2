"""Shared test fixtures for Betty 2.0."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from betty.config import (
    Config,
    DelegationConfig,
    EscalationConfig,
    LLMConfig,
    PolicyConfig,
)
from betty.db import UserModelDB


@pytest.fixture
def tmp_betty_dir(tmp_path):
    """Temporary ~/.betty/ directory for tests."""
    betty_dir = tmp_path / ".betty"
    betty_dir.mkdir()
    return betty_dir


@pytest.fixture
def mock_config():
    """Config with test-safe defaults."""
    return Config(
        llm=LLMConfig(model="test/mock-model", api_key="test-key"),
        delegation=DelegationConfig(autonomy_level=1),
        escalation=EscalationConfig(escalation_mode="queue"),
        policy=PolicyConfig(),
    )


@pytest_asyncio.fixture
async def test_db(tmp_path):
    """In-memory-like SQLite database for tests."""
    db_path = tmp_path / "test.db"
    db = UserModelDB(db_path)
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
def mock_llm():
    """Mock LLM service that returns predictable responses."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Mock LLM response")
    llm.complete_json = AsyncMock(return_value={"result": "mock", "confidence": 0.9})
    llm.embed = AsyncMock(return_value=[0.1] * 384)
    llm.get_usage = MagicMock(
        return_value={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "call_count": 1}
    )
    return llm


@pytest.fixture
def mock_session_jsonl(tmp_path):
    """Create a mock Claude Code JSONL transcript file."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    transcript = session_dir / "session_001.jsonl"
    turns = [
        {"type": "human", "message": {"content": "Fix the login bug"}, "timestamp": "2025-01-15T10:00:00Z"},
        {"type": "assistant", "message": {"content": "I'll look at the login code.", "tool_use": [{"name": "Read", "input": {"file_path": "src/auth/login.py"}}]}, "timestamp": "2025-01-15T10:00:05Z"},
        {"type": "human", "message": {"content": "Looks good, thanks!"}, "timestamp": "2025-01-15T10:00:30Z"},
    ]
    with open(transcript, "w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")
    return transcript
