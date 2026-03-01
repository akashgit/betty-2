"""Tests for the Betty FastAPI server."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from betty.db import UserModelDB
from betty.server.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def app_with_db(tmp_path):
    """App with a real test database wired in."""
    db_path = tmp_path / "test.db"
    db = UserModelDB(db_path)
    await db.connect()

    app = create_app()
    app.state.db = db

    yield app

    await db.close()


@pytest.fixture
async def client_with_db(app_with_db):
    transport = ASGITransport(app=app_with_db)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# -- Health -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime_seconds" in data


# -- Hook endpoints -----------------------------------------------------------

@pytest.mark.asyncio
async def test_hook_prompt_submit(client):
    resp = await client.post("/hooks/UserPromptSubmit", json={
        "session_id": "test-session", "prompt": "Fix the login bug"})
    assert resp.status_code == 200
    assert resp.json()["proceed"] is True

@pytest.mark.asyncio
async def test_hook_pre_tool_use(client):
    resp = await client.post("/hooks/PreToolUse", json={
        "session_id": "test-session", "tool_name": "Read",
        "tool_input": {"file_path": "/tmp/test.py"}})
    assert resp.status_code == 200
    assert resp.json()["decision"] == "allow"

@pytest.mark.asyncio
async def test_hook_post_tool_use(client):
    resp = await client.post("/hooks/PostToolUse", json={
        "session_id": "test-session", "tool_name": "Read",
        "tool_input": {"file_path": "/tmp/test.py"}, "tool_output": "contents"})
    assert resp.status_code == 200
    assert resp.json()["acknowledged"] is True

@pytest.mark.asyncio
async def test_hook_pre_tool_use_with_approval_model(app, client):
    """PreToolUse should consult the ApprovalModel when available."""
    from betty.approval import ApprovalModel
    app.state.approval_model = ApprovalModel(autonomy_level=1)

    resp = await client.post("/hooks/PreToolUse", json={
        "session_id": "s1", "tool_name": "Read",
        "tool_input": {"file_path": "/tmp/test.py"}})
    assert resp.status_code == 200
    assert resp.json()["decision"] == "allow"

@pytest.mark.asyncio
async def test_hook_pre_tool_use_destructive(app, client):
    """Destructive Bash commands should get 'ask' decision."""
    from betty.approval import ApprovalModel
    app.state.approval_model = ApprovalModel(autonomy_level=2)

    resp = await client.post("/hooks/PreToolUse", json={
        "session_id": "s1", "tool_name": "Bash",
        "tool_input": {"command": "rm -rf /"}})
    assert resp.status_code == 200
    assert resp.json()["decision"] == "ask"

@pytest.mark.asyncio
async def test_hook_post_tool_use_records_approval(app, client):
    """PostToolUse should record the tool use in the ApprovalModel."""
    from betty.approval import ApprovalModel
    model = ApprovalModel(autonomy_level=1)
    app.state.approval_model = model

    resp = await client.post("/hooks/PostToolUse", json={
        "session_id": "s1", "tool_name": "Write",
        "tool_input": {"file_path": "/tmp/out.py"}, "tool_output": ""})
    assert resp.status_code == 200

    prediction = model.predict("Write", {"file_path": "/tmp/out.py"})
    assert prediction.pattern_count >= 1


# -- Dashboard JSON API -------------------------------------------------------

@pytest.mark.asyncio
async def test_api_profile(client):
    resp = await client.get("/api/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_preferences" in data
    assert "preferences" in data

@pytest.mark.asyncio
async def test_api_sessions(client):
    assert (await client.get("/api/sessions")).status_code == 200

@pytest.mark.asyncio
async def test_api_session_detail(client):
    resp = await client.get("/api/sessions/test-123")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "test-123"

@pytest.mark.asyncio
async def test_api_policies(client):
    assert (await client.get("/api/policies")).status_code == 200

@pytest.mark.asyncio
async def test_api_create_policy(client):
    resp = await client.post("/api/policies", json={
        "policy_type": "testing", "rule": "Always run tests",
        "source": "team-guidelines.md"})
    assert resp.status_code == 200
    assert resp.json()["policy_type"] == "testing"

@pytest.mark.asyncio
async def test_api_approvals(client):
    assert (await client.get("/api/approvals")).status_code == 200

@pytest.mark.asyncio
async def test_api_preferences(client):
    assert (await client.get("/api/preferences")).status_code == 200

@pytest.mark.asyncio
async def test_api_escalations(client):
    assert (await client.get("/api/escalations")).status_code == 200

@pytest.mark.asyncio
async def test_api_config(client):
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    assert "port" in resp.json()

@pytest.mark.asyncio
async def test_api_update_config(client):
    assert (await client.put("/api/config", json={"llm_model": "openai/gpt-4o-mini"})).status_code == 200


# -- Dashboard HTML pages -----------------------------------------------------

@pytest.mark.asyncio
async def test_page_home(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text

@pytest.mark.asyncio
async def test_page_user_model(client):
    resp = await client.get("/user-model")
    assert resp.status_code == 200
    assert "User Model" in resp.text

@pytest.mark.asyncio
async def test_page_sessions(client):
    assert (await client.get("/sessions")).status_code == 200

@pytest.mark.asyncio
async def test_page_session_detail(client):
    resp = await client.get("/sessions/test-123")
    assert resp.status_code == 200
    assert "test-123" in resp.text

@pytest.mark.asyncio
async def test_page_policies(client):
    assert (await client.get("/policies")).status_code == 200

@pytest.mark.asyncio
async def test_page_approvals(client):
    assert (await client.get("/approvals")).status_code == 200

@pytest.mark.asyncio
async def test_page_settings(client):
    assert (await client.get("/settings")).status_code == 200


# -- HTMX partials ------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_preferences(client):
    resp = await client.get("/partials/preferences")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]

@pytest.mark.asyncio
async def test_partial_sessions(client):
    assert (await client.get("/partials/sessions")).status_code == 200

@pytest.mark.asyncio
async def test_partial_policies(client):
    assert (await client.get("/partials/policies")).status_code == 200

@pytest.mark.asyncio
async def test_partial_approvals(client):
    assert (await client.get("/partials/approvals")).status_code == 200


# -- Wired UserPromptSubmit (IntentEngine in heuristic mode) -------------------

@pytest.mark.asyncio
async def test_hook_prompt_submit_with_intent_engine(app, client):
    """UserPromptSubmit should use IntentEngine when available."""
    from betty.intent_engine import IntentEngine

    engine = IntentEngine()  # No LLM, heuristic mode
    app.state.intent_engine = engine

    resp = await client.post("/hooks/UserPromptSubmit", json={
        "session_id": "test-session", "prompt": "add user authentication"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["proceed"] is True
    # Heuristic mode should ask about tests for "add" prompts.
    assert any("test" in q.lower() for q in data["questions"])
    assert "confidence" in data
    assert "similar_sessions" in data
    assert "applicable_policies" in data


@pytest.mark.asyncio
async def test_hook_prompt_submit_response_schema(client):
    """Verify new fields exist in the response."""
    resp = await client.post("/hooks/UserPromptSubmit", json={
        "session_id": "s1", "prompt": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "suggested_context" in data
    assert "similar_sessions" in data
    assert "applicable_policies" in data
    assert "predicted_plan" in data
    assert "confidence" in data


# -- Dashboard API with real DB -----------------------------------------------

@pytest.mark.asyncio
async def test_api_profile_with_db(app_with_db, client_with_db):
    """Profile API returns real counts from the database."""
    db = app_with_db.state.db

    # Add a preference.
    await db.set_preference("coding_style", "test_framework", "pytest", 0.8)
    await db.save_session("s1", "/tmp/project", "2025-01-01T00:00:00Z", goal="fix bug")

    resp = await client_with_db.get("/api/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_preferences"] >= 1
    assert data["total_sessions"] >= 1
    assert len(data["preferences"]) >= 1


@pytest.mark.asyncio
async def test_api_sessions_with_db(app_with_db, client_with_db):
    """Sessions API returns real data from the database."""
    db = app_with_db.state.db
    await db.save_session("sess-1", "/tmp/proj", "2025-01-01T00:00:00Z",
                          goal="add auth", tools_used=["Read", "Edit"])

    resp = await client_with_db.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["session_id"] == "sess-1"
    assert data[0]["goal"] == "add auth"


@pytest.mark.asyncio
async def test_api_session_detail_with_db(app_with_db, client_with_db):
    """Session detail API returns full session data."""
    db = app_with_db.state.db
    await db.save_session("detail-1", "/tmp/proj", "2025-01-01T00:00:00Z",
                          goal="refactor", files_touched=["src/main.py"])

    resp = await client_with_db.get("/api/sessions/detail-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "detail-1"
    assert data["goal"] == "refactor"
    assert "src/main.py" in data["files_touched"]


@pytest.mark.asyncio
async def test_api_policies_with_db(app_with_db, client_with_db):
    """Policies API returns real data."""
    db = app_with_db.state.db
    await db.add_policy("testing", "Always run tests", source="team-guide")

    resp = await client_with_db.get("/api/policies")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["rule"] == "Always run tests"


@pytest.mark.asyncio
async def test_api_create_policy_with_db(app_with_db, client_with_db):
    """Creating a policy persists to the database."""
    resp = await client_with_db.post("/api/policies", json={
        "policy_type": "security", "rule": "No hardcoded secrets",
        "source": "security-checklist"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["policy_type"] == "security"
    assert data["id"] is not None

    # Verify it was persisted.
    resp2 = await client_with_db.get("/api/policies")
    assert any(p["rule"] == "No hardcoded secrets" for p in resp2.json())


@pytest.mark.asyncio
async def test_api_config_returns_real_values(client_with_db):
    """Config API returns actual config values, not empty defaults."""
    resp = await client_with_db.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    # Should have the default model, not empty string.
    assert data["llm_model"] != ""
    assert data["port"] == 7832


@pytest.mark.asyncio
async def test_api_escalations_with_db(app_with_db, client_with_db):
    """Escalations API returns real data."""
    db = app_with_db.state.db
    await db.log_escalation("s1", "Should I proceed?", "queue")

    resp = await client_with_db.get("/api/escalations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["reason"] == "Should I proceed?"


# -- Broad approval rules API ------------------------------------------------

@pytest.mark.asyncio
async def test_api_create_approval_rule(client_with_db):
    """Create a broad approval rule via API."""
    resp = await client_with_db.post("/api/approvals/rules", json={
        "tool_name": "Edit", "action_pattern": "*/*.py", "decision": "accepted"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["tool_name"] == "Edit"
    assert data["action_pattern"] == "*/*.py"
    assert data["auto_approve"] is True
    assert data["count"] == 0


@pytest.mark.asyncio
async def test_api_create_approval_rule_persists(app_with_db, client_with_db):
    """Created rule should appear in the approvals list."""
    await client_with_db.post("/api/approvals/rules", json={
        "tool_name": "Bash", "action_pattern": "bash:git-*", "decision": "accepted"})

    resp = await client_with_db.get("/api/approvals")
    assert resp.status_code == 200
    patterns = resp.json()
    assert any(p["action_pattern"] == "bash:git-*" for p in patterns)


@pytest.mark.asyncio
async def test_api_delete_approval(app_with_db, client_with_db):
    """Delete an approval pattern via API."""
    db = app_with_db.state.db
    rule_id = await db.create_approval_rule("Edit", "*", "accepted")

    resp = await client_with_db.delete(f"/api/approvals/{rule_id}")
    assert resp.status_code == 200

    # Verify deleted
    resp = await client_with_db.get("/api/approvals")
    patterns = resp.json()
    assert not any(p["action_pattern"] == "*" and p["tool_name"] == "Edit" for p in patterns)


@pytest.mark.asyncio
async def test_api_approval_suggestions(app_with_db, client_with_db):
    """Suggestions API works with an ApprovalModel."""
    from betty.approval import ApprovalModel
    model = ApprovalModel(autonomy_level=2)
    app_with_db.state.approval_model = model

    # Record enough similar patterns to trigger suggestion
    model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")
    model.record("Edit", {"file_path": "/proj/tests/test_main.py"}, "accepted")
    model.record("Edit", {"file_path": "/proj/lib/utils.py"}, "accepted")

    resp = await client_with_db.get("/api/approvals/suggestions")
    assert resp.status_code == 200
    data = resp.json()
    assert any(s["suggested_pattern"] == "*/*.py" for s in data)


@pytest.mark.asyncio
async def test_api_approval_suggestions_empty(client):
    """Suggestions API returns empty when no model."""
    resp = await client.get("/api/approvals/suggestions")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_partial_approval_suggestions(client):
    """Partial for approval suggestions renders."""
    resp = await client.get("/partials/approval-suggestions")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_api_create_rule_loads_into_model(app_with_db, client_with_db):
    """Creating a rule via API should also update the in-memory ApprovalModel."""
    from betty.approval import ApprovalModel, ApprovalDecision
    model = ApprovalModel(autonomy_level=2)
    app_with_db.state.approval_model = model

    await client_with_db.post("/api/approvals/rules", json={
        "tool_name": "Bash", "action_pattern": "bash:git-*", "decision": "accepted"})

    # The in-memory model should now approve git commands
    result = model.predict("Bash", {"command": "git status"})
    assert result.decision == ApprovalDecision.APPROVE
