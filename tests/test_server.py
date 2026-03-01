"""Tests for the Betty FastAPI server."""

import pytest
from httpx import ASGITransport, AsyncClient

from betty.server.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
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
    resp = await client.post("/hooks/prompt-submit", json={
        "session_id": "test-session", "prompt": "Fix the login bug"})
    assert resp.status_code == 200
    assert resp.json()["proceed"] is True

@pytest.mark.asyncio
async def test_hook_pre_tool_use(client):
    resp = await client.post("/hooks/pre-tool-use", json={
        "session_id": "test-session", "tool_name": "Read",
        "tool_input": {"file_path": "/tmp/test.py"}})
    assert resp.status_code == 200
    assert resp.json()["decision"] == "allow"

@pytest.mark.asyncio
async def test_hook_post_tool_use(client):
    resp = await client.post("/hooks/post-tool-use", json={
        "session_id": "test-session", "tool_name": "Read",
        "tool_input": {"file_path": "/tmp/test.py"}, "tool_output": "contents"})
    assert resp.status_code == 200
    assert resp.json()["acknowledged"] is True


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
