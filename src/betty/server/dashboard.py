"""Dashboard API + HTMX endpoints for the Betty web UI."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .models import (
    ApprovalPattern,
    ConfigResponse,
    ConfigUpdate,
    EscalationRecord,
    PolicyCreate,
    PolicyItem,
    PreferenceItem,
    SessionDetail,
    SessionSummary,
    UserProfile,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"])


def _templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates


def _render(request: Request, name: str, ctx: dict[str, Any] | None = None) -> HTMLResponse:
    return _templates(request).TemplateResponse(request, name, ctx or {})


# -- HTML pages ---------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def page_home(request: Request) -> HTMLResponse:
    profile = await _get_profile_data()
    sessions = await _get_sessions_data()
    return _render(request, "home.html", {"profile": profile, "sessions": sessions[:5]})

@router.get("/user-model", response_class=HTMLResponse)
async def page_user_model(request: Request) -> HTMLResponse:
    return _render(request, "user_model.html", {"profile": await _get_profile_data()})

@router.get("/sessions", response_class=HTMLResponse)
async def page_sessions(request: Request) -> HTMLResponse:
    return _render(request, "sessions.html", {"sessions": await _get_sessions_data()})

@router.get("/sessions/{session_id}", response_class=HTMLResponse)
async def page_session_detail(request: Request, session_id: str) -> HTMLResponse:
    return _render(request, "session_detail.html", {"session": await _get_session_detail(session_id)})

@router.get("/policies", response_class=HTMLResponse)
async def page_policies(request: Request) -> HTMLResponse:
    return _render(request, "policies.html", {"policies": await _get_policies_data()})

@router.get("/approvals", response_class=HTMLResponse)
async def page_approvals(request: Request) -> HTMLResponse:
    return _render(request, "approvals.html", {"patterns": await _get_approvals_data()})

@router.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request) -> HTMLResponse:
    return _render(request, "settings.html", {"config": await _get_config_data()})


# -- JSON API -----------------------------------------------------------------

@router.get("/api/profile", response_model=UserProfile)
async def api_profile() -> UserProfile:
    return await _get_profile_data()

@router.get("/api/sessions", response_model=list[SessionSummary])
async def api_sessions() -> list[SessionSummary]:
    return await _get_sessions_data()

@router.get("/api/sessions/{session_id}", response_model=SessionDetail)
async def api_session_detail(session_id: str) -> SessionDetail:
    return await _get_session_detail(session_id)

@router.get("/api/policies", response_model=list[PolicyItem])
async def api_policies() -> list[PolicyItem]:
    return await _get_policies_data()

@router.post("/api/policies", response_model=PolicyItem)
async def api_create_policy(policy: PolicyCreate) -> PolicyItem:
    logger.info("create policy: %s", policy.policy_type)
    return PolicyItem(id=1, policy_type=policy.policy_type, rule=policy.rule,
                      source=policy.source, project_scope=policy.project_scope)

@router.get("/api/approvals", response_model=list[ApprovalPattern])
async def api_approvals() -> list[ApprovalPattern]:
    return await _get_approvals_data()

@router.get("/api/preferences", response_model=list[PreferenceItem])
async def api_preferences() -> list[PreferenceItem]:
    return (await _get_profile_data()).preferences

@router.put("/api/preferences/{pref_id}", response_model=PreferenceItem)
async def api_update_preference(pref_id: int, update: dict[str, Any]) -> PreferenceItem:
    logger.info("update preference %d: %s", pref_id, update)
    return PreferenceItem(id=pref_id, category=update.get("category", "general"),
                          key=update.get("key", ""), value=update.get("value", ""),
                          confidence=update.get("confidence", 0.5))

@router.get("/api/escalations", response_model=list[EscalationRecord])
async def api_escalations() -> list[EscalationRecord]:
    return []

@router.get("/api/config", response_model=ConfigResponse)
async def api_get_config() -> ConfigResponse:
    return await _get_config_data()

@router.put("/api/config", response_model=ConfigResponse)
async def api_update_config(update: ConfigUpdate) -> ConfigResponse:
    logger.info("update config: %s", update.model_dump(exclude_none=True))
    return await _get_config_data()


# -- HTMX partials ------------------------------------------------------------

@router.get("/partials/preferences", response_class=HTMLResponse)
async def partial_preferences(request: Request) -> HTMLResponse:
    return _render(request, "partials/preferences_table.html",
                   {"preferences": (await _get_profile_data()).preferences})

@router.get("/partials/sessions", response_class=HTMLResponse)
async def partial_sessions(request: Request) -> HTMLResponse:
    return _render(request, "partials/sessions_list.html",
                   {"sessions": await _get_sessions_data()})

@router.get("/partials/policies", response_class=HTMLResponse)
async def partial_policies(request: Request) -> HTMLResponse:
    return _render(request, "partials/policies_list.html",
                   {"policies": await _get_policies_data()})

@router.get("/partials/approvals", response_class=HTMLResponse)
async def partial_approvals(request: Request) -> HTMLResponse:
    return _render(request, "partials/approvals_list.html",
                   {"patterns": await _get_approvals_data()})

@router.delete("/api/preferences/{pref_id}", response_class=HTMLResponse)
async def api_delete_preference(request: Request, pref_id: int) -> HTMLResponse:
    logger.info("delete preference %d", pref_id)
    remaining = [p for p in (await _get_profile_data()).preferences if p.id != pref_id]
    return _render(request, "partials/preferences_table.html", {"preferences": remaining})

@router.delete("/api/policies/{policy_id}", response_class=HTMLResponse)
async def api_delete_policy(request: Request, policy_id: int) -> HTMLResponse:
    logger.info("delete policy %d", policy_id)
    remaining = [p for p in await _get_policies_data() if p.id != policy_id]
    return _render(request, "partials/policies_list.html", {"policies": remaining})

@router.put("/api/approvals/{pattern_id}/toggle", response_class=HTMLResponse)
async def api_toggle_approval(request: Request, pattern_id: int) -> HTMLResponse:
    logger.info("toggle approval %d", pattern_id)
    patterns = await _get_approvals_data()
    for p in patterns:
        if p.id == pattern_id:
            p.auto_approve = not p.auto_approve
    return _render(request, "partials/approvals_list.html", {"patterns": patterns})


# -- Data access stubs (will be replaced with DB queries) ----------------------

async def _get_profile_data() -> UserProfile:
    return UserProfile(total_preferences=0, total_sessions=0, total_approvals=0, preferences=[])

async def _get_sessions_data() -> list[SessionSummary]:
    return []

async def _get_session_detail(session_id: str) -> SessionDetail:
    return SessionDetail(session_id=session_id)

async def _get_policies_data() -> list[PolicyItem]:
    return []

async def _get_approvals_data() -> list[ApprovalPattern]:
    return []

async def _get_config_data() -> ConfigResponse:
    return ConfigResponse()
