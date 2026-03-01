"""Dashboard API + HTMX endpoints for the Betty web UI."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .models import (
    ApprovalPattern,
    ApprovalRuleCreate,
    ApprovalRuleSuggestion,
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


def _db(request: Request):
    """Get the database from app state, or None."""
    return getattr(request.app.state, "db", None)


def _templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates


def _render(request: Request, name: str, ctx: dict[str, Any] | None = None) -> HTMLResponse:
    return _templates(request).TemplateResponse(request, name, ctx or {})


# -- HTML pages ---------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def page_home(request: Request) -> HTMLResponse:
    profile = await _get_profile_data(request)
    sessions = await _get_sessions_data(request)
    return _render(request, "home.html", {"profile": profile, "sessions": sessions[:5]})

@router.get("/user-model", response_class=HTMLResponse)
async def page_user_model(request: Request) -> HTMLResponse:
    return _render(request, "user_model.html", {"profile": await _get_profile_data(request)})

@router.get("/sessions", response_class=HTMLResponse)
async def page_sessions(request: Request) -> HTMLResponse:
    return _render(request, "sessions.html", {"sessions": await _get_sessions_data(request)})

@router.get("/sessions/{session_id}", response_class=HTMLResponse)
async def page_session_detail(request: Request, session_id: str) -> HTMLResponse:
    return _render(request, "session_detail.html", {"session": await _get_session_detail(request, session_id)})

@router.get("/policies", response_class=HTMLResponse)
async def page_policies(request: Request) -> HTMLResponse:
    return _render(request, "policies.html", {"policies": await _get_policies_data(request)})

@router.get("/approvals", response_class=HTMLResponse)
async def page_approvals(request: Request) -> HTMLResponse:
    return _render(request, "approvals.html", {"patterns": await _get_approvals_data(request)})

@router.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request) -> HTMLResponse:
    return _render(request, "settings.html", {"config": await _get_config_data(request)})


# -- JSON API -----------------------------------------------------------------

@router.get("/api/profile", response_model=UserProfile)
async def api_profile(request: Request) -> UserProfile:
    return await _get_profile_data(request)

@router.get("/api/sessions", response_model=list[SessionSummary])
async def api_sessions(request: Request) -> list[SessionSummary]:
    return await _get_sessions_data(request)

@router.get("/api/sessions/{session_id}", response_model=SessionDetail)
async def api_session_detail(request: Request, session_id: str) -> SessionDetail:
    return await _get_session_detail(request, session_id)

@router.get("/api/policies", response_model=list[PolicyItem])
async def api_policies(request: Request) -> list[PolicyItem]:
    return await _get_policies_data(request)

@router.post("/api/policies", response_model=PolicyItem)
async def api_create_policy(request: Request, policy: PolicyCreate) -> PolicyItem:
    logger.info("create policy: %s", policy.policy_type)
    db = _db(request)
    if db is not None:
        try:
            policy_id = await db.add_policy(
                policy_type=policy.policy_type,
                rule=policy.rule,
                source=policy.source,
                project_scope=policy.project_scope,
            )
            return PolicyItem(
                id=policy_id, policy_type=policy.policy_type, rule=policy.rule,
                source=policy.source, project_scope=policy.project_scope,
            )
        except Exception:
            logger.exception("Failed to create policy")
    return PolicyItem(id=1, policy_type=policy.policy_type, rule=policy.rule,
                      source=policy.source, project_scope=policy.project_scope)

@router.get("/api/approvals", response_model=list[ApprovalPattern])
async def api_approvals(request: Request) -> list[ApprovalPattern]:
    return await _get_approvals_data(request)

@router.get("/api/preferences", response_model=list[PreferenceItem])
async def api_preferences(request: Request) -> list[PreferenceItem]:
    return (await _get_profile_data(request)).preferences

@router.put("/api/preferences/{pref_id}", response_model=PreferenceItem)
async def api_update_preference(request: Request, pref_id: int, update: dict[str, Any]) -> PreferenceItem:
    logger.info("update preference %d: %s", pref_id, update)
    db = _db(request)
    if db is not None:
        try:
            await db.set_preference(
                category=update.get("category", "general"),
                key=update.get("key", ""),
                value=update.get("value", ""),
                confidence=update.get("confidence", 0.5),
                project_scope=update.get("project_scope"),
            )
        except Exception:
            logger.exception("Failed to update preference %d", pref_id)
    return PreferenceItem(id=pref_id, category=update.get("category", "general"),
                          key=update.get("key", ""), value=update.get("value", ""),
                          confidence=update.get("confidence", 0.5))

@router.get("/api/escalations", response_model=list[EscalationRecord])
async def api_escalations(request: Request) -> list[EscalationRecord]:
    db = _db(request)
    if db is not None:
        try:
            rows = await db.get_escalation_log(limit=50)
            return [
                EscalationRecord(
                    id=row.get("id", 0),
                    session_id=row.get("session_id", ""),
                    reason=row.get("question", ""),
                    channel=row.get("channel", ""),
                    resolved=row.get("response") is not None,
                    created_at=row.get("timestamp"),
                )
                for row in rows
            ]
        except Exception:
            logger.exception("Failed to fetch escalation log")
    return []

@router.get("/api/config", response_model=ConfigResponse)
async def api_get_config(request: Request) -> ConfigResponse:
    return await _get_config_data(request)

@router.put("/api/config", response_model=ConfigResponse)
async def api_update_config(request: Request, update: ConfigUpdate) -> ConfigResponse:
    logger.info("update config: %s", update.model_dump(exclude_none=True))
    try:
        from betty.config import load_config, save_config

        cfg = load_config()
        if update.llm_model is not None:
            cfg.llm.model = update.llm_model
        if update.llm_api_base is not None:
            cfg.llm.api_base = update.llm_api_base
        if update.delegation_level is not None:
            cfg.delegation.autonomy_level = update.delegation_level
        if update.auto_approve_read_tools is not None:
            cfg.delegation.auto_approve_read_tools = update.auto_approve_read_tools
        if update.confidence_threshold is not None:
            cfg.delegation.confidence_threshold = update.confidence_threshold
        save_config(cfg)

        # Apply to running daemon's in-memory state.
        approval_model = getattr(request.app.state, "approval_model", None)
        if approval_model is not None:
            if update.delegation_level is not None:
                approval_model.autonomy_level = update.delegation_level
            if update.confidence_threshold is not None:
                approval_model.confidence_threshold = update.confidence_threshold
    except Exception:
        logger.exception("Failed to save config")
    return await _get_config_data(request)


# -- HTMX partials ------------------------------------------------------------

@router.get("/partials/home-stats", response_class=HTMLResponse)
async def partial_home_stats(request: Request) -> HTMLResponse:
    profile = await _get_profile_data(request)
    return _render(request, "partials/home_stats.html", {"profile": profile})

@router.get("/partials/home-prefs", response_class=HTMLResponse)
async def partial_home_prefs(request: Request) -> HTMLResponse:
    profile = await _get_profile_data(request)
    return _render(request, "partials/home_prefs.html", {"preferences": profile.preferences[:5]})

@router.get("/partials/home-sessions", response_class=HTMLResponse)
async def partial_home_sessions(request: Request) -> HTMLResponse:
    sessions = await _get_sessions_data(request)
    return _render(request, "partials/home_sessions.html", {"sessions": sessions[:5]})

@router.get("/partials/preferences", response_class=HTMLResponse)
async def partial_preferences(request: Request) -> HTMLResponse:
    return _render(request, "partials/preferences_table.html",
                   {"preferences": (await _get_profile_data(request)).preferences})

@router.get("/partials/sessions", response_class=HTMLResponse)
async def partial_sessions(request: Request) -> HTMLResponse:
    return _render(request, "partials/sessions_list.html",
                   {"sessions": await _get_sessions_data(request)})

@router.get("/partials/policies", response_class=HTMLResponse)
async def partial_policies(request: Request) -> HTMLResponse:
    return _render(request, "partials/policies_list.html",
                   {"policies": await _get_policies_data(request)})

@router.get("/partials/approvals", response_class=HTMLResponse)
async def partial_approvals(request: Request) -> HTMLResponse:
    return _render(request, "partials/approvals_list.html",
                   {"patterns": await _get_approvals_data(request)})

@router.delete("/api/preferences/{pref_id}", response_class=HTMLResponse)
async def api_delete_preference(request: Request, pref_id: int) -> HTMLResponse:
    logger.info("delete preference %d", pref_id)
    db = _db(request)
    if db is not None:
        try:
            await db.db.execute("DELETE FROM user_preferences WHERE id = ?", (pref_id,))
            await db.db.commit()
        except Exception:
            logger.exception("Failed to delete preference %d", pref_id)
    remaining = [p for p in (await _get_profile_data(request)).preferences if p.id != pref_id]
    return _render(request, "partials/preferences_table.html", {"preferences": remaining})

@router.delete("/api/policies/{policy_id}", response_class=HTMLResponse)
async def api_delete_policy(request: Request, policy_id: int) -> HTMLResponse:
    logger.info("delete policy %d", policy_id)
    db = _db(request)
    if db is not None:
        try:
            await db.db.execute("DELETE FROM org_policies WHERE id = ?", (policy_id,))
            await db.db.commit()
        except Exception:
            logger.exception("Failed to delete policy %d", policy_id)
    remaining = [p for p in await _get_policies_data(request) if p.id != policy_id]
    return _render(request, "partials/policies_list.html", {"policies": remaining})

@router.put("/api/approvals/{pattern_id}/toggle", response_class=HTMLResponse)
async def api_toggle_approval(request: Request, pattern_id: int) -> HTMLResponse:
    logger.info("toggle approval %d", pattern_id)
    db = _db(request)
    if db is not None:
        try:
            await db.db.execute(
                "UPDATE approval_patterns SET auto_approve = CASE WHEN auto_approve = 0 THEN 1 ELSE 0 END WHERE id = ?",
                (pattern_id,),
            )
            await db.db.commit()
        except Exception:
            logger.exception("Failed to toggle approval %d", pattern_id)
    patterns = await _get_approvals_data(request)
    return _render(request, "partials/approvals_list.html", {"patterns": patterns})


@router.post("/api/approvals/rules", response_model=ApprovalPattern)
async def api_create_approval_rule(request: Request, rule: ApprovalRuleCreate) -> ApprovalPattern:
    logger.info("create approval rule: %s %s", rule.tool_name, rule.action_pattern)
    db = _db(request)
    pattern_id = None
    if db is not None:
        try:
            pattern_id = await db.create_approval_rule(
                tool_name=rule.tool_name,
                action_pattern=rule.action_pattern,
                decision=rule.decision,
                project_scope=rule.project_scope,
            )
        except Exception:
            logger.exception("Failed to create approval rule")

    # Also load into in-memory ApprovalModel if available
    approval_model = getattr(request.app.state, "approval_model", None)
    if approval_model is not None:
        approval_model.add_rule(
            tool_name=rule.tool_name,
            action_pattern=rule.action_pattern,
            decision=rule.decision,
            project_scope=rule.project_scope,
        )

    return ApprovalPattern(
        id=pattern_id or 0,
        tool_name=rule.tool_name,
        action_pattern=rule.action_pattern,
        decision=rule.decision,
        count=0,
        auto_approve=True,
        project_scope=rule.project_scope,
    )


@router.delete("/api/approvals/{pattern_id}", response_class=HTMLResponse)
async def api_delete_approval(request: Request, pattern_id: int) -> HTMLResponse:
    logger.info("delete approval %d", pattern_id)
    db = _db(request)
    if db is not None:
        try:
            await db.delete_approval_pattern(pattern_id)
        except Exception:
            logger.exception("Failed to delete approval %d", pattern_id)
    patterns = await _get_approvals_data(request)
    return _render(request, "partials/approvals_list.html", {"patterns": patterns})


@router.get("/api/approvals/suggestions", response_model=list[ApprovalRuleSuggestion])
async def api_approval_suggestions(request: Request) -> list[ApprovalRuleSuggestion]:
    approval_model = getattr(request.app.state, "approval_model", None)
    if approval_model is None:
        return []
    suggestions = approval_model.suggest_broader_rules()
    return [ApprovalRuleSuggestion(**s) for s in suggestions]


@router.get("/partials/approval-suggestions", response_class=HTMLResponse)
async def partial_approval_suggestions(request: Request) -> HTMLResponse:
    approval_model = getattr(request.app.state, "approval_model", None)
    suggestions = []
    if approval_model is not None:
        suggestions = approval_model.suggest_broader_rules()
    return _render(request, "partials/approval_suggestions.html", {"suggestions": suggestions})


# -- Data access (backed by real DB queries when available) --------------------

async def _get_profile_data(request: Request) -> UserProfile:
    db = _db(request)
    if db is None:
        return UserProfile()

    try:
        # Count preferences.
        async with db.db.execute("SELECT COUNT(*) FROM user_preferences") as cur:
            total_prefs = (await cur.fetchone())[0]

        # Count sessions.
        async with db.db.execute("SELECT COUNT(*) FROM session_summaries") as cur:
            total_sessions = (await cur.fetchone())[0]

        # Sum approval counts.
        async with db.db.execute("SELECT COALESCE(SUM(count), 0) FROM approval_patterns") as cur:
            total_approvals = (await cur.fetchone())[0]

        # Fetch preferences list.
        async with db.db.execute(
            "SELECT id, category, key, value, confidence, evidence_count, project_scope, last_seen "
            "FROM user_preferences ORDER BY confidence DESC LIMIT 100"
        ) as cur:
            rows = await cur.fetchall()
            preferences = [
                PreferenceItem(
                    id=row["id"],
                    category=row["category"],
                    key=row["key"],
                    value=row["value"],
                    confidence=row["confidence"],
                    evidence_count=row["evidence_count"],
                    project_scope=row["project_scope"] or None,
                    last_seen=row["last_seen"],
                )
                for row in rows
            ]

        return UserProfile(
            total_preferences=total_prefs,
            total_sessions=total_sessions,
            total_approvals=total_approvals,
            preferences=preferences,
        )
    except Exception:
        logger.exception("Failed to fetch profile data")
        return UserProfile()


async def _get_sessions_data(request: Request) -> list[SessionSummary]:
    db = _db(request)
    if db is None:
        return []

    try:
        async with db.db.execute(
            "SELECT session_id, project_dir, goal, outcome, tools_used, "
            "files_touched, started_at "
            "FROM session_summaries ORDER BY started_at DESC LIMIT 100"
        ) as cur:
            rows = await cur.fetchall()
            results = []
            for row in rows:
                started_at = row["started_at"] if row["started_at"] else None
                results.append(SessionSummary(
                    session_id=row["session_id"],
                    project_dir=row["project_dir"] or "",
                    goal=row["goal"] or "",
                    outcome=row["outcome"] or "",
                    tools_used=json.loads(row["tools_used"]) if row["tools_used"] else [],
                    files_touched=json.loads(row["files_touched"]) if row["files_touched"] else [],
                    started_at=started_at,
                ))
            return results
    except Exception:
        logger.exception("Failed to fetch sessions data")
        return []


async def _get_session_detail(request: Request, session_id: str) -> SessionDetail:
    db = _db(request)
    if db is None:
        return SessionDetail(session_id=session_id)

    try:
        row = await db.get_session(session_id)
        if row is None:
            return SessionDetail(session_id=session_id)

        return SessionDetail(
            session_id=row["session_id"],
            project_dir=row.get("project_dir") or "",
            goal=row.get("goal") or "",
            outcome=row.get("outcome") or "",
            tools_used=row.get("tools_used") or [],
            files_touched=row.get("files_touched") or [],
            started_at=row.get("started_at"),
            decisions_made=row.get("decisions_made") or [],
            patterns_observed=[
                {"pattern": p} if isinstance(p, str) else p
                for p in (row.get("patterns_observed") or [])
            ],
        )
    except Exception:
        logger.exception("Failed to fetch session detail %s", session_id)
        return SessionDetail(session_id=session_id)


async def _get_policies_data(request: Request) -> list[PolicyItem]:
    db = _db(request)
    if db is None:
        return []

    try:
        rows = await db.get_policies()
        return [
            PolicyItem(
                id=row.get("id"),
                policy_type=row.get("policy_type", ""),
                rule=row.get("rule", ""),
                source=row.get("source", ""),
                project_scope=row.get("project_scope"),
            )
            for row in rows
        ]
    except Exception:
        logger.exception("Failed to fetch policies data")
        return []


async def _get_approvals_data(request: Request) -> list[ApprovalPattern]:
    db = _db(request)
    if db is None:
        return []

    try:
        async with db.db.execute(
            "SELECT id, tool_name, action_pattern, decision, count, auto_approve, project_scope "
            "FROM approval_patterns ORDER BY count DESC LIMIT 100"
        ) as cur:
            rows = await cur.fetchall()
            return [
                ApprovalPattern(
                    id=row["id"],
                    tool_name=row["tool_name"],
                    action_pattern=row["action_pattern"],
                    decision=row["decision"],
                    count=row["count"],
                    auto_approve=bool(row["auto_approve"]),
                    project_scope=row["project_scope"] or None,
                )
                for row in rows
            ]
    except Exception:
        logger.exception("Failed to fetch approvals data")
        return []


async def _get_config_data(request: Request) -> ConfigResponse:
    try:
        from betty.config import load_config

        cfg = load_config()
        return ConfigResponse(
            llm_model=cfg.llm.model,
            llm_api_base=cfg.llm.api_base,
            delegation_level=cfg.delegation.autonomy_level,
            auto_approve_read_tools=cfg.delegation.auto_approve_read_tools,
            confidence_threshold=cfg.delegation.confidence_threshold,
            telegram_configured=bool(cfg.escalation.telegram_token),
            port=7832,
        )
    except Exception:
        logger.exception("Failed to load config")
        return ConfigResponse()
