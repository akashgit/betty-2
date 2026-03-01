"""Pydantic request/response models for the Betty API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Hook models (fast-path, <100ms)
# ---------------------------------------------------------------------------

class HookPromptSubmitRequest(BaseModel):
    session_id: str
    prompt: str
    project_dir: str = ""

class HookPromptSubmitResponse(BaseModel):
    proceed: bool = True
    enriched_prompt: str | None = None
    questions: list[str] = Field(default_factory=list)
    suggested_context: str | None = None
    similar_sessions: list[dict[str, Any]] = Field(default_factory=list)
    applicable_policies: list[dict[str, Any]] = Field(default_factory=list)
    predicted_plan: str | None = None
    confidence: float = 0.0

class ToolDecision(str, Enum):
    allow = "allow"
    block = "block"
    ask = "ask"

class HookPreToolUseRequest(BaseModel):
    session_id: str
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)

class HookPreToolUseResponse(BaseModel):
    decision: ToolDecision = ToolDecision.allow
    reason: str | None = None

class HookPostToolUseRequest(BaseModel):
    session_id: str
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: str = ""

class HookPostToolUseResponse(BaseModel):
    acknowledged: bool = True


# ---------------------------------------------------------------------------
# Dashboard API models
# ---------------------------------------------------------------------------

class PreferenceItem(BaseModel):
    id: int
    category: str
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = 0
    project_scope: str | None = None
    last_seen: datetime | None = None

class UserProfile(BaseModel):
    total_preferences: int = 0
    total_sessions: int = 0
    total_approvals: int = 0
    preferences: list[PreferenceItem] = Field(default_factory=list)

class SessionSummary(BaseModel):
    session_id: str
    project_dir: str = ""
    goal: str = ""
    outcome: str = ""
    tools_used: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    turn_count: int = 0

class SessionDetail(SessionSummary):
    decisions_made: list[dict[str, Any]] = Field(default_factory=list)
    patterns_observed: list[dict[str, Any]] = Field(default_factory=list)

class PolicyItem(BaseModel):
    id: int | None = None
    policy_type: str
    rule: str
    source: str = ""
    project_scope: str | None = None

class PolicyCreate(BaseModel):
    policy_type: str
    rule: str
    source: str = ""
    project_scope: str | None = None

class ApprovalPattern(BaseModel):
    id: int
    tool_name: str
    action_pattern: str
    decision: str
    count: int = 0
    auto_approve: bool = False
    project_scope: str | None = None

class EscalationRecord(BaseModel):
    id: int
    session_id: str
    reason: str
    channel: str
    resolved: bool = False
    created_at: datetime | None = None

class ConfigResponse(BaseModel):
    llm_model: str = ""
    llm_api_base: str | None = None
    delegation_level: int = 1
    auto_approve_read_tools: bool = True
    confidence_threshold: float = 0.8
    telegram_configured: bool = False
    port: int = 7832

class ConfigUpdate(BaseModel):
    llm_model: str | None = None
    llm_api_base: str | None = None
    delegation_level: int | None = None
    auto_approve_read_tools: bool | None = None
    confidence_threshold: float | None = None

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    uptime_seconds: float = 0.0
