"""Hook endpoints — fast-path API for Claude Code hooks (<100ms)."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from .models import (
    HookPostToolUseRequest,
    HookPostToolUseResponse,
    HookPreToolUseRequest,
    HookPreToolUseResponse,
    HookPromptSubmitRequest,
    HookPromptSubmitResponse,
    ToolDecision,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hooks", tags=["hooks"])


@router.post("/prompt-submit", response_model=HookPromptSubmitResponse)
async def hook_prompt_submit(req: HookPromptSubmitRequest) -> HookPromptSubmitResponse:
    logger.debug("prompt-submit session=%s len=%d", req.session_id, len(req.prompt))
    return HookPromptSubmitResponse(proceed=True)


@router.post("/pre-tool-use", response_model=HookPreToolUseResponse)
async def hook_pre_tool_use(req: HookPreToolUseRequest) -> HookPreToolUseResponse:
    logger.debug("pre-tool-use session=%s tool=%s", req.session_id, req.tool_name)
    return HookPreToolUseResponse(decision=ToolDecision.allow)


@router.post("/post-tool-use", response_model=HookPostToolUseResponse)
async def hook_post_tool_use(req: HookPostToolUseRequest) -> HookPostToolUseResponse:
    logger.debug("post-tool-use session=%s tool=%s", req.session_id, req.tool_name)
    return HookPostToolUseResponse(acknowledged=True)
