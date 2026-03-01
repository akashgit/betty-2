"""Hook endpoints — fast-path API for Claude Code hooks (<100ms).

Routes match Claude Code hook type names so the hook_handler can POST
to ``/hooks/{hook_type}`` without translation.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

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


@router.post("/UserPromptSubmit", response_model=HookPromptSubmitResponse)
async def hook_prompt_submit(
    req: HookPromptSubmitRequest,
    request: Request,
) -> HookPromptSubmitResponse:
    """Handle UserPromptSubmit hook — analyze intent and surface questions."""
    logger.debug("UserPromptSubmit session=%s len=%d", req.session_id, len(req.prompt))

    intent_engine = getattr(request.app.state, "intent_engine", None)
    if intent_engine is None:
        return HookPromptSubmitResponse(proceed=True)

    try:
        # Overall timeout: must respond before hook_handler's timeout.
        analysis = await asyncio.wait_for(
            intent_engine.analyze(req.prompt, req.project_dir or None),
            timeout=35.0,
        )

        return HookPromptSubmitResponse(
            proceed=True,
            questions=[q.text for q in analysis.questions],
            suggested_context=analysis.suggested_context or None,
            similar_sessions=analysis.similar_sessions,
            applicable_policies=analysis.applicable_policies,
            predicted_plan=analysis.predicted_plan or None,
            confidence=analysis.confidence,
        )
    except asyncio.TimeoutError:
        logger.warning("UserPromptSubmit timed out, proceeding without analysis")
        return HookPromptSubmitResponse(proceed=True)
    except Exception:
        logger.exception("UserPromptSubmit intent analysis failed")
        return HookPromptSubmitResponse(proceed=True)


@router.post("/PreToolUse", response_model=HookPreToolUseResponse)
async def hook_pre_tool_use(
    req: HookPreToolUseRequest,
    request: Request,
) -> HookPreToolUseResponse:
    """Handle PreToolUse hook — consult ApprovalModel for tool decisions."""
    logger.debug("PreToolUse session=%s tool=%s", req.session_id, req.tool_name)

    approval_model = getattr(request.app.state, "approval_model", None)
    if approval_model is None:
        return HookPreToolUseResponse(decision=ToolDecision.allow)

    from betty.approval import ApprovalDecision

    prediction = approval_model.predict(
        tool_name=req.tool_name,
        tool_input=req.tool_input,
    )

    if prediction.decision == ApprovalDecision.APPROVE:
        decision = ToolDecision.allow
    elif prediction.decision == ApprovalDecision.REJECT:
        decision = ToolDecision.block
    else:
        decision = ToolDecision.ask

    return HookPreToolUseResponse(decision=decision, reason=prediction.reason)


@router.post("/PostToolUse", response_model=HookPostToolUseResponse)
async def hook_post_tool_use(
    req: HookPostToolUseRequest,
    request: Request,
) -> HookPostToolUseResponse:
    """Handle PostToolUse hook — record the tool use for future predictions."""
    logger.debug("PostToolUse session=%s tool=%s", req.session_id, req.tool_name)

    approval_model = getattr(request.app.state, "approval_model", None)
    db = getattr(request.app.state, "db", None)

    if approval_model is not None:
        from betty.approval import make_action_pattern

        action_pattern = make_action_pattern(req.tool_name, req.tool_input)
        approval_model.record(
            tool_name=req.tool_name,
            tool_input=req.tool_input,
            decision="accepted",
        )

        if db is not None:
            try:
                await db.record_approval(
                    tool_name=req.tool_name,
                    action_pattern=action_pattern,
                    decision="accepted",
                )
            except Exception:
                logger.exception("Failed to persist approval record")

    return HookPostToolUseResponse(acknowledged=True)
