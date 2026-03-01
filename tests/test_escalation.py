"""Tests for betty.escalation — escalation router."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from betty.escalation import (
    Channel,
    EscalationRouter,
    Question,
    Urgency,
    UserResponse,
)


def _make_question(
    urgency: Urgency = Urgency.MEDIUM,
    text: str = "What framework?",
    default: str | None = None,
    options: list[str] | None = None,
) -> Question:
    return Question(
        text=text,
        urgency=urgency,
        default=default,
        options=options,
        source="test",
    )


async def _mock_telegram_sender(question: Question) -> UserResponse | None:
    """Simulate a Telegram reply."""
    return UserResponse(
        answer="telegram-reply",
        channel=Channel.TELEGRAM,
        response_time_secs=0.1,
    )


# ── Channel routing ───────────────────────────────────────────────────────


class TestPickChannel:
    def test_high_urgency_active_user_goes_to_tui(self):
        router = EscalationRouter()
        with mock.patch.object(router, "is_user_active", return_value=True):
            channel = router._pick_channel(Urgency.HIGH)
        assert channel == Channel.TUI

    def test_low_urgency_with_telegram_goes_to_telegram(self):
        router = EscalationRouter(telegram_sender=_mock_telegram_sender)
        channel = router._pick_channel(Urgency.LOW)
        assert channel == Channel.TELEGRAM

    def test_medium_urgency_with_telegram_goes_to_telegram(self):
        router = EscalationRouter(telegram_sender=_mock_telegram_sender)
        channel = router._pick_channel(Urgency.MEDIUM)
        assert channel == Channel.TELEGRAM

    def test_away_user_no_telegram_goes_to_queue(self):
        router = EscalationRouter()
        with mock.patch.object(router, "is_user_active", return_value=False):
            channel = router._pick_channel(Urgency.HIGH)
        assert channel == Channel.QUEUE

    def test_active_user_no_telegram_goes_to_tui(self):
        router = EscalationRouter()
        with mock.patch.object(router, "is_user_active", return_value=True):
            channel = router._pick_channel(Urgency.LOW)
        assert channel == Channel.TUI

    def test_preferred_channel_telegram(self):
        router = EscalationRouter(
            preferred_channel=Channel.TELEGRAM,
            telegram_sender=_mock_telegram_sender,
        )
        channel = router._pick_channel(Urgency.HIGH)
        assert channel == Channel.TELEGRAM

    def test_preferred_channel_tui(self):
        router = EscalationRouter(preferred_channel=Channel.TUI)
        channel = router._pick_channel(Urgency.LOW)
        assert channel == Channel.TUI


# ── Escalation end-to-end ─────────────────────────────────────────────────


class TestEscalate:
    @pytest.mark.asyncio
    async def test_telegram_escalation(self):
        router = EscalationRouter(telegram_sender=_mock_telegram_sender)
        question = _make_question(urgency=Urgency.MEDIUM)
        response = await router.escalate(question)

        assert response.answer == "telegram-reply"
        assert response.channel == Channel.TELEGRAM
        assert not response.timed_out

    @pytest.mark.asyncio
    async def test_tui_escalation(self):
        router = EscalationRouter()
        question = _make_question(urgency=Urgency.HIGH)

        with mock.patch.object(router, "is_user_active", return_value=True):
            with mock.patch("asyncio.to_thread", return_value="user-answer"):
                response = await router.escalate(question)

        assert response.answer == "user-answer"
        assert response.channel == Channel.TUI
        assert not response.timed_out

    @pytest.mark.asyncio
    async def test_timeout_returns_default(self):
        router = EscalationRouter(timeout_secs=0.01)
        question = _make_question(
            urgency=Urgency.HIGH,
            default="use-default",
        )

        async def slow_telegram(q: Question) -> UserResponse | None:
            await asyncio.sleep(5)
            return None

        router.set_telegram_sender(slow_telegram)
        with mock.patch.object(router, "_pick_channel", return_value=Channel.TELEGRAM):
            response = await router.escalate(question)

        assert response.timed_out
        assert response.answer == "use-default"

    @pytest.mark.asyncio
    async def test_timeout_empty_when_no_default(self):
        router = EscalationRouter(timeout_secs=0.01)
        question = _make_question(urgency=Urgency.HIGH)

        with mock.patch.object(router, "_pick_channel", return_value=Channel.QUEUE):
            response = await router.escalate(question)

        assert response.timed_out
        assert response.answer == ""

    @pytest.mark.asyncio
    async def test_escalation_logged(self):
        router = EscalationRouter(telegram_sender=_mock_telegram_sender)
        question = _make_question(urgency=Urgency.LOW)
        await router.escalate(question)

        log = router.get_log()
        assert len(log) == 1
        assert log[0].question is question
        assert log[0].response is not None
        assert log[0].response.answer == "telegram-reply"


# ── Queue channel ─────────────────────────────────────────────────────────


class TestQueueChannel:
    @pytest.mark.asyncio
    async def test_answer_queued(self):
        router = EscalationRouter(timeout_secs=5.0)
        question = _make_question(urgency=Urgency.HIGH)

        async def escalate_and_check():
            with mock.patch.object(
                router, "_pick_channel", return_value=Channel.QUEUE
            ):
                return await router.escalate(question)

        # Start escalation in background
        task = asyncio.create_task(escalate_and_check())
        # Give it time to queue
        await asyncio.sleep(0.05)
        # Answer the queued question
        answered = await router.answer_queued("queued-answer")
        assert answered

        response = await task
        assert response.answer == "queued-answer"
        assert response.channel == Channel.QUEUE
        assert not response.timed_out

    @pytest.mark.asyncio
    async def test_answer_queued_nothing_pending(self):
        router = EscalationRouter()
        answered = await router.answer_queued("no-one-home")
        assert not answered


# ── Properties & setters ──────────────────────────────────────────────────


class TestRouterProperties:
    def test_has_telegram_false(self):
        router = EscalationRouter()
        assert not router.has_telegram

    def test_has_telegram_true(self):
        router = EscalationRouter(telegram_sender=_mock_telegram_sender)
        assert router.has_telegram

    def test_set_preferred_channel(self):
        router = EscalationRouter()
        router.set_preferred_channel(Channel.TELEGRAM)
        assert router._preferred_channel == Channel.TELEGRAM

    def test_set_telegram_sender(self):
        router = EscalationRouter()
        assert not router.has_telegram
        router.set_telegram_sender(_mock_telegram_sender)
        assert router.has_telegram

    def test_pending_count(self):
        router = EscalationRouter()
        assert router.get_pending_count() == 0


# ── TUI prompt formatting ────────────────────────────────────────────────


class TestFormatTuiPrompt:
    def test_simple_question(self):
        q = _make_question(text="Which DB?")
        prompt = EscalationRouter._format_tui_prompt(q)
        assert "[Betty] Which DB?" in prompt
        assert "> " in prompt

    def test_with_options(self):
        q = _make_question(text="Pick one", options=["A", "B", "C"])
        prompt = EscalationRouter._format_tui_prompt(q)
        assert "1. A" in prompt
        assert "2. B" in prompt
        assert "3. C" in prompt

    def test_with_default(self):
        q = _make_question(text="Framework?", default="FastAPI")
        prompt = EscalationRouter._format_tui_prompt(q)
        assert "(default: FastAPI)" in prompt
