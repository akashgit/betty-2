"""Tests for betty.telegram_bot — Telegram escalation channel."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from betty.escalation import Channel, Question, Urgency, UserResponse
from betty.telegram_bot import TelegramBot, make_telegram_sender


def _make_question(**kwargs) -> Question:
    defaults = {
        "text": "Which database?",
        "urgency": Urgency.MEDIUM,
        "source": "test",
    }
    defaults.update(kwargs)
    return Question(**defaults)


# ── Configuration ─────────────────────────────────────────────────────────


class TestConfiguration:
    def test_not_configured_without_chat_id(self):
        bot = TelegramBot(token="test-token")
        assert not bot.is_configured

    def test_configured_with_both(self):
        bot = TelegramBot(token="test-token", chat_id=12345)
        assert bot.is_configured

    def test_not_configured_without_token(self):
        bot = TelegramBot(token="", chat_id=12345)
        assert not bot.is_configured

    def test_set_chat_id(self):
        bot = TelegramBot(token="test-token")
        bot.chat_id = 99999
        assert bot.chat_id == 99999
        assert bot.is_configured


# ── Message formatting ────────────────────────────────────────────────────


class TestFormatMessage:
    def test_basic_question(self):
        q = _make_question(text="Which framework?")
        msg = TelegramBot._format_message(q)
        assert "Betty needs your input" in msg
        assert "Which framework?" in msg

    def test_with_session_id(self):
        q = _make_question(session_id="abc-123")
        msg = TelegramBot._format_message(q)
        assert "abc-123" in msg

    def test_with_source(self):
        q = _make_question(source="intent_engine")
        msg = TelegramBot._format_message(q)
        assert "intent_engine" in msg

    def test_with_default(self):
        q = _make_question(default="FastAPI")
        msg = TelegramBot._format_message(q)
        assert "FastAPI" in msg


# ── Send question (no Telegram) ──────────────────────────────────────────


class TestSendQuestionUnconfigured:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_configured(self):
        bot = TelegramBot(token="test-token")
        result = await bot.send_question(_make_question())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_running(self):
        bot = TelegramBot(token="test-token", chat_id=12345)
        result = await bot.send_question(_make_question())
        assert result is None


# ── Send alert ────────────────────────────────────────────────────────────


class TestSendAlert:
    @pytest.mark.asyncio
    async def test_returns_false_when_not_configured(self):
        bot = TelegramBot(token="test-token")
        result = await bot.send_alert("test alert")
        assert result is False


# ── Rate limiting ─────────────────────────────────────────────────────────


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        bot = TelegramBot(token="t", chat_id=1, rate_limit_secs=0.1)
        bot._last_message_time = asyncio.get_event_loop().time()

        start = asyncio.get_event_loop().time()
        await bot._enforce_rate_limit()
        elapsed = asyncio.get_event_loop().time() - start

        # Should have waited approximately 0.1 seconds
        assert elapsed >= 0.05


# ── make_telegram_sender ─────────────────────────────────────────────────


class TestMakeTelegramSender:
    @pytest.mark.asyncio
    async def test_sender_delegates_to_bot(self):
        bot = TelegramBot(token="test-token")
        sender = make_telegram_sender(bot)

        # Bot is not running, should return None
        result = await sender(_make_question())
        assert result is None

    def test_sender_is_callable(self):
        bot = TelegramBot(token="test-token")
        sender = make_telegram_sender(bot)
        assert callable(sender)


# ── Handler logic (unit-level) ───────────────────────────────────────────


class TestHandlerLogic:
    @pytest.mark.asyncio
    async def test_handle_text_no_pending(self):
        bot = TelegramBot(token="test-token", chat_id=12345)
        bot._running = True

        # Simulate an update
        update = mock.AsyncMock()
        update.message.text = "some answer"
        context = mock.AsyncMock()

        await bot._handle_text(update, context)
        update.message.reply_text.assert_called_once_with(
            "No pending questions right now."
        )

    @pytest.mark.asyncio
    async def test_handle_text_with_pending(self):
        bot = TelegramBot(token="test-token", chat_id=12345)
        bot._running = True

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[UserResponse] = loop.create_future()
        bot._pending["test_cb"] = fut

        update = mock.AsyncMock()
        update.message.text = "my answer"
        context = mock.AsyncMock()

        await bot._handle_text(update, context)

        assert fut.done()
        result = fut.result()
        assert result.answer == "my answer"
        assert result.channel == Channel.TELEGRAM

    @pytest.mark.asyncio
    async def test_handle_start_sets_chat_id(self):
        bot = TelegramBot(token="test-token")

        update = mock.AsyncMock()
        update.effective_chat.id = 42
        context = mock.AsyncMock()

        await bot._handle_start(update, context)
        assert bot.chat_id == 42

    @pytest.mark.asyncio
    async def test_handle_status(self):
        bot = TelegramBot(token="test-token", chat_id=12345)
        bot._running = True

        update = mock.AsyncMock()
        context = mock.AsyncMock()

        await bot._handle_status(update, context)
        call_args = update.message.reply_text.call_args[0][0]
        assert "running" in call_args
        assert "Pending questions: 0" in call_args
