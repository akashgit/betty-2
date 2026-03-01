"""Telegram bot — async escalation channel for Betty.

Sends formatted questions to the user via Telegram and receives
responses via inline keyboard buttons or free-form text.  Designed
to run as an async task inside the Betty daemon.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from betty.escalation import Channel, Question, UserResponse

logger = logging.getLogger(__name__)

# Rate limiting: minimum seconds between messages.
MIN_MESSAGE_INTERVAL = 5.0


class TelegramBot:
    """Async Telegram bot for Betty escalation.

    Usage::

        bot = TelegramBot(token="...", chat_id=12345)
        await bot.start()
        response = await bot.send_question(question)
        await bot.stop()
    """

    def __init__(
        self,
        token: str,
        chat_id: int | None = None,
        *,
        rate_limit_secs: float = MIN_MESSAGE_INTERVAL,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._rate_limit_secs = rate_limit_secs
        self._last_message_time: float = 0.0
        self._app: Any = None  # telegram.ext.Application
        self._pending: dict[str, asyncio.Future[UserResponse]] = {}
        self._running = False

    @property
    def is_configured(self) -> bool:
        """True if both token and chat_id are set."""
        return bool(self._token) and self._chat_id is not None

    @property
    def chat_id(self) -> int | None:
        return self._chat_id

    @chat_id.setter
    def chat_id(self, value: int) -> None:
        self._chat_id = value

    async def start(self) -> None:
        """Initialize and start the Telegram bot."""
        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CallbackQueryHandler,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.warning(
                "python-telegram-bot not installed; Telegram bot disabled"
            )
            return

        self._app = (
            Application.builder().token(self._token).build()
        )

        # Register handlers.
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("status", self._handle_status))
        self._app.add_handler(
            CallbackQueryHandler(self._handle_callback)
        )
        self._app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._handle_text,
            )
        )

        await self._app.initialize()
        await self._app.start()
        # Start polling in the background (non-blocking).
        await self._app.updater.start_polling(drop_pending_updates=True)
        self._running = True
        logger.info("Telegram bot started")

    async def stop(self) -> None:
        """Gracefully stop the bot."""
        if self._app and self._running:
            self._running = False
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram bot stopped")

    # ------------------------------------------------------------------
    # Sending questions
    # ------------------------------------------------------------------

    async def send_question(self, question: Question) -> UserResponse | None:
        """Send a question to the user and wait for their reply.

        Returns ``None`` if Telegram is not configured or the message
        fails to send.  The caller (EscalationRouter) handles timeouts.
        """
        if not self.is_configured or not self._running:
            return None

        await self._enforce_rate_limit()

        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        except ImportError:
            return None

        text = self._format_message(question)
        callback_id = f"q_{id(question)}_{time.monotonic()}"

        # Build inline keyboard if options provided.
        reply_markup = None
        if question.options:
            buttons = [
                [
                    InlineKeyboardButton(
                        opt, callback_data=f"{callback_id}:{i}"
                    )
                ]
                for i, opt in enumerate(question.options)
            ]
            reply_markup = InlineKeyboardMarkup(buttons)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[UserResponse] = loop.create_future()
        self._pending[callback_id] = fut

        start = time.monotonic()
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )
        except Exception as exc:
            logger.error("Failed to send Telegram message: %s", exc)
            self._pending.pop(callback_id, None)
            return None

        try:
            response = await fut
            return response
        except asyncio.CancelledError:
            self._pending.pop(callback_id, None)
            return None

    async def send_alert(self, message: str) -> bool:
        """Send a one-way alert message (no response expected)."""
        if not self.is_configured or not self._running:
            return False

        await self._enforce_rate_limit()

        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode="Markdown",
            )
            return True
        except Exception as exc:
            logger.error("Failed to send Telegram alert: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Telegram update handlers
    # ------------------------------------------------------------------

    async def _handle_start(self, update: Any, context: Any) -> None:
        """Handle /start command — used for linking."""
        chat_id = update.effective_chat.id
        self._chat_id = chat_id
        await update.message.reply_text(
            f"Betty linked to this chat (id: {chat_id}).\n"
            "You'll receive questions and alerts here."
        )
        logger.info("Telegram chat linked: %d", chat_id)

    async def _handle_status(self, update: Any, context: Any) -> None:
        """Handle /status command."""
        pending_count = len(
            [f for f in self._pending.values() if not f.done()]
        )
        await update.message.reply_text(
            f"Betty Telegram bot is running.\n"
            f"Pending questions: {pending_count}"
        )

    async def _handle_callback(self, update: Any, context: Any) -> None:
        """Handle inline keyboard button press."""
        query = update.callback_query
        await query.answer()

        data = query.data or ""
        parts = data.rsplit(":", 1)
        if len(parts) != 2:
            return

        callback_id, idx_str = parts
        fut = self._pending.pop(callback_id, None)
        if fut is None or fut.done():
            await query.edit_message_text(
                text=query.message.text + "\n\n(already answered)"
            )
            return

        try:
            idx = int(idx_str)
            answer = query.message.reply_markup.inline_keyboard[idx][0].text
        except (ValueError, IndexError):
            answer = idx_str

        fut.set_result(
            UserResponse(
                answer=answer,
                channel=Channel.TELEGRAM,
                response_time_secs=0.0,
            )
        )

        await query.edit_message_text(
            text=query.message.text + f"\n\n*Answered:* {answer}",
            parse_mode="Markdown",
        )

    async def _handle_text(self, update: Any, context: Any) -> None:
        """Handle free-form text response."""
        text = update.message.text.strip()

        # Find the oldest pending future and answer it.
        for callback_id, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_result(
                    UserResponse(
                        answer=text,
                        channel=Channel.TELEGRAM,
                        response_time_secs=0.0,
                    )
                )
                self._pending.pop(callback_id, None)
                await update.message.reply_text(f"Got it: {text}")
                return

        await update.message.reply_text(
            "No pending questions right now."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _enforce_rate_limit(self) -> None:
        """Wait if necessary to respect the rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_message_time
        if elapsed < self._rate_limit_secs:
            await asyncio.sleep(self._rate_limit_secs - elapsed)
        self._last_message_time = time.monotonic()

    @staticmethod
    def _format_message(question: Question) -> str:
        """Format a question for Telegram."""
        parts: list[str] = ["*Betty needs your input*\n"]

        if question.session_id:
            parts.append(f"_Session:_ `{question.session_id}`")
        if question.source:
            parts.append(f"_Source:_ {question.source}")

        parts.append(f"\n{question.text}")

        if question.default:
            parts.append(f"\n_Default:_ {question.default}")

        return "\n".join(parts)


def make_telegram_sender(
    bot: TelegramBot,
) -> Any:
    """Create an escalation-compatible sender function from a TelegramBot.

    Returns an async callable suitable for ``EscalationRouter(telegram_sender=...)``.
    """

    async def sender(question: Question) -> UserResponse | None:
        return await bot.send_question(question)

    return sender
