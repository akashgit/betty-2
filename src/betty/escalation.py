"""Escalation router — decide when and how to reach the user.

Routes questions to the appropriate channel (TUI, Telegram, or queue)
based on urgency and user availability.  All escalations are logged.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class Urgency(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Channel(str, enum.Enum):
    TUI = "tui"
    TELEGRAM = "telegram"
    QUEUE = "queue"


@dataclass
class Question:
    """A question that needs user input."""

    text: str
    urgency: Urgency = Urgency.MEDIUM
    options: list[str] | None = None
    default: str | None = None
    context: str | None = None
    source: str = ""  # e.g. "intent_engine", "approval_model"
    session_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class UserResponse:
    """The user's response to an escalated question."""

    answer: str
    channel: Channel
    response_time_secs: float
    timed_out: bool = False


@dataclass
class EscalationRecord:
    """Log entry for an escalation attempt."""

    question: Question
    channel: Channel
    response: UserResponse | None = None
    timestamp: float = field(default_factory=time.time)


class EscalationRouter:
    """Routes questions to the appropriate channel based on urgency and
    user availability.

    Usage::

        router = EscalationRouter()
        response = await router.escalate(question)
    """

    def __init__(
        self,
        *,
        timeout_secs: float = 120.0,
        preferred_channel: Channel | None = None,
        telegram_sender: Callable[
            [Question], Coroutine[Any, Any, UserResponse | None]
        ]
        | None = None,
    ) -> None:
        self._timeout_secs = timeout_secs
        self._preferred_channel = preferred_channel
        self._telegram_sender = telegram_sender
        self._log: list[EscalationRecord] = []
        self._queue: asyncio.Queue[Question] = asyncio.Queue()
        self._pending_responses: dict[
            float, asyncio.Future[UserResponse]
        ] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def escalate(self, question: Question) -> UserResponse:
        """Route *question* to the best channel and return the response.

        If no response is received within the timeout, returns a
        ``UserResponse`` with ``timed_out=True`` and the question's
        default answer (or empty string).
        """
        channel = self._pick_channel(question.urgency)
        record = EscalationRecord(
            question=question,
            channel=channel,
        )

        start = time.monotonic()
        response: UserResponse | None = None

        try:
            if channel == Channel.TUI:
                response = await self._escalate_tui(question)
            elif channel == Channel.TELEGRAM:
                response = await self._escalate_telegram(question)
            else:
                response = await self._escalate_queue(question)
        except asyncio.TimeoutError:
            pass

        if response is None:
            elapsed = time.monotonic() - start
            response = UserResponse(
                answer=question.default or "",
                channel=channel,
                response_time_secs=elapsed,
                timed_out=True,
            )
            logger.info(
                "Escalation timed out after %.1fs, using default: %r",
                elapsed,
                response.answer,
            )

        record.response = response
        self._log.append(record)
        return response

    def is_user_active(self) -> bool:
        """Heuristic: is the user likely at the terminal?

        Checks whether stdout is connected to a TTY as a simple proxy.
        """
        try:
            return os.isatty(1)
        except (OSError, ValueError):
            return False

    def set_preferred_channel(self, channel: Channel) -> None:
        """Set the user's preferred notification channel."""
        self._preferred_channel = channel

    def set_telegram_sender(
        self,
        sender: Callable[
            [Question], Coroutine[Any, Any, UserResponse | None]
        ],
    ) -> None:
        """Register the async callable used to send Telegram messages."""
        self._telegram_sender = sender

    @property
    def has_telegram(self) -> bool:
        return self._telegram_sender is not None

    def get_log(self) -> list[EscalationRecord]:
        """Return a copy of the escalation log."""
        return list(self._log)

    def get_pending_count(self) -> int:
        """Return how many questions are sitting in the queue."""
        return self._queue.qsize()

    async def answer_queued(
        self, answer: str, channel: Channel = Channel.QUEUE
    ) -> bool:
        """Supply an answer for the oldest queued question.

        Returns ``True`` if a waiting question was answered.
        """
        for ts, fut in list(self._pending_responses.items()):
            if not fut.done():
                fut.set_result(
                    UserResponse(
                        answer=answer,
                        channel=channel,
                        response_time_secs=time.monotonic() - ts,
                    )
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _pick_channel(self, urgency: Urgency) -> Channel:
        """Determine the best channel for a question."""
        # Explicit preference wins.
        if self._preferred_channel is not None:
            if self._preferred_channel == Channel.TELEGRAM and self.has_telegram:
                return Channel.TELEGRAM
            if self._preferred_channel == Channel.TUI:
                return Channel.TUI

        active = self.is_user_active()

        if urgency == Urgency.HIGH and active:
            return Channel.TUI
        if urgency in (Urgency.LOW, Urgency.MEDIUM) and self.has_telegram:
            return Channel.TELEGRAM
        if not active and self.has_telegram:
            return Channel.TELEGRAM

        # Fallback: queue if user is away and no Telegram, or TUI if active.
        return Channel.TUI if active else Channel.QUEUE

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    async def _escalate_tui(self, question: Question) -> UserResponse | None:
        """Prompt the user in the terminal."""
        start = time.monotonic()

        prompt = self._format_tui_prompt(question)

        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(input, prompt),
                timeout=self._timeout_secs,
            )
        except (asyncio.TimeoutError, EOFError, OSError):
            return None

        return UserResponse(
            answer=answer.strip(),
            channel=Channel.TUI,
            response_time_secs=time.monotonic() - start,
        )

    async def _escalate_telegram(
        self, question: Question
    ) -> UserResponse | None:
        """Send the question via Telegram and wait for a reply."""
        if self._telegram_sender is None:
            return None

        try:
            return await asyncio.wait_for(
                self._telegram_sender(question),
                timeout=self._timeout_secs,
            )
        except asyncio.TimeoutError:
            return None

    async def _escalate_queue(
        self, question: Question
    ) -> UserResponse | None:
        """Put the question in a queue for later answering."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[UserResponse] = loop.create_future()
        ts = time.monotonic()
        self._pending_responses[ts] = fut

        try:
            return await asyncio.wait_for(fut, timeout=self._timeout_secs)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_responses.pop(ts, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tui_prompt(question: Question) -> str:
        """Build a formatted terminal prompt string."""
        parts: list[str] = [f"\n[Betty] {question.text}"]
        if question.options:
            for i, opt in enumerate(question.options, 1):
                parts.append(f"  {i}. {opt}")
        if question.default:
            parts.append(f"  (default: {question.default})")
        parts.append("> ")
        return "\n".join(parts)
