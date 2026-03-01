"""Betty daemon — main process that orchestrates all components.

Starts the FastAPI server, session watcher, Telegram bot, and scheduler
as concurrent async tasks.  Manages PID file and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger("betty")

BETTY_DIR = Path.home() / ".betty"
PID_FILE = BETTY_DIR / "betty.pid"
LOG_FILE = BETTY_DIR / "betty.log"
DEFAULT_PORT = 7832


def _setup_logging(log_level: str = "INFO") -> None:
    """Configure rotating file logger and optional stderr output."""
    BETTY_DIR.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger("betty")
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.addHandler(handler)

    # Also log to stderr when attached to a terminal.
    if sys.stderr.isatty():
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(
            logging.Formatter("%(levelname)-8s %(message)s")
        )
        root.addHandler(stderr_handler)


# -- PID file management --------------------------------------------------


def _write_pid() -> None:
    BETTY_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _remove_pid() -> None:
    PID_FILE.unlink(missing_ok=True)


def _read_pid() -> int | None:
    """Read the PID from the file, or None if not running."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process is still alive.
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        # Stale PID file.
        _remove_pid()
        return None


def is_running() -> bool:
    """Return True if the Betty daemon is currently running."""
    return _read_pid() is not None


def stop_daemon() -> bool:
    """Send SIGTERM to the running daemon.  Returns True if signal was sent."""
    pid = _read_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        _remove_pid()
        return False


# -- Main daemon loop -----------------------------------------------------


async def _run_server(port: int) -> None:
    """Run the FastAPI server via uvicorn."""
    import uvicorn

    from betty.server.app import create_app

    app = create_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _run_telegram_bot() -> None:
    """Start the Telegram bot if configured."""
    try:
        from betty.config import load_config
        from betty.telegram_bot import TelegramBot

        cfg = load_config()
        token = cfg.escalation.telegram_token
        chat_id = cfg.escalation.telegram_chat_id

        if not token:
            logger.info("Telegram not configured, skipping bot")
            return

        bot = TelegramBot(
            token=token,
            chat_id=int(chat_id) if chat_id else None,
        )
        await bot.start()
        logger.info("Telegram bot started")

        # Keep running until cancelled.
        try:
            await asyncio.Event().wait()
        finally:
            await bot.stop()
    except ImportError:
        logger.info("python-telegram-bot not installed, skipping")
    except Exception:
        logger.exception("Telegram bot failed")


async def _run_scheduler() -> None:
    """Periodic background tasks."""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes.
            logger.debug("Scheduler tick")
        except asyncio.CancelledError:
            break


async def run_daemon(port: int = DEFAULT_PORT) -> None:
    """Start all daemon components and run until shutdown."""
    existing = _read_pid()
    if existing is not None:
        logger.error("Daemon already running (PID %d)", existing)
        raise SystemExit(1)

    _setup_logging()
    _write_pid()
    logger.info("Betty daemon starting (PID %d, port %d)", os.getpid(), port)

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    # Create component tasks.
    tasks = [
        asyncio.create_task(_run_server(port), name="server"),
        asyncio.create_task(_run_telegram_bot(), name="telegram"),
        asyncio.create_task(_run_scheduler(), name="scheduler"),
    ]

    try:
        # Wait for shutdown signal.
        await shutdown_event.wait()
    finally:
        logger.info("Shutting down...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _remove_pid()
        logger.info("Betty daemon stopped")
