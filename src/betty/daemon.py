"""Betty daemon — main process that orchestrates all components.

Starts the FastAPI server, session watcher, Telegram bot, and scheduler
as concurrent async tasks.  Manages PID file and graceful shutdown.

Initializes shared state (database, approval model, escalation router)
and injects it into the FastAPI app for hook endpoints to use.
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


# -- Shared state initialization ------------------------------------------


async def _init_shared_state() -> dict:
    """Initialize shared components: database, approval model, escalation router."""
    from betty.approval import ApprovalModel
    from betty.config import load_config
    from betty.db import UserModelDB
    from betty.escalation import EscalationRouter

    cfg = load_config()

    # Database.
    db = UserModelDB()
    await db.connect()
    logger.info("Database connected")

    # Approval model.
    approval_model = ApprovalModel(
        autonomy_level=cfg.delegation.autonomy_level,
        confidence_threshold=cfg.delegation.confidence_threshold,
    )

    # Load existing approval patterns from DB.
    try:
        async with db.db.execute(
            "SELECT tool_name, action_pattern, decision, count, project_scope "
            "FROM approval_patterns"
        ) as cursor:
            rows = await cursor.fetchall()
            records = [dict(row) for row in rows]
        approval_model.load_patterns(records)
        logger.info("Loaded %d approval patterns", len(records))
    except Exception:
        logger.exception("Failed to load approval patterns")

    # Escalation router.
    escalation_router = EscalationRouter(timeout_secs=120.0)

    return {
        "db": db,
        "approval_model": approval_model,
        "escalation_router": escalation_router,
        "config": cfg,
    }


# -- Main daemon loop -----------------------------------------------------


async def _run_server(port: int, shared_state: dict) -> None:
    """Run the FastAPI server via uvicorn with shared state injected."""
    import uvicorn

    from betty.server.app import create_app

    app = create_app()

    # Inject shared state into the FastAPI app for hook endpoints.
    for key, value in shared_state.items():
        setattr(app.state, key, value)

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _run_telegram_bot(shared_state: dict) -> None:
    """Start the Telegram bot if configured."""
    try:
        from betty.telegram_bot import TelegramBot, make_telegram_sender

        cfg = shared_state["config"]
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

        # Wire Telegram sender into escalation router.
        escalation_router = shared_state.get("escalation_router")
        if escalation_router is not None:
            escalation_router.set_telegram_sender(make_telegram_sender(bot))
            logger.info("Telegram sender connected to escalation router")

        # Keep running until cancelled.
        try:
            await asyncio.Event().wait()
        finally:
            await bot.stop()
    except ImportError:
        logger.info("python-telegram-bot not installed, skipping")
    except Exception:
        logger.exception("Telegram bot failed")


async def _run_scheduler(shared_state: dict) -> None:
    """Periodic background tasks: preference decay."""
    db = shared_state.get("db")
    tick_count = 0

    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes.
            tick_count += 1
            logger.debug("Scheduler tick %d", tick_count)

            # Every hour (12 ticks): decay stale preferences.
            if db and tick_count % 12 == 0:
                try:
                    from betty.user_model import UserModel

                    user_model = UserModel(db)
                    updated = await user_model.decay_stale_preferences()
                    if updated:
                        logger.info("Decayed %d stale preferences", updated)
                except Exception:
                    logger.exception("Preference decay failed")

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

    # Initialize shared components.
    shared_state = await _init_shared_state()

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    # Create component tasks.
    tasks = [
        asyncio.create_task(_run_server(port, shared_state), name="server"),
        asyncio.create_task(_run_telegram_bot(shared_state), name="telegram"),
        asyncio.create_task(_run_scheduler(shared_state), name="scheduler"),
    ]

    try:
        # Wait for shutdown signal.
        await shutdown_event.wait()
    finally:
        logger.info("Shutting down...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Close database.
        db = shared_state.get("db")
        if db:
            await db.close()
            logger.info("Database closed")

        _remove_pid()
        logger.info("Betty daemon stopped")
