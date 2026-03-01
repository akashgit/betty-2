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
    """Initialize shared components: database, models, engines."""
    from betty.approval import ApprovalModel
    from betty.claude_md import ClaudeMdMaintainer
    from betty.config import load_config
    from betty.db import UserModelDB
    from betty.escalation import EscalationRouter
    from betty.intent_engine import IntentEngine
    from betty.policy import PolicyEngine
    from betty.session_analyzer import SessionAnalyzer
    from betty.session_search import SessionSearcher
    from betty.user_model import UserModel

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

    # LLM service (optional — heuristic mode if unavailable).
    llm = None
    try:
        from betty.llm import LLMService

        llm = LLMService(cfg.llm)
        logger.info("LLM service initialized (model=%s)", cfg.llm.model)
    except Exception:
        logger.info("LLM service unavailable, running in heuristic mode")

    # User model.
    user_model = UserModel(db)

    # Session searcher.
    searcher = SessionSearcher(db, llm)

    # Policy engine.
    policy_engine = PolicyEngine()

    # Session analyzer.
    session_analyzer = SessionAnalyzer(llm)

    # Intent engine.
    intent_engine = IntentEngine(
        user_model=user_model,
        searcher=searcher,
        db=db,
        llm=llm,
    )

    # CLAUDE.md maintainer.
    claude_md_maintainer = ClaudeMdMaintainer()

    return {
        "db": db,
        "approval_model": approval_model,
        "escalation_router": escalation_router,
        "config": cfg,
        "llm": llm,
        "user_model": user_model,
        "searcher": searcher,
        "policy_engine": policy_engine,
        "session_analyzer": session_analyzer,
        "intent_engine": intent_engine,
        "claude_md_maintainer": claude_md_maintainer,
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


async def _run_session_watcher(shared_state: dict) -> None:
    """Background task: discover and analyze completed sessions."""
    from betty.session_reader import discover_sessions, parse_session

    db = shared_state.get("db")
    session_analyzer = shared_state.get("session_analyzer")
    if not db or not session_analyzer:
        logger.info("Session watcher: missing db or analyzer, skipping")
        return

    # Seed analyzed IDs from existing session_summaries.
    analyzed_ids: set[str] = set()
    try:
        async with db.db.execute("SELECT session_id FROM session_summaries") as cursor:
            rows = await cursor.fetchall()
            analyzed_ids = {row["session_id"] for row in rows}
        logger.info("Session watcher: %d sessions already analyzed", len(analyzed_ids))
    except Exception:
        logger.exception("Session watcher: failed to seed analyzed IDs")

    while True:
        try:
            await asyncio.sleep(60)

            discovered = discover_sessions(limit=100)
            new_count = 0

            for session_id, path in discovered:
                if session_id in analyzed_ids:
                    continue

                try:
                    session = parse_session(path)

                    # Skip sessions with < 3 turns (likely still active).
                    if len(session.turns) < 3:
                        continue

                    # Analyze the session.
                    llm = shared_state.get("llm")
                    if llm is not None:
                        analysis = await session_analyzer.analyze_with_llm(session)
                    else:
                        analysis = session_analyzer.analyze(session)

                    # Persist results.
                    started_at = session.started_at.isoformat() if session.started_at else None
                    await session_analyzer.persist(analysis, db, started_at=started_at)
                    analyzed_ids.add(session_id)
                    new_count += 1
                except Exception:
                    logger.exception("Session watcher: failed to analyze %s", session_id)

            if new_count:
                logger.info("Session watcher: analyzed %d new sessions", new_count)

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Session watcher: unexpected error")


async def _run_scheduler(shared_state: dict) -> None:
    """Periodic background tasks: preference decay, CLAUDE.md maintenance."""
    db = shared_state.get("db")
    user_model = shared_state.get("user_model")
    claude_md_maintainer = shared_state.get("claude_md_maintainer")
    tick_count = 0

    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes.
            tick_count += 1
            logger.debug("Scheduler tick %d", tick_count)

            # Every hour (12 ticks): decay stale preferences.
            if db and user_model and tick_count % 12 == 0:
                try:
                    updated = await user_model.decay_stale_preferences()
                    if updated:
                        logger.info("Decayed %d stale preferences", updated)
                except Exception:
                    logger.exception("Preference decay failed")

            # Every 6 hours (72 ticks): CLAUDE.md maintenance.
            if db and user_model and claude_md_maintainer and tick_count % 72 == 0:
                try:
                    async with db.db.execute(
                        "SELECT DISTINCT project_dir FROM session_summaries"
                    ) as cursor:
                        rows = await cursor.fetchall()
                        project_dirs = [row["project_dir"] for row in rows if row["project_dir"]]

                    for project_dir in project_dirs:
                        try:
                            suggestions = await claude_md_maintainer.suggest_updates(
                                project_dir, user_model
                            )
                            auto_updates = [s for s in suggestions if s.should_auto_apply]
                            if auto_updates:
                                applied = claude_md_maintainer.apply_updates(
                                    project_dir, auto_updates, auto=True
                                )
                                if applied:
                                    logger.info(
                                        "CLAUDE.md: applied %d updates to %s",
                                        len(applied), project_dir,
                                    )
                        except Exception:
                            logger.exception(
                                "CLAUDE.md maintenance failed for %s", project_dir
                            )
                except Exception:
                    logger.exception("CLAUDE.md maintenance failed")

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
        asyncio.create_task(_run_session_watcher(shared_state), name="session_watcher"),
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
