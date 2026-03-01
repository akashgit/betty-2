"""FastAPI application factory for the Betty daemon."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from betty import __version__

from .dashboard import router as dashboard_router
from .hooks import router as hooks_router
from .models import HealthResponse

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7832

_PKG_DIR = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _PKG_DIR / "templates"
_STATIC_DIR = _PKG_DIR / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # If no DB was injected by the daemon, connect one for dev mode.
        own_db = None
        if getattr(app.state, "db", None) is None:
            try:
                from betty.db import UserModelDB

                own_db = UserModelDB()
                await own_db.connect()
                app.state.db = own_db
                logger.info("Dev mode: connected database")
            except Exception:
                logger.warning("Dev mode: database unavailable, running without DB")
        yield
        if own_db is not None:
            await own_db.close()

    app = FastAPI(title="Betty", version=__version__,
                  description="Peer programming agent for Claude Code",
                  lifespan=lifespan)

    app.state.start_time = time.monotonic()
    app.state.templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            uptime_seconds=time.monotonic() - app.state.start_time,
        )

    app.include_router(hooks_router)
    app.include_router(dashboard_router)

    return app
