"""FastAPI application factory for the Betty daemon."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from betty import __version__

from .dashboard import router as dashboard_router
from .hooks import router as hooks_router
from .models import HealthResponse

DEFAULT_PORT = 7832

_PKG_DIR = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _PKG_DIR / "templates"
_STATIC_DIR = _PKG_DIR / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Betty", version=__version__,
                  description="Peer programming agent for Claude Code")

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
