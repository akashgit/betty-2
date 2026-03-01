# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Betty 2.0 is a peer programming agent for Claude Code. It integrates via Claude Code's hooks system (UserPromptSubmit, PreToolUse, PostToolUse) to amplify user intent, learn preferences, and auto-pilot tool approvals.

## Development Commands

```bash
# Install in dev mode
uv sync --dev

# Run the CLI
uv run betty --help
uv run betty status
uv run betty config show

# Run tests
uv run pytest                    # All tests
uv run pytest tests/test_server.py -v   # Server tests only
uv run pytest -x                 # Stop on first failure

# Start the FastAPI server directly (dev mode)
uv run uvicorn betty.server.app:create_app --factory --reload --port 7832

# Type check (if mypy is installed)
uv run mypy src/betty/
```

## Architecture

### Data Flow

```
User prompt -> Claude Code -> UserPromptSubmit hook
                                  |
                                  v
                          Betty daemon (FastAPI on :7832)
                                  |
                     +------------+------------+
                     |            |            |
              Intent Engine  Approval Model  Session Analyzer
                     |            |            |
                     v            v            v
              Clarifying    allow/block    Pattern extraction
              questions     decision       -> User Model DB
                                  |
                          Escalation Router
                          (TUI | Telegram | Queue)
```

### Source Tree

```
src/betty/
  __init__.py          # Package init, version
  _version.py          # Auto-generated version from VCS
  cli.py               # Click CLI: start/stop/status, config, hooks, profile, sessions, policies
  config.py            # TOML config with env var overrides (LLM, delegation, escalation, policy)
  db.py                # SQLite database via aiosqlite (user_preferences, sessions, approvals, policies)
  escalation.py        # Escalation router: routes questions to TUI, Telegram, or queue
  hook_handler.py      # Bridge: reads stdin JSON from Claude Code, POSTs to daemon, prints response
  hooks.py             # Install/uninstall Betty hook entries in ~/.claude/settings.json
  llm.py               # Unified LLM service via litellm (complete, complete_json, embed)
  models.py            # Core dataclasses: Session, Turn, ToolCall, ToolApproval
  session_reader.py    # Parse Claude Code JSONL transcripts, discover sessions, SessionWatcher
  telegram_bot.py      # Async Telegram bot for remote escalation with inline keyboards
  server/
    __init__.py
    app.py             # FastAPI app factory (create_app), static files, health endpoint
    models.py          # Pydantic request/response schemas for API
    hooks.py           # Hook endpoints: /hooks/prompt-submit, /hooks/pre-tool-use, /hooks/post-tool-use
    dashboard.py       # Dashboard pages (Jinja2), JSON API (/api/*), HTMX partials (/partials/*)
  templates/           # Jinja2 templates (PicoCSS + HTMX)
    base.html          # Layout with nav
    home.html          # Dashboard overview with stats
    user_model.html    # Preferences by category
    sessions.html      # Session history list
    session_detail.html # Single session detail
    policies.html      # Policy CRUD
    approvals.html     # Approval pattern management
    settings.html      # Configuration UI
    partials/          # HTMX fragments for in-page updates
  static/
    style.css          # PicoCSS overrides
tests/
  conftest.py          # Shared fixtures (tmp dirs, mock sessions, etc.)
  test_cli.py          # CLI command tests
  test_config.py       # Config loading/saving tests
  test_db.py           # Database CRUD tests
  test_escalation.py   # Escalation router tests
  test_hook_handler.py # Hook handler tests
  test_hooks.py        # Hook install/uninstall tests
  test_llm.py          # LLM service tests
  test_models.py       # Data model tests
  test_server.py       # FastAPI endpoint tests (httpx + ASGITransport)
  test_session_reader.py # Session parser tests
  test_telegram_bot.py # Telegram bot tests
```

### Key Files and Their Roles

| File | Role |
|------|------|
| `cli.py` | All `betty` commands. Click groups for config, hooks, policies, telegram. Hidden `hook-handler` command called by hook scripts. |
| `config.py` | Layered config: env vars > `~/.betty/config.toml` > defaults. Dataclasses for LLM, delegation, escalation, policy. Never saves secrets to disk. |
| `db.py` | `UserModelDB` class with async CRUD for preferences, sessions, approval patterns, policies, escalation log. WAL mode, schema migrations via version table. |
| `escalation.py` | `EscalationRouter` picks channel (TUI/Telegram/Queue) based on urgency and user availability. `Question`/`UserResponse` dataclasses. Timeout handling with defaults. |
| `hook_handler.py` | Synchronous bridge callable from CLI. Reads stdin, POSTs to `http://localhost:7832/hooks/<type>`, prints response. 2s timeout. Fails silently. |
| `hooks.py` | `install_hooks()`/`uninstall_hooks()` manage entries in `~/.claude/settings.json`. Non-destructive. Uses `# betty-hook` marker for identification. |
| `llm.py` | `LLMService` class wrapping litellm. Methods: `complete()`, `complete_json()`, `embed()`. Tracks cumulative token usage. |
| `models.py` | `Session` (parsed from JSONL), `Turn` (user/assistant message), `ToolCall` (single tool invocation), `ToolApproval` enum. |
| `session_reader.py` | `parse_session()` reads a JSONL transcript into a `Session`. `discover_sessions()` finds transcripts. `SessionWatcher` polls active sessions for new turns via async generator. |
| `telegram_bot.py` | `TelegramBot` class using python-telegram-bot. `/start` for linking, inline keyboards for options, free-form text for answers. Rate limited. `make_telegram_sender()` creates an escalation-compatible callable. |
| `server/app.py` | `create_app()` factory. Mounts static files, configures Jinja2 templates, registers hook and dashboard routers. |
| `server/hooks.py` | Fast-path endpoints (<100ms target). Currently pass-through stubs; will wire to intent engine and approval model. |
| `server/dashboard.py` | HTML page routes, JSON API, HTMX partials. Data access via `_get_*` helper stubs (to be wired to `UserModelDB`). |
| `server/models.py` | Pydantic schemas: `HookPromptSubmitRequest/Response`, `HookPreToolUseRequest/Response`, `UserProfile`, `SessionSummary`, `PolicyItem`, `ConfigResponse`, etc. |

### Database Schema (SQLite)

Tables in `~/.betty/betty.db`:
- `user_preferences` -- category, key, value, confidence, evidence_count, project_scope
- `session_summaries` -- session_id, project_dir, goal, outcome, tools/files/decisions/patterns (JSON)
- `approval_patterns` -- tool_name, action_pattern, decision, count, project_scope
- `org_policies` -- policy_type, rule, description, source, project_scope
- `escalation_log` -- session_id, question, response, channel, response_time
- `schema_version` -- migration tracking

### Configuration Hierarchy

```
Environment variables (BETTY_*)
    | overrides
~/.betty/config.toml
    | overrides
Hardcoded defaults (config.py dataclass defaults)
```

## Design Principles

- **Asyncio-first**: All I/O (database, LLM, HTTP) is async. The daemon runs on a single event loop.
- **Graceful degradation**: Betty works without an LLM configured (heuristic-only mode). The dashboard works without the database (shows empty states). Individual components fail independently.
- **Local-only**: All data stays on the user's machine. No telemetry, no cloud services except the configured LLM provider.
- **Hook latency**: PreToolUse hook responses must be <100ms to avoid blocking Claude Code. Expensive operations (LLM calls, session analysis) happen asynchronously after the response.
- **Separation of concerns**: Hooks router (fast path) is separate from dashboard router (standard web). Pydantic models validate all API boundaries. Data access is behind helper functions for easy DB wiring.
- **Zero JS build step**: Dashboard uses HTMX + Jinja2 for server-rendered interactivity. PicoCSS for styling. No webpack, no npm, no node_modules.
- **Silent failure**: Hook handler and daemon communication never block Claude Code. If the daemon is down, hooks exit silently.

## Testing

Tests use pytest + pytest-asyncio + httpx (for FastAPI test client):

```bash
uv run pytest                           # Run all
uv run pytest tests/test_server.py -v   # Server/dashboard tests
uv run pytest tests/test_db.py -v       # Database tests
uv run pytest tests/test_config.py -v   # Config tests
uv run pytest tests/test_llm.py -v      # LLM service tests
uv run pytest tests/test_escalation.py -v  # Escalation router tests
uv run pytest tests/test_session_reader.py -v  # Session reader tests
uv run pytest tests/test_telegram_bot.py -v    # Telegram bot tests
```

The server test suite uses `httpx.AsyncClient` with `ASGITransport` to test FastAPI endpoints without starting a real server.
