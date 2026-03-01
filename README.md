<p align="center">
  <img src="src/betty/static/logo.png" alt="Betty" width="100">
</p>

<h1 align="center">Betty</h1>

<p align="center">
  A peer programming agent for <a href="https://docs.anthropic.com/en/docs/claude-code">Claude Code</a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#what-betty-does">Features</a> &middot;
  <a href="#configuration">Configure</a> &middot;
  <a href="#dashboard">Dashboard</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#development">Development</a>
</p>

---

Betty runs as a local daemon alongside Claude Code and integrates via the [hooks system](https://docs.anthropic.com/en/docs/claude-code/hooks) to act as a peer programmer. It intercepts your prompts, asks clarifying questions before Claude starts working, learns your preferences over time, and auto-approves routine tool calls. All data stays on your machine.

## What Betty Does

**Intent Amplification** — When you type a prompt like *"add user authentication"*, Betty intercepts it and uses an LLM to generate clarifying questions (*"JWT or session-based? Should I add tests?"*). The questions are presented interactively so Claude builds the right thing the first time.

**Approval Auto-Pilot** — Betty watches which tool calls you approve and learns patterns. After seeing you approve `Read` on `*.py` files enough times, it starts auto-approving them so you don't have to click "Allow" repeatedly.

**Session Memory** — Betty parses Claude Code transcripts after each session, extracts patterns (coding style, preferred tools, common decisions), and builds a persistent user model that carries context across sessions.

## Quick Start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and install
git clone https://github.com/akashgit/betty-2.git
cd betty-2
uv sync

# Install hooks into Claude Code
betty hooks install

# Start the daemon
betty start

# Verify
betty status
```

Open the dashboard at http://localhost:7832 to configure and monitor Betty.

## Configuration

Betty stores config at `~/.betty/config.toml`. You can edit it directly, use the CLI, or use the dashboard at http://localhost:7832/settings.

### LLM Provider

Betty uses a small LLM to generate clarifying questions. Pick a provider:

```toml
[llm]
# Vertex AI (recommended if you have gcloud auth)
model = "vertex_ai/claude-3-5-haiku@20241022"

# Anthropic API (requires ANTHROPIC_API_KEY)
model = "anthropic/claude-haiku-4-5-20251001"

# OpenAI (requires OPENAI_API_KEY)
model = "openai/gpt-4o-mini"

# Claude Code subprocess (no API key needed, but slow ~30s)
model = "claude-code/haiku"

# Local via Ollama (no API key needed)
model = "ollama/llama3.1:8b"
```

Betty uses [litellm](https://github.com/BerriAI/litellm) for provider routing. Set the appropriate API key as an environment variable.

### Autonomy Level

Controls how much Betty does on its own:

```toml
[delegation]
autonomy_level = 1          # 0=observe, 1=suggest, 2=semi-auto, 3=full-auto
auto_approve_read_tools = true
confidence_threshold = 0.8
```

| Level | Behavior |
|-------|----------|
| 0 | Observer only — Betty watches but never acts |
| 1 | Suggest — Betty suggests actions, you approve all |
| 2 | Semi-auto — Betty auto-approves read-only tools, asks for writes |
| 3 | Full auto — Betty acts autonomously within policy constraints |

### Environment Variable Overrides

All settings can be overridden via `BETTY_*` environment variables:

| Variable | Description |
|----------|-------------|
| `BETTY_LLM_MODEL` | LLM model string (litellm format) |
| `BETTY_LLM_API_BASE` | Custom API base URL |
| `BETTY_LLM_API_KEY` | API key (never saved to disk) |
| `BETTY_DELEGATION_LEVEL` | Autonomy level (0-3) |
| `BETTY_CONFIDENCE_THRESHOLD` | Min confidence to auto-approve |
| `BETTY_TELEGRAM_TOKEN` | Telegram bot token for remote escalation |
| `BETTY_ESCALATION_MODE` | `queue`, `telegram`, or `both` |

## CLI Reference

```bash
# Daemon
betty start                  # Start the daemon
betty stop                   # Stop the daemon
betty status                 # Check status

# Configuration
betty config show            # Show current config
betty config llm-preset      # List or apply LLM presets
betty config delegation N    # Set autonomy level (0-3)

# Hooks
betty hooks install          # Install hooks into Claude Code
betty hooks uninstall        # Remove hooks
betty hooks status           # Show hook status

# Data
betty profile                # View learned preferences
betty sessions               # List recent sessions
betty policies list          # List active policies
betty policies add           # Add a policy rule

# Integrations
betty telegram link          # Link a Telegram chat
betty logs                   # View daemon logs
betty dashboard              # Open the web dashboard
```

## Dashboard

Betty includes a web dashboard at http://localhost:7832 with pages for:

- **Home** — Overview stats (preferences, sessions, approvals)
- **Preferences** — View and edit learned coding preferences
- **Sessions** — Browse past Claude Code session summaries
- **Policies** — Define org/project rules Betty should enforce
- **Approvals** — See and manage tool approval patterns
- **Settings** — Configure LLM, autonomy level, and integrations

---

## Architecture

```
User prompt ──> Claude Code ──> UserPromptSubmit hook
                                      |
                                      v
                              Betty daemon (:7832)
                                      |
                        +─────────────+─────────────+
                        |             |             |
                  Intent Engine  Approval Model  Session Analyzer
                        |             |             |
                        v             v             v
                  Clarifying     allow/block    Pattern extraction
                  questions      decision       -> User Model DB
```

The daemon runs as a single async process (FastAPI + uvicorn) with background tasks for session watching and preference maintenance.

### Key Components

| Component | File | Role |
|-----------|------|------|
| Hook handler | `hook_handler.py` | Thin bridge: reads stdin from Claude Code, POSTs to daemon, prints response |
| Intent engine | `intent_engine.py` | Analyzes prompts, gathers context, generates clarifying questions via LLM |
| Approval model | `approval.py` | Predicts tool approval decisions based on learned patterns |
| Session analyzer | `session_analyzer.py` | Extracts patterns and preferences from completed sessions |
| User model | `user_model.py` | Stores and retrieves learned user preferences |
| LLM service | `llm.py` | Three-way routing: claude-code subprocess, OpenAI-compatible API, or litellm |
| Dashboard | `server/` | FastAPI + HTMX + Jinja2 web UI (zero JS build step) |
| Escalation | `escalation.py` | Routes questions to TUI, Telegram, or queue |

### Data Storage

All data stays local in `~/.betty/`:

```
~/.betty/
  config.toml      # Configuration (secrets never saved to disk)
  betty.db         # SQLite database (preferences, sessions, approvals, policies)
  betty.log        # Rotating daemon log (5MB max, 3 backups)
  betty.pid        # PID file for daemon management
```

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest                           # All tests (95+)
uv run pytest tests/test_server.py -v   # Specific module
uv run pytest -x                        # Stop on first failure

# Dev server with auto-reload
uv run uvicorn betty.server.app:create_app --factory --reload --port 7832

# Type check
uv run mypy src/betty/
```

### Source Layout

```
src/betty/
  cli.py               # Click CLI commands
  config.py            # TOML config with env var overrides
  db.py                # SQLite via aiosqlite (WAL mode)
  hook_handler.py      # Bridge between Claude Code hooks and daemon
  hooks.py             # Install/uninstall hooks in Claude Code settings
  intent_engine.py     # Prompt analysis and question generation
  llm.py               # LLM service with three-way routing
  approval.py          # Tool approval prediction model
  session_analyzer.py  # Session pattern extraction
  session_reader.py    # Parse Claude Code JSONL transcripts
  user_model.py        # User preference management
  escalation.py        # Escalation router (TUI/Telegram/queue)
  telegram_bot.py      # Telegram bot for remote escalation
  server/
    app.py             # FastAPI app factory
    hooks.py           # Hook endpoints
    dashboard.py       # Dashboard pages and API
    models.py          # Pydantic schemas
  templates/           # Jinja2 + HTMX templates (PicoCSS)
  static/              # CSS, logo
```

## License

Apache 2.0
