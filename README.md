# Betty 2.0

A peer programming agent for Claude Code.

Betty integrates with Claude Code via its hooks system to learn your preferences, auto-pilot tool approvals, and surface clarifying questions before work begins. It runs as a local daemon alongside Claude Code and stores all data on your machine.

## Core Capabilities

- **Intent Amplification** -- Intercept user prompts via `UserPromptSubmit` hooks to detect ambiguity and surface clarifying questions before Claude Code acts.
- **Approval Auto-Pilot** -- Learn tool approval patterns from your history. Auto-approve routine operations and flag anomalies via `PreToolUse` hooks.
- **Session Memory** -- Parse Claude Code JSONL transcripts, extract patterns, and build a persistent user model so context carries across sessions.

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/akashgit/betty-2.git
cd betty-2
uv sync --dev

# Verify
uv run betty --version
uv run betty status
```

## Quick Start

```bash
# 1. Install hooks into Claude Code
uv run betty hooks install

# 2. (Optional) Configure the LLM backend
uv run betty config llm-preset anthropic-sonnet
# Or set via environment:
export BETTY_LLM_MODEL="anthropic/claude-sonnet-4-20250514"
export ANTHROPIC_API_KEY="sk-..."

# 3. Start the daemon
uv run betty start

# 4. Open the dashboard
open http://localhost:7832
```

## Architecture

```
Claude Code                         Betty Daemon (:7832)
+-------------------+              +-----------------------+
| User types prompt |              |  FastAPI (uvicorn)    |
|        |          |  stdin JSON  |                       |
|  UserPromptSubmit +------------->+  /hooks/prompt-submit |
|                   |              |  /hooks/pre-tool-use  |
|  PreToolUse       +------------->+  /hooks/post-tool-use |
|                   |              |         |             |
|  PostToolUse      +------------->+    +----+----+        |
+-------------------+              |    |         |        |
                                   | Intent   Approval    |
                                   | Engine   Model       |
                                   |    |         |        |
                                   |    +----+----+        |
                                   |         |             |
                                   |   User Model DB      |
                                   |   (~/.betty/betty.db) |
                                   +-----------+-----------+
                                               |
                                   +-----------+-----------+
                                   | Escalation Router     |
                                   |   TUI | Telegram | Q  |
                                   +-----------------------+
```

## Key Components

| Component | File(s) | Description |
|-----------|---------|-------------|
| CLI | `cli.py` | Click commands: `start`, `stop`, `status`, `config`, `hooks`, `profile`, `sessions`, `policies`, `telegram`, `logs` |
| Config | `config.py` | TOML config at `~/.betty/config.toml` with env var overrides (`BETTY_*`). Never saves secrets to disk. |
| Database | `db.py` | SQLite via aiosqlite. Tables: `user_preferences`, `session_summaries`, `approval_patterns`, `org_policies`, `escalation_log`. WAL mode. |
| LLM Service | `llm.py` | Unified async LLM calls via litellm. Methods: `complete()`, `complete_json()`, `embed()`. Supports all litellm providers. |
| Session Reader | `session_reader.py` | Parse Claude Code JSONL transcripts from `~/.claude/projects/`. Includes `SessionWatcher` for live polling of active sessions. |
| Hooks | `hooks.py` | Install/uninstall Betty entries in `~/.claude/settings.json`. Non-destructive -- preserves existing hooks. |
| Hook Handler | `hook_handler.py` | Bridge between Claude Code hooks and the daemon. Reads stdin JSON, POSTs to daemon, prints response. Fails silently if daemon is down. |
| Escalation | `escalation.py` | Routes questions to TUI, Telegram, or queue based on urgency and user availability. |
| Telegram Bot | `telegram_bot.py` | Async Telegram bot for remote escalation. Sends questions with inline keyboards, receives replies. |
| FastAPI Server | `server/app.py` | App factory (`create_app()`). Health endpoint, static files, Jinja2 templates. |
| Hook Endpoints | `server/hooks.py` | Fast-path hook API: `/hooks/prompt-submit`, `/hooks/pre-tool-use`, `/hooks/post-tool-use` |
| Dashboard | `server/dashboard.py` | HTML pages (Jinja2 + HTMX), JSON API (`/api/*`), HTMX partials (`/partials/*`) |
| API Models | `server/models.py` | Pydantic request/response schemas for all API boundaries |
| Data Models | `models.py` | Core dataclasses: `Session`, `Turn`, `ToolCall`, `ToolApproval` |

## Configuration

Betty uses a layered configuration system:

```
Environment variables (BETTY_*)     <-- highest priority
         |
~/.betty/config.toml
         |
Hardcoded defaults                  <-- lowest priority
```

### Config File

```toml
# ~/.betty/config.toml

[llm]
model = "anthropic/claude-sonnet-4-20250514"
# api_base = "http://localhost:11434"   # for Ollama

[delegation]
autonomy_level = 1          # 0=observe, 1=suggest, 2=semi-auto, 3=full-auto
auto_approve_read_tools = true
confidence_threshold = 0.8

[escalation]
escalation_mode = "queue"   # "queue", "telegram", "both"
# telegram_chat_id = "12345"

[policy]
# policy_dirs = ["./policies"]
# strict_mode = false
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BETTY_LLM_MODEL` | LLM model string (litellm format) |
| `BETTY_LLM_API_BASE` | Custom API base URL |
| `BETTY_LLM_API_KEY` | API key for the LLM provider |
| `BETTY_DELEGATION_LEVEL` | Autonomy level (0-3) |
| `BETTY_AUTO_APPROVE_READ` | Auto-approve read-only tools (true/false) |
| `BETTY_CONFIDENCE_THRESHOLD` | Confidence threshold for auto-approval |
| `BETTY_TELEGRAM_TOKEN` | Telegram bot token |
| `BETTY_TELEGRAM_CHAT_ID` | Telegram chat ID for escalation |
| `BETTY_ESCALATION_MODE` | Escalation mode (queue/telegram/both) |

## CLI Commands

```
betty --version              Show version
betty status                 Show daemon status
betty start                  Start the daemon
betty stop                   Stop the daemon

betty config show            Display current configuration
betty config llm-preset      List or apply LLM presets
betty config delegation N    Set autonomy level (0-3)
betty config telegram-token  Set Telegram bot token

betty hooks install          Install Betty hooks into Claude Code
betty hooks uninstall        Remove Betty hooks
betty hooks status           Show hook installation status

betty profile                View learned user profile
betty sessions               List recent sessions
betty policies list          List active policies
betty policies add           Add a policy rule
betty policies import        Import policies from file

betty telegram link          Link a Telegram chat
betty telegram test          Send a test message
betty logs                   View daemon logs
betty dashboard              Open the web dashboard
```

## Data Storage

All data is stored locally:

```
~/.betty/
  config.toml        # User configuration
  betty.db           # SQLite database (preferences, sessions, approvals, policies)

~/.claude/
  settings.json      # Claude Code settings (hooks are installed here)
  projects/          # Session transcripts (JSONL, read-only by Betty)
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest
uv run pytest -v              # verbose
uv run pytest -x              # stop on first failure

# Start dev server
uv run uvicorn betty.server.app:create_app --factory --reload --port 7832

# Run CLI
uv run betty --help
```

## License

Apache 2.0
