"""Configuration management for Betty.

Priority (highest to lowest):
1. Environment variables (BETTY_*)
2. Config file (~/.betty/config.toml)
3. Hardcoded defaults
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

logger = logging.getLogger(__name__)

# Paths
BETTY_DIR = Path.home() / ".betty"
CONFIG_FILE = BETTY_DIR / "config.toml"
DB_FILE = BETTY_DIR / "betty.db"


@dataclass
class LLMConfig:
    """LLM provider configuration.

    The model string uses litellm conventions:
    - "openai/gpt-4o-mini" -> OpenAI API
    - "anthropic/claude-sonnet-4-20250514" -> Anthropic API
    - "openrouter/openai/gpt-4o-mini" -> OpenRouter
    - "ollama/qwen2.5:7b" -> Ollama
    """

    model: str = "anthropic/claude-sonnet-4-20250514"
    api_base: str | None = None
    api_key: str | None = None


@dataclass
class DelegationConfig:
    """Controls how much autonomy Betty has.

    Autonomy levels:
    0 - Observer only: Betty watches but never acts
    1 - Suggest: Betty suggests actions, user must approve all
    2 - Semi-auto: Betty auto-approves read-only tools, asks for writes
    3 - Full auto: Betty acts autonomously (within policy constraints)
    """

    autonomy_level: int = 1
    auto_approve_read_tools: bool = True
    confidence_threshold: float = 0.8


@dataclass
class EscalationConfig:
    """Configuration for reaching the user when they're away."""

    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    escalation_mode: str = "queue"  # "queue", "telegram", "both"


@dataclass
class PolicyConfig:
    """Configuration for organizational policies."""

    policy_dirs: list[str] = field(default_factory=list)
    strict_mode: bool = False


@dataclass
class Config:
    """Top-level Betty configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""
    # LLM overrides
    if val := os.getenv("BETTY_LLM_MODEL"):
        config.llm.model = val
    if val := os.getenv("BETTY_LLM_API_BASE"):
        config.llm.api_base = val
    if val := os.getenv("BETTY_LLM_API_KEY"):
        config.llm.api_key = val

    # Delegation overrides
    if val := os.getenv("BETTY_DELEGATION_LEVEL"):
        try:
            level = int(val)
            if 0 <= level <= 3:
                config.delegation.autonomy_level = level
        except ValueError:
            logger.warning("Invalid BETTY_DELEGATION_LEVEL: %s", val)
    if val := os.getenv("BETTY_AUTO_APPROVE_READ"):
        config.delegation.auto_approve_read_tools = val.lower() in ("true", "1", "yes")
    if val := os.getenv("BETTY_CONFIDENCE_THRESHOLD"):
        try:
            config.delegation.confidence_threshold = float(val)
        except ValueError:
            logger.warning("Invalid BETTY_CONFIDENCE_THRESHOLD: %s", val)

    # Escalation overrides
    if val := os.getenv("BETTY_TELEGRAM_TOKEN"):
        config.escalation.telegram_token = val
    if val := os.getenv("BETTY_TELEGRAM_CHAT_ID"):
        config.escalation.telegram_chat_id = val
    if val := os.getenv("BETTY_ESCALATION_MODE"):
        if val in ("queue", "telegram", "both"):
            config.escalation.escalation_mode = val

    return config


def _config_from_dict(data: dict[str, Any]) -> Config:
    """Build a Config from a parsed TOML dict."""
    llm_data = data.get("llm", {})
    llm = LLMConfig(
        model=llm_data.get("model", LLMConfig.model),
        api_base=llm_data.get("api_base"),
        api_key=llm_data.get("api_key"),
    )

    del_data = data.get("delegation", {})
    delegation = DelegationConfig(
        autonomy_level=del_data.get("autonomy_level", DelegationConfig.autonomy_level),
        auto_approve_read_tools=del_data.get(
            "auto_approve_read_tools", DelegationConfig.auto_approve_read_tools
        ),
        confidence_threshold=del_data.get(
            "confidence_threshold", DelegationConfig.confidence_threshold
        ),
    )

    esc_data = data.get("escalation", {})
    escalation = EscalationConfig(
        telegram_token=esc_data.get("telegram_token"),
        telegram_chat_id=esc_data.get("telegram_chat_id"),
        escalation_mode=esc_data.get("escalation_mode", EscalationConfig.escalation_mode),
    )

    pol_data = data.get("policy", {})
    policy = PolicyConfig(
        policy_dirs=pol_data.get("policy_dirs", []),
        strict_mode=pol_data.get("strict_mode", False),
    )

    return Config(llm=llm, delegation=delegation, escalation=escalation, policy=policy)


def _config_to_dict(config: Config) -> dict[str, Any]:
    """Serialize a Config to a dict suitable for TOML output.

    Never saves secrets (api_key, telegram_token) to file.
    """
    data: dict[str, Any] = {
        "llm": {"model": config.llm.model},
        "delegation": {
            "autonomy_level": config.delegation.autonomy_level,
            "auto_approve_read_tools": config.delegation.auto_approve_read_tools,
            "confidence_threshold": config.delegation.confidence_threshold,
        },
        "escalation": {
            "escalation_mode": config.escalation.escalation_mode,
        },
    }

    if config.llm.api_base:
        data["llm"]["api_base"] = config.llm.api_base

    if config.escalation.telegram_chat_id:
        data["escalation"]["telegram_chat_id"] = config.escalation.telegram_chat_id

    if config.policy.policy_dirs:
        data["policy"] = {
            "policy_dirs": config.policy.policy_dirs,
            "strict_mode": config.policy.strict_mode,
        }

    return data


def load_config() -> Config:
    """Load configuration with layered overrides.

    Priority: env vars > config file > defaults.
    """
    config = Config()

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                data = tomllib.load(f)
            config = _config_from_dict(data)
        except Exception:
            logger.warning("Failed to load config from %s, using defaults", CONFIG_FILE)

    config = _apply_env_overrides(config)
    return config


def save_config(config: Config) -> None:
    """Save configuration to ~/.betty/config.toml.

    Secrets (api_key, telegram_token) are never written to disk.
    """
    BETTY_DIR.mkdir(parents=True, exist_ok=True)
    data = _config_to_dict(config)
    with open(CONFIG_FILE, "wb") as f:
        tomli_w.dump(data, f)


def get_llm_presets() -> dict[str, dict[str, str]]:
    """Return common LLM presets for quick config."""
    return {
        "anthropic-sonnet": {
            "model": "anthropic/claude-sonnet-4-20250514",
            "description": "Anthropic Claude Sonnet (requires ANTHROPIC_API_KEY)",
        },
        "anthropic-haiku": {
            "model": "anthropic/claude-haiku-4-5-20251001",
            "description": "Anthropic Claude Haiku (requires ANTHROPIC_API_KEY)",
        },
        "openai-gpt4o": {
            "model": "openai/gpt-4o",
            "description": "OpenAI GPT-4o (requires OPENAI_API_KEY)",
        },
        "openai-gpt4o-mini": {
            "model": "openai/gpt-4o-mini",
            "description": "OpenAI GPT-4o Mini (requires OPENAI_API_KEY)",
        },
        "openrouter": {
            "model": "openrouter/anthropic/claude-sonnet-4-20250514",
            "description": "OpenRouter (requires OPENROUTER_API_KEY)",
        },
        "ollama": {
            "model": "ollama/llama3.1:8b",
            "description": "Ollama local (no API key needed)",
        },
    }
