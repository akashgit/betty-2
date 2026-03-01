"""Tests for betty.config module."""

import pytest

from betty.config import (
    Config,
    DelegationConfig,
    LLMConfig,
    EscalationConfig,
    _apply_env_overrides,
    _config_from_dict,
    _config_to_dict,
    load_config,
    save_config,
)


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.model == "claude-code/haiku"
        assert cfg.api_base is None

    def test_custom_values(self):
        cfg = LLMConfig(model="openai/gpt-4o", api_base="http://localhost:8080")
        assert cfg.model == "openai/gpt-4o"


class TestDelegationConfig:
    def test_defaults(self):
        cfg = DelegationConfig()
        assert cfg.autonomy_level == 1
        assert cfg.auto_approve_read_tools is True
        assert cfg.confidence_threshold == 0.8


class TestConfigFromDict:
    def test_empty_dict(self):
        cfg = _config_from_dict({})
        assert cfg.llm.model == LLMConfig.model

    def test_full_dict(self):
        data = {
            "llm": {"model": "ollama/llama3"},
            "delegation": {"autonomy_level": 2, "confidence_threshold": 0.9},
            "escalation": {"escalation_mode": "telegram"},
            "policy": {"policy_dirs": ["/etc/betty/policies"], "strict_mode": True},
        }
        cfg = _config_from_dict(data)
        assert cfg.llm.model == "ollama/llama3"
        assert cfg.delegation.autonomy_level == 2
        assert cfg.policy.strict_mode is True


class TestConfigToDict:
    def test_round_trip(self):
        cfg = Config(llm=LLMConfig(model="test/model", api_base="http://test"))
        data = _config_to_dict(cfg)
        assert data["llm"]["model"] == "test/model"
        assert data["llm"]["api_base"] == "http://test"

    def test_no_secrets_in_output(self):
        cfg = Config(
            llm=LLMConfig(api_key="secret-key"),
            escalation=EscalationConfig(telegram_token="secret-token"),
        )
        data = _config_to_dict(cfg)
        assert "api_key" not in data.get("llm", {})
        assert "telegram_token" not in data.get("escalation", {})


class TestEnvOverrides:
    def test_llm_model_override(self, monkeypatch):
        monkeypatch.setenv("BETTY_LLM_MODEL", "openai/gpt-4o-mini")
        cfg = Config()
        cfg = _apply_env_overrides(cfg)
        assert cfg.llm.model == "openai/gpt-4o-mini"

    def test_delegation_level_override(self, monkeypatch):
        monkeypatch.setenv("BETTY_DELEGATION_LEVEL", "3")
        cfg = Config()
        cfg = _apply_env_overrides(cfg)
        assert cfg.delegation.autonomy_level == 3

    def test_invalid_delegation_level(self, monkeypatch):
        monkeypatch.setenv("BETTY_DELEGATION_LEVEL", "5")
        cfg = Config()
        cfg = _apply_env_overrides(cfg)
        assert cfg.delegation.autonomy_level == 1

    def test_telegram_token_override(self, monkeypatch):
        monkeypatch.setenv("BETTY_TELEGRAM_TOKEN", "bot123:ABC")
        cfg = Config()
        cfg = _apply_env_overrides(cfg)
        assert cfg.escalation.telegram_token == "bot123:ABC"

    def test_invalid_escalation_mode_ignored(self, monkeypatch):
        monkeypatch.setenv("BETTY_ESCALATION_MODE", "invalid")
        cfg = Config()
        cfg = _apply_env_overrides(cfg)
        assert cfg.escalation.escalation_mode == "queue"


class TestSaveLoadConfig:
    def test_save_and_load(self, tmp_path, monkeypatch):
        import betty.config as config_mod

        config_dir = tmp_path / ".betty"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        monkeypatch.setattr(config_mod, "BETTY_DIR", config_dir)
        monkeypatch.setattr(config_mod, "CONFIG_FILE", config_file)
        for var in ("BETTY_LLM_MODEL", "BETTY_LLM_API_BASE", "BETTY_DELEGATION_LEVEL"):
            monkeypatch.delenv(var, raising=False)

        cfg = Config(llm=LLMConfig(model="test/save-load"), delegation=DelegationConfig(autonomy_level=2))
        save_config(cfg)
        assert config_file.exists()
        loaded = load_config()
        assert loaded.llm.model == "test/save-load"
        assert loaded.delegation.autonomy_level == 2
