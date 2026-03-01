"""Tests for betty.llm module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from betty.config import LLMConfig
from betty.llm import LLMService, TokenUsage


class TestTokenUsage:
    def test_record(self):
        usage = TokenUsage()
        usage.record({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.call_count == 1

    def test_accumulates(self):
        usage = TokenUsage()
        usage.record({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        usage.record({"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300})
        assert usage.prompt_tokens == 300
        assert usage.total_tokens == 450
        assert usage.call_count == 2


class TestLLMService:
    def test_init(self):
        config = LLMConfig(model="test/model")
        service = LLMService(config)
        assert service.config.model == "test/model"
        assert service.usage.call_count == 0

    def test_call_kwargs(self):
        config = LLMConfig(model="openai/gpt-4o", api_base="http://localhost:8080", api_key="sk-test")
        service = LLMService(config)
        kwargs = service._call_kwargs()
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["api_base"] == "http://localhost:8080"
        assert kwargs["api_key"] == "sk-test"

    def test_call_kwargs_minimal(self):
        config = LLMConfig(model="anthropic/claude-sonnet-4-20250514")
        service = LLMService(config)
        kwargs = service._call_kwargs()
        assert kwargs == {"model": "anthropic/claude-sonnet-4-20250514"}

    @pytest.mark.asyncio
    async def test_complete(self):
        config = LLMConfig(model="test/model")
        service = LLMService(config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        with patch("betty.llm.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            result = await service.complete("Say hello")
            assert result == "Hello, world!"
            assert service.usage.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_json(self):
        config = LLMConfig(model="test/model")
        service = LLMService(config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"answer": 42}'
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        with patch("betty.llm.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            result = await service.complete_json("What is the answer?")
            assert result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_complete_json_markdown_fallback(self):
        config = LLMConfig(model="test/model")
        service = LLMService(config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Here:\n```json\n{"answer": 42}\n```'
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        with patch("betty.llm.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            result = await service.complete_json("What is the answer?")
            assert result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_complete_json_invalid(self):
        config = LLMConfig(model="test/model")
        service = LLMService(config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not JSON"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        with patch("betty.llm.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            with pytest.raises(ValueError, match="Could not parse"):
                await service.complete_json("What is the answer?")

    def test_get_usage(self):
        config = LLMConfig(model="test/model")
        service = LLMService(config)
        service.usage.record({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        usage = service.get_usage()
        assert usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "call_count": 1}
