"""Unified LLM service for Betty.

Centralizes all LLM calls through litellm. Supports all providers
that litellm handles (OpenAI, Anthropic, Ollama, OpenRouter, etc.).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import litellm

from betty.config import LLMConfig

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


@dataclass
class TokenUsage:
    """Tracks cumulative token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def record(self, usage: dict[str, int]) -> None:
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)
        self.call_count += 1


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class LLMService:
    """Async LLM service backed by litellm.

    Usage:
        config = LLMConfig(model="anthropic/claude-sonnet-4-20250514")
        llm = LLMService(config)
        response = await llm.complete("What is 2+2?")
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.usage = TokenUsage()

    def _call_kwargs(self) -> dict[str, Any]:
        """Build kwargs for litellm calls."""
        kwargs: dict[str, Any] = {"model": self.config.model}
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        return kwargs

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The user message.
            system: Optional system prompt.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.

        Returns:
            The generated text content.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = self._call_kwargs()
        kwargs["messages"] = messages
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens

        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content or ""

        if hasattr(response, "usage") and response.usage:
            self.usage.record(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )

        return content

    async def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate a JSON-structured completion.

        Instructs the model to return valid JSON. Parses the response
        and returns a dict. Falls back to extracting JSON from markdown
        code blocks if the raw response isn't valid JSON.

        Args:
            prompt: The user message.
            system: Optional system prompt (JSON instruction is appended).
            schema: Optional JSON schema hint included in the prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            Parsed JSON dict.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        json_instruction = "Respond with valid JSON only. No markdown, no explanation."
        if schema:
            json_instruction += f"\n\nExpected schema:\n{json.dumps(schema, indent=2)}"

        full_system = f"{system}\n\n{json_instruction}" if system else json_instruction

        raw = await self.complete(
            prompt=prompt,
            system=full_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```" in raw:
            for block in raw.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse LLM response as JSON: {raw[:200]}")

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for text.

        Uses litellm's embedding API. The model should support embeddings
        (e.g. "openai/text-embedding-3-small").

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        kwargs: dict[str, Any] = {"model": self.config.model}
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        response = await litellm.aembedding(input=[text], **kwargs)
        return response.data[0]["embedding"]

    def get_usage(self) -> dict[str, int]:
        """Get cumulative token usage stats."""
        return {
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "call_count": self.usage.call_count,
        }
