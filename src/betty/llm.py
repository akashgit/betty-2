"""Unified LLM service for Betty.

Three-way routing based on model string:
- "claude-code/*" → subprocess call to `claude` CLI (no API key needed)
- api_base set    → OpenAI SDK direct (local/custom servers)
- otherwise       → litellm (cloud providers with auto-routing)
"""

import asyncio
import json
import logging
import subprocess
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
    """Async LLM service with three-way routing.

    Usage:
        # Default: uses claude CLI subprocess (no API key needed)
        llm = LLMService(LLMConfig(model="claude-code/haiku"))

        # Cloud provider via litellm
        llm = LLMService(LLMConfig(model="anthropic/claude-sonnet-4-20250514"))

        # Local server via OpenAI SDK
        llm = LLMService(LLMConfig(model="qwen2.5:7b", api_base="http://localhost:1234/v1"))

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

    def _call_claude_code(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Call claude CLI in single-prompt mode.

        Prompt is piped via stdin to avoid OS ARG_MAX limits on long prompts.
        """
        claude_model = self.config.model.split("/", 1)[1] if "/" in self.config.model else "haiku"

        cmd = [
            "claude", "-p",
            "--no-session-persistence",
            "--model", claude_model,
        ]
        if system:
            cmd.extend(["--system-prompt", system])

        # Strip env vars that trigger nested-session detection, but
        # keep auth/routing vars (e.g. CLAUDE_CODE_USE_VERTEX) so the
        # child process can authenticate.
        _STRIP_VARS = {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"}
        env = {
            k: v for k, v in __import__("os").environ.items()
            if k not in _STRIP_VARS
        }

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if result.returncode != 0:
            error_detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                error_detail or f"claude exited with code {result.returncode}"
            )
        return result.stdout.strip()

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text completion.

        Routes: claude-code/* → subprocess, api_base → openai SDK, else → litellm.
        """
        # Route 1: Claude Code subprocess
        if self.config.is_claude_code:
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None, self._call_claude_code, prompt, system, max_tokens,
            )
            self.usage.call_count += 1
            return content

        # Route 2 & 3: OpenAI SDK (local) or litellm (cloud)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.config.api_base:
            # Route 2: Local/custom OpenAI-compatible server
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url=self.config.api_base.rstrip("/"),
                api_key=self.config.api_key or "no-key-required",
            )
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            if hasattr(response, "usage") and response.usage:
                self.usage.record({
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                })
            return content

        # Route 3: Cloud provider via litellm
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

        Note: embedding is not supported for claude-code subprocess.
        Use a cloud embedding model instead.
        """
        if self.config.is_claude_code:
            raise NotImplementedError("Embedding not supported for claude-code subprocess")

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
