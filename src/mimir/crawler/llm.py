"""LLM client wrapper for OpenRouter / Gemma 4.

Uses the OpenAI-compatible API surface that OpenRouter exposes.  Prompt
caching is handled automatically by OpenRouter for supported models; no
application-level cache_control headers are required.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_MODEL = "google/gemma-3-27b-it"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    """Thin wrapper around the openai.OpenAI client pointed at OpenRouter.

    Args:
        api_key:   OpenRouter API key (``sk-or-…``).
        model:     Model ID to use for completions.
        base_url:  API base URL; override for testing.
        client:    Pre-constructed openai.OpenAI instance (for injection in tests).
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        client: Any = None,
    ) -> None:
        self._model = model
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Return the model's text response for a single *prompt* string."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return str(response.choices[0].message.content or "")
