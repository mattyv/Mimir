"""Phase 5 — LLMClient tests (using injected fake client)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mimir.crawler.llm import LLMClient


def _fake_openai(response_text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = response_text
    completion = MagicMock()
    completion.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


@pytest.mark.phase5
def test_complete_returns_text() -> None:
    client = LLMClient(api_key="sk-test", client=_fake_openai('{"entities":[]}'))
    result = client.complete("extract entities")
    assert result == '{"entities":[]}'


@pytest.mark.phase5
def test_complete_passes_temperature() -> None:
    fake = _fake_openai("ok")
    client = LLMClient(api_key="sk-test", client=fake)
    client.complete("prompt", temperature=0.5)
    call_kwargs = fake.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.5


@pytest.mark.phase5
def test_complete_uses_configured_model() -> None:
    fake = _fake_openai("ok")
    client = LLMClient(api_key="sk-test", model="google/gemma-3-27b-it", client=fake)
    client.complete("prompt")
    call_kwargs = fake.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "google/gemma-3-27b-it"


@pytest.mark.phase5
def test_complete_none_content_returns_empty_string() -> None:
    choice = MagicMock()
    choice.message.content = None
    completion = MagicMock()
    completion.choices = [choice]
    fake = MagicMock()
    fake.chat.completions.create.return_value = completion
    client = LLMClient(api_key="sk-test", client=fake)
    assert client.complete("prompt") == ""
