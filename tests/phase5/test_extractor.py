"""Phase 5 — extractor tests using FakeLLM."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from mimir.adapters.base import Chunk
from mimir.crawler.extractor import ExtractionResult, extract
from tests.conftest import FakeLLM

_NOW = datetime(2026, 4, 19, tzinfo=UTC)


def _chunk(content: str) -> Chunk:
    return Chunk(
        id="test_chunk",
        source_type="confluence",
        content=content,
        retrieved_at=_NOW,
        reference="https://wiki.example.com/test",
    )


def _llm_with_response(data: dict) -> FakeLLM:
    llm = FakeLLM()
    from mimir.crawler.prompts import build_extraction_prompt
    prompt = build_extraction_prompt("OMMS is owned by the APAC team.")
    llm.set_response(prompt, json.dumps(data))
    return llm


@pytest.mark.phase5
def test_extract_returns_result_with_chunk_id() -> None:
    llm = FakeLLM()
    result = extract(_chunk("no entities here"), llm)
    assert isinstance(result, ExtractionResult)
    assert result.chunk_id == "test_chunk"


@pytest.mark.phase5
def test_extract_parses_entities() -> None:
    payload = {
        "entities": [
            {"name": "OMMS", "type": "auros:TradingService", "description": "Options market maker"}
        ]
    }
    llm = FakeLLM()
    chunk = _chunk("OMMS is owned by the APAC team.")
    from mimir.crawler.prompts import build_extraction_prompt
    llm.set_response(build_extraction_prompt(chunk.content), json.dumps(payload))
    result = extract(chunk, llm)
    assert len(result.entities) == 1
    assert result.entities[0].name == "OMMS"
    assert result.entities[0].type == "auros:TradingService"


@pytest.mark.phase5
def test_extract_parses_relationships() -> None:
    payload = {
        "entities": [
            {"name": "panic_server", "type": "auros:RiskSystem"},
            {"name": "risk_engine", "type": "auros:RiskSystem"},
        ],
        "relationships": [
            {"subject": "panic_server", "predicate": "auros:dependsOn", "object": "risk_engine"}
        ],
    }
    llm = FakeLLM()
    chunk = _chunk("panic_server depends on the risk engine.")
    from mimir.crawler.prompts import build_extraction_prompt
    llm.set_response(build_extraction_prompt(chunk.content), json.dumps(payload))
    result = extract(chunk, llm)
    assert len(result.relationships) == 1
    assert result.relationships[0].predicate == "auros:dependsOn"


@pytest.mark.phase5
def test_extract_parses_observations() -> None:
    payload = {
        "entities": [{"name": "OMMS", "type": "auros:TradingService"}],
        "observations": [
            {"entity_name": "OMMS", "type": "risk", "description": "SLO breach risk"}
        ],
    }
    llm = FakeLLM()
    chunk = _chunk("OMMS has SLO breach risk.")
    from mimir.crawler.prompts import build_extraction_prompt
    llm.set_response(build_extraction_prompt(chunk.content), json.dumps(payload))
    result = extract(chunk, llm)
    assert len(result.observations) == 1
    assert result.observations[0].type == "risk"


@pytest.mark.phase5
def test_extract_invalid_json_sets_parse_error() -> None:
    llm = FakeLLM()
    chunk = _chunk("some content")
    from mimir.crawler.prompts import build_extraction_prompt
    llm.set_response(build_extraction_prompt(chunk.content), "not json at all {{{")
    result = extract(chunk, llm)
    assert result.parse_error is not None
    assert result.entities == []


@pytest.mark.phase5
def test_extract_empty_response_returns_empty_result() -> None:
    llm = FakeLLM()
    chunk = _chunk("some content")
    from mimir.crawler.prompts import build_extraction_prompt
    llm.set_response(build_extraction_prompt(chunk.content), "{}")
    result = extract(chunk, llm)
    assert result.entities == []
    assert result.parse_error is None


@pytest.mark.phase5
def test_extract_source_type_preserved() -> None:
    llm = FakeLLM()
    chunk = Chunk(
        id="gh_001", source_type="github", content="readme content", retrieved_at=_NOW
    )
    from mimir.crawler.prompts import build_extraction_prompt
    llm.set_response(build_extraction_prompt(chunk.content), "{}")
    result = extract(chunk, llm)
    assert result.source_type == "github"
