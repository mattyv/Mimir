"""Phase 5 — extractor tests using FakeLLM."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from mimir.adapters.base import Chunk
from mimir.crawler.extractor import (
    ExtractionResult,
    RawEntity,
    extract,
    extract_grounding_candidates,
    extract_observations,
    extract_spine,
    extract_three_pass,
)
from mimir.crawler.prompts import (
    build_grounding_candidates_prompt,
    build_observations_prompt,
    build_spine_prompt,
)
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



# ── Three-pass extractor tests ────────────────────────────────────────────────


def _spine_llm(content: str, data: dict) -> FakeLLM:
    llm = FakeLLM()
    llm.set_response(build_spine_prompt(content), json.dumps(data))
    return llm


@pytest.mark.phase5
def test_extract_spine_returns_entities() -> None:
    content = "OMMS is the options market maker."
    data = {"entities": [{"name": "OMMS", "type": "auros:TradingService", "description": "MM"}]}
    result = extract_spine(_chunk(content), _spine_llm(content, data))
    assert len(result.entities) == 1
    assert result.entities[0].name == "OMMS"
    assert result.parse_error is None


@pytest.mark.phase5
def test_extract_spine_parse_error_sets_field() -> None:
    content = "some content"
    llm = FakeLLM()
    llm.set_response(build_spine_prompt(content), "NOT JSON {{{")
    result = extract_spine(_chunk(content), llm)
    assert result.parse_error is not None
    assert result.entities == []


@pytest.mark.phase5
def test_extract_grounding_candidates_returns_matches() -> None:
    content = "We use TDD in our testing workflow."
    entity = RawEntity(name="TDD", type="auros:provisional:methodology", description="")
    llm = FakeLLM()
    entity_list = "- TDD (auros:provisional:methodology)"
    prompt = build_grounding_candidates_prompt(content, entity_list)
    response = json.dumps({"candidates": [
        {"entity_name": "TDD", "wikidata_qid": "Q7838228", "label": "test-driven development", "category": "methodology"}
    ]})
    llm.set_response(prompt, response)
    candidates = extract_grounding_candidates(_chunk(content), llm, [entity])
    assert len(candidates) == 1
    assert candidates[0].wikidata_qid == "Q7838228"
    assert candidates[0].category == "methodology"


@pytest.mark.phase5
def test_extract_grounding_candidates_graceful_on_invalid_json() -> None:
    content = "some content"
    entity = RawEntity(name="OMMS", type="auros:TradingService", description="")
    llm = FakeLLM()
    # No response set → FakeLLM returns non-JSON → graceful empty
    candidates = extract_grounding_candidates(_chunk(content), llm, [entity])
    assert candidates == []


@pytest.mark.phase5
def test_extract_grounding_candidates_empty_when_no_entities() -> None:
    content = "no entities here"
    llm = FakeLLM()
    candidates = extract_grounding_candidates(_chunk(content), llm, [])
    assert candidates == []


@pytest.mark.phase5
def test_extract_observations_returns_obs() -> None:
    content = "OMMS has SLO breach risk."
    llm = FakeLLM()
    obs_data = {"observations": [{"entity_name": "OMMS", "type": "risk", "description": "SLO breach"}]}
    llm.set_response(build_observations_prompt(content), json.dumps(obs_data))
    obs = extract_observations(_chunk(content), llm)
    assert len(obs) == 1
    assert obs[0].type == "risk"
    assert obs[0].entity_name == "OMMS"


@pytest.mark.phase5
def test_extract_observations_graceful_on_invalid_json() -> None:
    content = "some content"
    llm = FakeLLM()
    # No response set → graceful empty
    obs = extract_observations(_chunk(content), llm)
    assert obs == []


@pytest.mark.phase5
def test_extract_three_pass_combines_spine_and_observations() -> None:
    content = "panic_server depends on risk_engine and has high latency risk."
    llm = FakeLLM()
    spine = {
        "entities": [
            {"name": "panic_server", "type": "auros:RiskSystem", "description": ""},
            {"name": "risk_engine", "type": "auros:RiskSystem", "description": ""},
        ],
        "relationships": [
            {"subject": "panic_server", "predicate": "auros:dependsOn", "object": "risk_engine"}
        ],
    }
    obs = {"observations": [{"entity_name": "panic_server", "type": "risk", "description": "High latency"}]}
    llm.set_response(build_spine_prompt(content), json.dumps(spine))
    llm.set_response(build_observations_prompt(content), json.dumps(obs))
    result = extract_three_pass(_chunk(content), llm)
    assert result.parse_error is None
    assert len(result.entities) == 2
    assert len(result.relationships) == 1
    assert len(result.observations) == 1
    assert result.observations[0].type == "risk"


@pytest.mark.phase5
def test_extract_three_pass_spine_error_stops_early() -> None:
    content = "some content"
    llm = FakeLLM()
    llm.set_response(build_spine_prompt(content), "BROKEN JSON {{{")
    result = extract_three_pass(_chunk(content), llm)
    assert result.parse_error is not None
    assert result.entities == []
    assert result.observations == []


@pytest.mark.phase5
def test_extract_three_pass_obs_error_is_graceful() -> None:
    content = "OMMS service."
    llm = FakeLLM()
    spine = {"entities": [{"name": "OMMS", "type": "auros:TradingService", "description": ""}]}
    llm.set_response(build_spine_prompt(content), json.dumps(spine))
    # Observations prompt not set → invalid JSON → graceful empty
    result = extract_three_pass(_chunk(content), llm)
    assert result.parse_error is None
    assert len(result.entities) == 1
    assert result.observations == []


@pytest.mark.phase5
def test_extract_three_pass_grounding_candidates_populated() -> None:
    content = "We use TDD for all our tests."
    llm = FakeLLM()
    spine = {"entities": [{"name": "TDD", "type": "auros:provisional:methodology", "description": ""}]}
    llm.set_response(build_spine_prompt(content), json.dumps(spine))
    entity_list = "- TDD (auros:provisional:methodology)"
    grounding_prompt = build_grounding_candidates_prompt(content, entity_list)
    grounding_response = json.dumps({"candidates": [
        {"entity_name": "TDD", "wikidata_qid": "Q7838228", "label": "test-driven development", "category": "methodology"}
    ]})
    llm.set_response(grounding_prompt, grounding_response)
    result = extract_three_pass(_chunk(content), llm)
    assert len(result.grounding_candidates) == 1
    assert result.grounding_candidates[0].wikidata_qid == "Q7838228"
