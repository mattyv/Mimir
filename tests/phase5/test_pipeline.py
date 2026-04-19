"""Phase 5 — end-to-end pipeline tests (chunk → DB)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.adapters.base import Chunk
from mimir.crawler.pipeline import PipelineResult, process_chunk
from mimir.crawler.prompts import build_observations_prompt, build_spine_prompt
from mimir.persistence.repository import EntityRepository
from tests.conftest import FakeLLM

_NOW = datetime(2026, 4, 19, tzinfo=UTC)


def _chunk(content: str, chunk_id: str = "pipe_chunk") -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type="confluence",
        content=content,
        acl=["space:trading-eng"],
        retrieved_at=_NOW,
        reference="https://wiki.example.com/page",
    )


def _llm(content: str, spine_payload: dict, obs_payload: dict | None = None) -> FakeLLM:
    llm = FakeLLM()
    llm.set_response(build_spine_prompt(content), json.dumps(spine_payload))
    if obs_payload is not None:
        llm.set_response(build_observations_prompt(content), json.dumps(obs_payload))
    return llm


@pytest.mark.phase5
def test_pipeline_upserts_entities(pg: psycopg.Connection[dict[str, Any]]) -> None:
    content = "The OMMS service handles options market making."
    payload = {"entities": [{"name": "OMMS", "type": "auros:TradingService", "description": "Options MM"}]}
    result = process_chunk(_chunk(content), _llm(content, payload), pg)
    assert result.entities_upserted == 1
    assert EntityRepository(pg).count() >= 1


@pytest.mark.phase5
def test_pipeline_inserts_relationships(pg: psycopg.Connection[dict[str, Any]]) -> None:
    content = "panic_server depends on risk_engine."
    payload = {
        "entities": [
            {"name": "panic_server", "type": "auros:RiskSystem"},
            {"name": "risk_engine", "type": "auros:RiskSystem"},
        ],
        "relationships": [
            {"subject": "panic_server", "predicate": "auros:dependsOn", "object": "risk_engine"}
        ],
    }
    result = process_chunk(_chunk(content), _llm(content, payload), pg)
    assert result.relationships_inserted == 1


@pytest.mark.phase5
def test_pipeline_inserts_observations(pg: psycopg.Connection[dict[str, Any]]) -> None:
    content = "OMMS has high SLO breach risk."
    spine = {"entities": [{"name": "OMMS", "type": "auros:TradingService"}]}
    obs = {"observations": [{"entity_name": "OMMS", "type": "risk", "description": "SLO breach risk"}]}
    result = process_chunk(_chunk(content), _llm(content, spine, obs), pg)
    assert result.observations_inserted == 1


@pytest.mark.phase5
def test_pipeline_skips_relationship_with_unknown_entity(pg: psycopg.Connection[dict[str, Any]]) -> None:
    content = "service_a connects to unknown_b."
    payload = {
        "entities": [{"name": "service_a", "type": "auros:TradingService"}],
        "relationships": [{"subject": "service_a", "predicate": "auros:connects", "object": "unknown_b"}],
    }
    result = process_chunk(_chunk(content), _llm(content, payload), pg)
    assert result.relationships_inserted == 0
    assert "unknown_b" in result.unknown_entity_refs


@pytest.mark.phase5
def test_pipeline_handles_parse_error(pg: psycopg.Connection[dict[str, Any]]) -> None:
    content = "some content"
    llm = FakeLLM()
    llm.set_response(build_spine_prompt(content), "INVALID {{{")
    result = process_chunk(_chunk(content), llm, pg)
    assert result.parse_error is not None
    assert result.entities_upserted == 0


@pytest.mark.phase5
def test_pipeline_result_chunk_id(pg: psycopg.Connection[dict[str, Any]]) -> None:
    content = "plain content"
    llm = FakeLLM()
    llm.set_response(build_spine_prompt(content), "{}")
    result = process_chunk(_chunk(content, "my_chunk"), llm, pg)
    assert result.chunk_id == "my_chunk"
    assert isinstance(result, PipelineResult)


@pytest.mark.phase5
def test_pipeline_idempotent_entity_upsert(pg: psycopg.Connection[dict[str, Any]]) -> None:
    content = "OMMS service."
    payload = {"entities": [{"name": "OMMS", "type": "auros:TradingService"}]}
    process_chunk(_chunk(content, "chunk_a"), _llm(content, payload), pg)
    process_chunk(_chunk(content, "chunk_b"), _llm(content, payload), pg)
    rows = pg.execute(
        "SELECT COUNT(*) AS n FROM entities WHERE name = 'OMMS' AND entity_type = 'auros:TradingService'"
    ).fetchone()
    assert rows is not None and rows["n"] == 1
