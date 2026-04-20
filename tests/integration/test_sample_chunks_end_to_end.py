"""End-to-end integration test: 5 sample chunks → pipeline → expected entities."""

from __future__ import annotations

import json
from typing import Any

import psycopg
import pytest

from mimir.adapters.base import Chunk
from mimir.crawler.pipeline import process_chunk
from mimir.persistence.repository import EntityRepository, PropertyRepository


def _fake_llm_for_chunk(chunk: Chunk) -> Any:
    """Return a FakeLLM pre-programmed with spine + observation responses for each chunk."""
    from mimir.crawler.prompts import build_observations_prompt, build_spine_prompt
    from tests.conftest import FakeLLM

    llm = FakeLLM()

    if chunk.id == "confluence_001":
        llm.set_response(
            build_spine_prompt(chunk.content),
            json.dumps({
                "entities": [
                    {"name": "OMMS", "type": "auros:TradingService", "description": "Options market making service"},
                    {"name": "APAC team", "type": "auros:TradingTeam", "description": "APAC trading team"},
                    {"name": "FIX connector", "type": "auros:Connector", "description": "FIX protocol connector"},
                    {"name": "CME", "type": "auros:Venue", "description": "CME exchange"},
                    {"name": "ICE", "type": "auros:Venue", "description": "ICE exchange"},
                ],
                "properties": [
                    {"entity_name": "OMMS", "key": "schema:name", "value": "OMMS"},
                ],
                "relationships": [
                    {"subject": "APAC team", "predicate": "auros:owns", "object": "OMMS"},
                    {"subject": "OMMS", "predicate": "auros:dependsOn", "object": "FIX connector"},
                    {"subject": "FIX connector", "predicate": "auros:connects", "object": "CME"},
                    {"subject": "FIX connector", "predicate": "auros:connects", "object": "ICE"},
                ],
            }),
        )

    elif chunk.id == "github_001":
        llm.set_response(
            build_spine_prompt(chunk.content),
            json.dumps({
                "entities": [
                    {"name": "panic_server", "type": "auros:TradingService", "description": "Safety circuit breaker"},
                    {"name": "risk_engine", "type": "auros:RiskSystem", "description": "Risk computation engine"},
                    {"name": "risk-infra team", "type": "auros:TradingTeam", "description": "Risk infrastructure team"},
                ],
                "relationships": [
                    {"subject": "panic_server", "predicate": "auros:dependsOn", "object": "risk_engine"},
                    {"subject": "risk-infra team", "predicate": "auros:owns", "object": "panic_server"},
                ],
            }),
        )

    elif chunk.id == "slack_001":
        llm.set_response(
            build_spine_prompt(chunk.content),
            json.dumps({
                "entities": [
                    {"name": "PAN-12445", "type": "schema:SoftwareApplication", "description": "Ticket PAN-12445"},
                    {"name": "hawkeye", "type": "auros:TradingService", "description": "Hawkeye service"},
                ],
                "relationships": [
                    {"subject": "PAN-12445", "predicate": "auros:dependsOn", "object": "hawkeye"},
                ],
            }),
        )
        llm.set_response(
            build_observations_prompt(chunk.content),
            json.dumps({"observations": [
                {"entity_name": "PAN-12445", "type": "risk", "description": "Blocked on sub-account consolidation"},
            ]}),
        )

    elif chunk.id == "interview_001":
        llm.set_response(
            build_spine_prompt(chunk.content),
            json.dumps({
                "entities": [
                    {"name": "hedge book feed", "type": "auros:TradingService", "description": "Hedge book feed"},
                    {"name": "CME clearing", "type": "auros:Venue", "description": "CME clearing house"},
                    {"name": "FIX connector", "type": "auros:Connector", "description": "FIX protocol connector"},
                ],
                "relationships": [
                    {"subject": "hedge book feed", "predicate": "auros:dependsOn", "object": "FIX connector"},
                    {"subject": "FIX connector", "predicate": "auros:connects", "object": "CME clearing"},
                ],
            }),
        )
        llm.set_response(
            build_observations_prompt(chunk.content),
            json.dumps({"observations": [
                {"entity_name": "hedge book feed", "type": "risk",
                 "description": "Hard dependency on FIX connector — outage stops quoting"},
            ]}),
        )

    elif chunk.id == "code_001":
        llm.set_response(
            build_spine_prompt(chunk.content),
            json.dumps({
                "entities": [
                    {"name": "risk_engine", "type": "auros:RiskSystem", "description": "Python risk engine"},
                ],
                "properties": [
                    {"entity_name": "risk_engine", "key": "schema:programmingLanguage", "value": "Python 3.12"},
                ],
            }),
        )
        llm.set_response(
            build_observations_prompt(chunk.content),
            json.dumps({"observations": [
                {"entity_name": "risk_engine", "type": "smell",
                 "description": "Cyclomatic complexity 14 — high"},
            ]}),
        )

    return llm


@pytest.mark.integration
def test_sample_chunks_produce_expected_entities(
    pg: psycopg.Connection[dict[str, Any]],
    sample_chunks: list[Chunk],
) -> None:
    """Run the 5 sample chunks through the pipeline. Assert eval-relevant entities appear."""
    for chunk in sample_chunks:
        llm = _fake_llm_for_chunk(chunk)
        result = process_chunk(chunk, llm, pg)
        assert result.parse_error is None, f"Parse error on {chunk.id}: {result.parse_error}"

    entities = EntityRepository(pg).list_active(limit=200)
    names = {e["name"].lower() for e in entities}

    assert "omms" in names
    assert "panic_server" in names
    assert any("risk" in n for n in names)
    assert "pan-12445" in names
    assert any("fix connector" in n for n in names)
    assert any("hedge book feed" in n for n in names)


@pytest.mark.integration
def test_sample_chunks_insert_properties(
    pg: psycopg.Connection[dict[str, Any]],
    sample_chunks: list[Chunk],
) -> None:
    """Properties are persisted, not dropped."""
    for chunk in sample_chunks:
        llm = _fake_llm_for_chunk(chunk)
        process_chunk(chunk, llm, pg)

    repo = PropertyRepository(pg)
    entities = EntityRepository(pg).list_active(limit=200)
    all_props: list[dict[str, Any]] = []
    for e in entities:
        all_props.extend(repo.list_for_entity(e["id"]))

    assert len(all_props) >= 1


@pytest.mark.integration
def test_pipeline_result_counts_are_nonzero(
    pg: psycopg.Connection[dict[str, Any]],
    sample_chunks: list[Chunk],
) -> None:
    """Each real chunk produces >0 entities."""
    for chunk in sample_chunks:
        llm = _fake_llm_for_chunk(chunk)
        result = process_chunk(chunk, llm, pg)
        assert result.entities_upserted > 0, f"No entities from {chunk.id}"
