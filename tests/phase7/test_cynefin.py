"""Phase 7 — Cynefin domain classifier tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.cynefin.classifier import ClassificationResult, classify_entity
from mimir.cynefin.domain import CynefinDomain, classify
from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Observation, Relationship
from mimir.persistence.repository import (
    EntityRepository,
    ObservationRepository,
    RelationshipRepository,
)

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_VOCAB = "0.1.0"


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=1, stop_reason="test")


def _temporal() -> Temporal:
    return Temporal(valid_from=_NOW)


def _visibility() -> Visibility:
    return Visibility(acl=["internal"], sensitivity="internal")


def _source(st: str = "confluence") -> Source:
    return Source(type=st, reference="https://example.com", retrieved_at=_NOW)  # type: ignore[arg-type]


def _make_entity(conn: psycopg.Connection[dict[str, Any]], name: str, confidence: float = 0.9) -> str:
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}:auros:TradingService"))
    EntityRepository(conn).upsert(
        Entity(
            id=eid,
            type="auros:TradingService",
            name=name,
            description="",
            created_at=_NOW,
            confidence=confidence,
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )
    return eid


def _add_obs(
    conn: psycopg.Connection[dict[str, Any]],
    entity_id: str,
    obs_type: str,
    confidence: float = 0.9,
) -> None:
    ObservationRepository(conn).insert(
        Observation(
            entity_id=entity_id,
            type=obs_type,  # type: ignore[arg-type]
            description=f"test {obs_type}",
            confidence=confidence,
            source=_source(),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )


def _add_rel(
    conn: psycopg.Connection[dict[str, Any]],
    subj: str,
    obj: str,
) -> None:
    RelationshipRepository(conn).insert(
        Relationship(
            subject_id=subj,
            predicate="auros:dependsOn",
            object_id=obj,
            confidence=0.9,
            source=_source("github"),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )


# ── unit tests: classify() ────────────────────────────────────────────────────


@pytest.mark.phase7
def test_classify_chaotic_on_inconsistency() -> None:
    obs = [{"type": "inconsistency"}]
    assert classify(obs) == CynefinDomain.chaotic


@pytest.mark.phase7
def test_classify_chaotic_on_anti_pattern() -> None:
    obs = [{"type": "anti_pattern"}]
    assert classify(obs) == CynefinDomain.chaotic


@pytest.mark.phase7
def test_classify_chaotic_on_low_confidence() -> None:
    assert classify([], avg_confidence=0.3) == CynefinDomain.chaotic


@pytest.mark.phase7
def test_classify_complex_on_risk() -> None:
    obs = [{"type": "risk"}]
    assert classify(obs) == CynefinDomain.complex


@pytest.mark.phase7
def test_classify_complex_on_smell() -> None:
    obs = [{"type": "smell"}]
    assert classify(obs) == CynefinDomain.complex


@pytest.mark.phase7
def test_classify_complex_on_opportunity() -> None:
    obs = [{"type": "opportunity"}]
    assert classify(obs) == CynefinDomain.complex


@pytest.mark.phase7
def test_classify_complex_on_high_coupling() -> None:
    assert classify([], relationship_count=10) == CynefinDomain.complex


@pytest.mark.phase7
def test_classify_complicated_on_maturity() -> None:
    obs = [{"type": "maturity"}]
    assert classify(obs) == CynefinDomain.complicated


@pytest.mark.phase7
def test_classify_complicated_on_functional_state() -> None:
    obs = [{"type": "functional_state"}]
    assert classify(obs) == CynefinDomain.complicated


@pytest.mark.phase7
def test_classify_complicated_on_moderate_coupling() -> None:
    assert classify([], relationship_count=5) == CynefinDomain.complicated


@pytest.mark.phase7
def test_classify_clear_on_no_observations_high_confidence() -> None:
    assert classify([], avg_confidence=0.95) == CynefinDomain.clear


@pytest.mark.phase7
def test_classify_confused_default() -> None:
    assert classify([{"type": "strength"}], avg_confidence=0.6) == CynefinDomain.confused


@pytest.mark.phase7
def test_classify_chaotic_beats_complex() -> None:
    obs = [{"type": "inconsistency"}, {"type": "risk"}]
    assert classify(obs) == CynefinDomain.chaotic


@pytest.mark.phase7
def test_cynefin_domain_values_exhaustive() -> None:
    expected = {"clear", "complicated", "complex", "chaotic", "confused"}
    assert {d.value for d in CynefinDomain} == expected


@pytest.mark.phase7
def test_classify_observation_type_key_fallback() -> None:
    obs = [{"observation_type": "inconsistency"}]
    assert classify(obs) == CynefinDomain.chaotic


# ── integration tests: classify_entity() ─────────────────────────────────────


@pytest.mark.phase7
def test_classify_entity_clear(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "clean_svc", confidence=0.95)
    result = classify_entity(eid, pg)
    assert result.domain == CynefinDomain.clear
    assert result.observation_count == 0
    assert isinstance(result, ClassificationResult)


@pytest.mark.phase7
def test_classify_entity_complex_via_risk(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "risky_svc")
    _add_obs(pg, eid, "risk")
    result = classify_entity(eid, pg)
    assert result.domain == CynefinDomain.complex
    assert result.observation_count == 1


@pytest.mark.phase7
def test_classify_entity_chaotic_via_inconsistency(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "chaos_svc")
    _add_obs(pg, eid, "inconsistency")
    result = classify_entity(eid, pg)
    assert result.domain == CynefinDomain.chaotic


@pytest.mark.phase7
def test_classify_entity_complicated_via_relationships(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "hub_svc")
    deps = [_make_entity(pg, f"dep_{i}") for i in range(5)]
    for dep in deps:
        _add_rel(pg, eid, dep)
    result = classify_entity(eid, pg)
    assert result.domain == CynefinDomain.complicated
    assert result.relationship_count == 5


@pytest.mark.phase7
def test_classify_entity_result_fields(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "field_svc", confidence=0.8)
    result = classify_entity(eid, pg)
    assert result.entity_id == eid
    assert 0.0 <= result.avg_confidence <= 1.0
    assert result.relationship_count == 0
