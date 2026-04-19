"""Phase 6 — source authority and conflict resolution tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.authority.conflicts import (
    detect_polarity_conflicts,
    detect_property_conflicts,
)
from mimir.authority.ranking import higher_authority, trust_score
from mimir.authority.resolver import (
    ResolutionResult,
    flag_polarity_conflict,
    resolve_property_conflict,
)
from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Property, Relationship
from mimir.persistence.repository import (
    EntityRepository,
    ObservationRepository,
    PropertyRepository,
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


def _source(source_type: str = "confluence", ref: str = "https://example.com") -> Source:
    return Source(type=source_type, reference=ref, retrieved_at=_NOW)  # type: ignore[arg-type]


def _make_entity(
    conn: psycopg.Connection[dict[str, Any]],
    name: str,
    entity_type: str = "auros:TradingService",
) -> str:
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}:{entity_type}"))
    EntityRepository(conn).upsert(
        Entity(
            id=eid,
            type=entity_type,
            name=name,
            description="",
            created_at=_NOW,
            confidence=0.9,
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )
    return eid


def _make_property(
    conn: psycopg.Connection[dict[str, Any]],
    entity_id: str,
    key: str,
    value: Any,
    source_type: str = "confluence",
) -> int:
    return PropertyRepository(conn).insert(
        Property(
            entity_id=entity_id,
            key=key,
            value=value,
            value_type=type(value).__name__,
            confidence=0.9,
            source=_source(source_type),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )


# ── ranking tests ─────────────────────────────────────────────────────────────


@pytest.mark.phase6
def test_trust_scores_ordered() -> None:
    assert trust_score("code_analysis") > trust_score("github")
    assert trust_score("github") > trust_score("confluence")
    assert trust_score("confluence") > trust_score("interview")
    assert trust_score("interview") > trust_score("slack")


@pytest.mark.phase6
def test_trust_score_range() -> None:
    for st in ("code_analysis", "github", "confluence", "interview", "slack"):
        s = trust_score(st)  # type: ignore[arg-type]
        assert 0.0 <= s <= 1.0


@pytest.mark.phase6
def test_higher_authority_picks_winner() -> None:
    assert higher_authority("github", "slack") == "github"
    assert higher_authority("slack", "code_analysis") == "code_analysis"


@pytest.mark.phase6
def test_higher_authority_tie_returns_first() -> None:
    assert higher_authority("confluence", "confluence") == "confluence"


# ── conflict detection tests ──────────────────────────────────────────────────


@pytest.mark.phase6
def test_detect_no_property_conflicts_when_values_agree(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    eid = _make_entity(pg, "svc_a")
    _make_property(pg, eid, "auros:hasOwner", "team-a", "confluence")
    _make_property(pg, eid, "auros:hasOwner", "team-a", "github")
    conflicts = detect_property_conflicts(eid, pg)
    assert conflicts == []


@pytest.mark.phase6
def test_detect_property_conflict_different_values(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    eid = _make_entity(pg, "svc_b")
    _make_property(pg, eid, "auros:hasOwner", "team-a", "confluence")
    _make_property(pg, eid, "auros:hasOwner", "team-b", "github")
    conflicts = detect_property_conflicts(eid, pg)
    assert len(conflicts) == 1
    assert conflicts[0].key == "auros:hasOwner"
    assert len(conflicts[0].values) == 2


@pytest.mark.phase6
def test_detect_property_conflict_ignores_expired(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    eid = _make_entity(pg, "svc_c")
    _make_property(pg, eid, "auros:hasOwner", "team-a", "confluence")
    pid = _make_property(pg, eid, "auros:hasOwner", "team-b", "github")
    # Expire the second property
    pg.execute("UPDATE properties SET valid_until = NOW() WHERE id = %s", (pid,))
    conflicts = detect_property_conflicts(eid, pg)
    assert conflicts == []


@pytest.mark.phase6
def test_detect_polarity_conflict(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    subj = _make_entity(pg, "svc_d")
    obj = _make_entity(pg, "svc_e")
    RelationshipRepository(pg).insert(
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
    RelationshipRepository(pg).insert(
        Relationship(
            subject_id=subj,
            predicate="auros:independentOf",
            object_id=obj,
            confidence=0.9,
            source=_source("slack"),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )
    conflicts = detect_polarity_conflicts(pg, subject_id=subj)
    assert len(conflicts) == 1
    assert subj == conflicts[0].subject_id
    assert set(conflicts[0].predicates) == {"auros:dependsOn", "auros:independentOf"}


@pytest.mark.phase6
def test_detect_polarity_no_conflict_single_predicate(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    subj = _make_entity(pg, "svc_f")
    obj = _make_entity(pg, "svc_g")
    RelationshipRepository(pg).insert(
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
    conflicts = detect_polarity_conflicts(pg, subject_id=subj)
    assert conflicts == []


# ── resolver tests ────────────────────────────────────────────────────────────


@pytest.mark.phase6
def test_resolve_property_conflict_expires_lower_authority(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    eid = _make_entity(pg, "svc_h")
    _make_property(pg, eid, "auros:hasOwner", "team-slack", "slack")
    _make_property(pg, eid, "auros:hasOwner", "team-github", "github")

    conflicts = detect_property_conflicts(eid, pg)
    assert len(conflicts) == 1

    result = resolve_property_conflict(conflicts[0], pg)
    assert isinstance(result, ResolutionResult)
    assert result.kept_source == "github"
    assert result.expired_count == 1

    # Only one active row should remain
    rows = pg.execute(
        "SELECT * FROM properties WHERE entity_id = %s AND valid_until IS NULL",
        (eid,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["payload"]["source"]["type"] == "github"


@pytest.mark.phase6
def test_resolve_property_conflict_result_fields(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    eid = _make_entity(pg, "svc_i")
    _make_property(pg, eid, "schema:name", "alpha", "confluence")
    _make_property(pg, eid, "schema:name", "beta", "slack")

    conflicts = detect_property_conflicts(eid, pg)
    result = resolve_property_conflict(conflicts[0], pg)
    assert result.entity_id == eid
    assert result.key == "schema:name"


@pytest.mark.phase6
def test_flag_polarity_conflict_creates_observation(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    subj = _make_entity(pg, "svc_j")
    obj = _make_entity(pg, "svc_k")

    from mimir.authority.conflicts import PolarityConflict as PC

    polarity = PC(
        subject_id=subj,
        object_id=obj,
        predicates=["auros:dependsOn", "auros:independentOf"],
    )
    obs_id = flag_polarity_conflict(polarity, pg)
    assert isinstance(obs_id, int)
    assert obs_id > 0

    obs_rows = ObservationRepository(pg).list_for_entity(subj, observation_type="inconsistency")
    assert len(obs_rows) == 1
    assert "Polarity conflict" in obs_rows[0]["description"]


@pytest.mark.phase6
def test_resolve_then_no_conflict(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    eid = _make_entity(pg, "svc_l")
    _make_property(pg, eid, "auros:hasOwner", "x", "slack")
    _make_property(pg, eid, "auros:hasOwner", "y", "code_analysis")

    conflicts = detect_property_conflicts(eid, pg)
    resolve_property_conflict(conflicts[0], pg)

    # After resolution there should be no more conflicts
    post_conflicts = detect_property_conflicts(eid, pg)
    assert post_conflicts == []


@pytest.mark.phase6
def test_detect_polarity_conflicts_global(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    subj = _make_entity(pg, "svc_m")
    obj = _make_entity(pg, "svc_n")
    RelationshipRepository(pg).insert(
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
    RelationshipRepository(pg).insert(
        Relationship(
            subject_id=subj,
            predicate="auros:independentOf",
            object_id=obj,
            confidence=0.9,
            source=_source("slack"),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )
    # Without subject_id filter — should still find it
    conflicts = detect_polarity_conflicts(pg)
    assert any(c.subject_id == subj for c in conflicts)
