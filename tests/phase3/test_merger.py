"""Phase 3 — entity merge tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Observation, Property, Relationship
from mimir.persistence.graph_version import current_graph_version
from mimir.persistence.repository import (
    EntityRepository,
    ObservationRepository,
    PropertyRepository,
    RelationshipRepository,
)
from mimir.resolution.merger import MergeResult, merge_entities

# ── helpers ────────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 4, 19, tzinfo=UTC)


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=1, stop_reason="test")


def _temporal(*, days_ago: int = 0) -> Temporal:
    return Temporal(valid_from=_NOW - timedelta(days=days_ago))


def _visibility() -> Visibility:
    return Visibility(acl=["internal"], sensitivity="internal")


def _source() -> Source:
    return Source(type="github", reference="repo://test", retrieved_at=_NOW)


def _entity(entity_id: str, name: str = "Test") -> Entity:
    return Entity(
        id=entity_id,
        type="schema:Organization",
        name=name,
        description="",
        created_at=_NOW,
        confidence=1.0,
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )


def _property(entity_id: str, key: str = "schema:name", value: str = "val") -> Property:
    return Property(
        entity_id=entity_id,
        key=key,
        value=value,
        value_type="str",
        confidence=1.0,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )


def _observation(entity_id: str) -> Observation:
    return Observation(
        entity_id=entity_id,
        type="strength",
        description="test observation",
        confidence=1.0,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )


def _relationship(subject_id: str, object_id: str) -> Relationship:
    return Relationship(
        subject_id=subject_id,
        predicate="auros:dependsOn",
        object_id=object_id,
        confidence=1.0,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )


# ── tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.phase3
def test_merge_reroutes_properties(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = EntityRepository(pg)
    prepo = PropertyRepository(pg)
    repo.upsert(_entity("merge_kept_p"))
    repo.upsert(_entity("merge_drop_p", "Drop"))
    prepo.insert(_property("merge_drop_p"))

    merge_entities("merge_kept_p", "merge_drop_p", pg)

    props = prepo.list_for_entity("merge_kept_p")
    assert len(props) == 1
    assert props[0]["entity_id"] == "merge_kept_p"


@pytest.mark.phase3
def test_merge_reroutes_observations(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = EntityRepository(pg)
    orepo = ObservationRepository(pg)
    repo.upsert(_entity("merge_kept_o"))
    repo.upsert(_entity("merge_drop_o", "Drop"))
    orepo.insert(_observation("merge_drop_o"))

    merge_entities("merge_kept_o", "merge_drop_o", pg)

    obs = orepo.list_for_entity("merge_kept_o")
    assert len(obs) == 1


@pytest.mark.phase3
def test_merge_reroutes_relationships_as_subject(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    repo = EntityRepository(pg)
    rrepo = RelationshipRepository(pg)
    repo.upsert(_entity("merge_kept_rs"))
    repo.upsert(_entity("merge_drop_rs", "Drop"))
    repo.upsert(_entity("merge_third_rs", "Third"))
    rrepo.insert(_relationship("merge_drop_rs", "merge_third_rs"))

    merge_entities("merge_kept_rs", "merge_drop_rs", pg)

    rels = rrepo.list_for_subject("merge_kept_rs")
    assert any(r["object_id"] == "merge_third_rs" for r in rels)


@pytest.mark.phase3
def test_merge_reroutes_relationships_as_object(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    repo = EntityRepository(pg)
    rrepo = RelationshipRepository(pg)
    repo.upsert(_entity("merge_kept_ro"))
    repo.upsert(_entity("merge_drop_ro", "Drop"))
    repo.upsert(_entity("merge_third_ro", "Third"))
    rrepo.insert(_relationship("merge_third_ro", "merge_drop_ro"))

    merge_entities("merge_kept_ro", "merge_drop_ro", pg)

    rels = rrepo.list_for_object("merge_kept_ro")
    assert any(r["subject_id"] == "merge_third_ro" for r in rels)


@pytest.mark.phase3
def test_merge_removes_self_loops(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = EntityRepository(pg)
    rrepo = RelationshipRepository(pg)
    repo.upsert(_entity("merge_kept_sl"))
    repo.upsert(_entity("merge_drop_sl", "Drop"))
    # Relationship between the two entities that will become a self-loop after merge
    rrepo.insert(_relationship("merge_drop_sl", "merge_kept_sl"))

    merge_entities("merge_kept_sl", "merge_drop_sl", pg)

    self_loops = pg.execute(
        "SELECT COUNT(*) AS n FROM relationships WHERE subject_id = %s AND object_id = %s",
        ("merge_kept_sl", "merge_kept_sl"),
    ).fetchone()
    assert self_loops is not None and self_loops["n"] == 0


@pytest.mark.phase3
def test_merge_expires_dropped_entity(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("merge_kept_ex"))
    repo.upsert(_entity("merge_drop_ex", "Drop"))

    merge_entities("merge_kept_ex", "merge_drop_ex", pg)

    row = pg.execute(
        "SELECT valid_until FROM entities WHERE id = %s",
        ("merge_drop_ex",),
    ).fetchone()
    assert row is not None and row["valid_until"] is not None


@pytest.mark.phase3
def test_merge_dropped_excluded_from_active_list(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("merge_kept_al"))
    repo.upsert(_entity("merge_drop_al", "Drop"))

    merge_entities("merge_kept_al", "merge_drop_al", pg)

    active_ids = {r["id"] for r in repo.list_active()}
    assert "merge_drop_al" not in active_ids
    assert "merge_kept_al" in active_ids


@pytest.mark.phase3
def test_merge_bumps_graph_version(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("merge_kept_gv"))
    repo.upsert(_entity("merge_drop_gv", "Drop"))
    v0 = current_graph_version(pg)

    merge_entities("merge_kept_gv", "merge_drop_gv", pg)

    assert current_graph_version(pg) == v0 + 1


@pytest.mark.phase3
def test_merge_result_fields(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = EntityRepository(pg)
    prepo = PropertyRepository(pg)
    orepo = ObservationRepository(pg)
    repo.upsert(_entity("merge_kept_rf"))
    repo.upsert(_entity("merge_drop_rf", "Drop"))
    prepo.insert(_property("merge_drop_rf"))
    orepo.insert(_observation("merge_drop_rf"))

    result = merge_entities("merge_kept_rf", "merge_drop_rf", pg)

    assert isinstance(result, MergeResult)
    assert result.kept_id == "merge_kept_rf"
    assert result.dropped_id == "merge_drop_rf"
    assert result.properties_rerouted == 1
    assert result.observations_rerouted == 1
    assert result.relationships_rerouted == 0
    assert result.graph_version > 0


@pytest.mark.phase3
def test_merge_missing_kept_raises(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("merge_drop_mk", "Drop"))

    with pytest.raises(ValueError, match="not found or already superseded"):
        merge_entities("does_not_exist", "merge_drop_mk", pg)


@pytest.mark.phase3
def test_merge_missing_dropped_raises(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("merge_kept_md"))

    with pytest.raises(ValueError, match="not found or already superseded"):
        merge_entities("merge_kept_md", "does_not_exist", pg)


@pytest.mark.phase3
def test_merge_already_expired_dropped_raises(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("merge_kept_ae"))
    # Insert an already-expired entity directly
    pg.execute(
        """
        INSERT INTO entities
            (id, entity_type, name, name_normalized, description, confidence,
             valid_from, valid_until, vocabulary_version, payload, graph_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            "merge_drop_ae",
            "schema:Organization",
            "Already Expired",
            "already expired",
            "",
            1.0,
            _NOW - timedelta(days=2),
            _NOW - timedelta(days=1),  # already expired
            "0.1.0",
            "{}",
            0,
        ),
    )

    with pytest.raises(ValueError, match="not found or already superseded"):
        merge_entities("merge_kept_ae", "merge_drop_ae", pg)
