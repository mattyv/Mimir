"""Phase 12 — MCP tool tests (no live MCP server required)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.mcp.tools import (
    TOOL_REGISTRY,
    tool_classify_entity,
    tool_entity_cascade_risk,
    tool_explain_axiom,
    tool_find_relationships,
    tool_get_contradictions,
    tool_get_entity,
    tool_get_neighborhood,
    tool_graph_metrics,
    tool_list_entities,
    tool_list_observations,
    tool_search,
)
from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Observation, Relationship
from mimir.persistence.repository import (
    EntityRepository,
    ObservationRepository,
    RelationshipRepository,
)

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_VOCAB = "0.1.0"
_INTERNAL = {"internal"}


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=1, stop_reason="test")


def _source() -> Source:
    return Source(type="confluence", reference="https://example.com", retrieved_at=_NOW)


def _make_entity(
    conn: psycopg.Connection[dict[str, Any]],
    name: str,
    acl: list[str] | None = None,
) -> str:
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}:auros:TradingService"))
    EntityRepository(conn).upsert(
        Entity(
            id=eid,
            type="auros:TradingService",
            name=name,
            description=f"desc for {name}",
            created_at=_NOW,
            confidence=0.9,
            grounding=_grounding(),
            temporal=Temporal(valid_from=_NOW),
            visibility=Visibility(acl=acl or ["internal"], sensitivity="internal"),
            vocabulary_version=_VOCAB,
        )
    )
    return eid


def _make_obs(conn: psycopg.Connection[dict[str, Any]], entity_id: str, obs_type: str) -> None:
    ObservationRepository(conn).insert(
        Observation(
            entity_id=entity_id,
            type=obs_type,  # type: ignore[arg-type]
            description="test obs",
            confidence=0.9,
            source=_source(),
            grounding=_grounding(),
            temporal=Temporal(valid_from=_NOW),
            visibility=Visibility(acl=["internal"], sensitivity="internal"),
            vocabulary_version=_VOCAB,
        )
    )


# ── tool_get_entity ───────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_get_entity_found(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "svc_get")
    result = tool_get_entity({"entity_id": eid}, pg, _INTERNAL)
    assert "entity" in result
    assert result["entity"]["id"] == eid


@pytest.mark.phase12
def test_get_entity_not_found(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_get_entity({"entity_id": "ghost"}, pg, _INTERNAL)
    assert result["error"] == "not_found"


@pytest.mark.phase12
def test_get_entity_forbidden(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "svc_forbidden", acl=["team:secret"])
    result = tool_get_entity({"entity_id": eid}, pg, {"other-group"})
    assert result["error"] == "forbidden"


# ── tool_list_entities ────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_list_entities_returns_visible(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "list_svc")
    result = tool_list_entities({}, pg, _INTERNAL)
    ids = [e["id"] for e in result["entities"]]
    assert eid in ids
    assert result["count"] >= 1


@pytest.mark.phase12
def test_list_entities_type_filter(pg: psycopg.Connection[dict[str, Any]]) -> None:
    _make_entity(pg, "typed_svc")
    result = tool_list_entities({"entity_type": "auros:TradingService"}, pg, _INTERNAL)
    assert result["count"] >= 1
    for e in result["entities"]:
        assert e["entity_type"] == "auros:TradingService"


@pytest.mark.phase12
def test_list_entities_acl_filters_out_restricted(pg: psycopg.Connection[dict[str, Any]]) -> None:
    _make_entity(pg, "restricted_svc", acl=["team:secret"])
    result = tool_list_entities({}, pg, {"unrelated-group"})
    ids = [e["id"] for e in result["entities"]]
    restricted_eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "restricted_svc:auros:TradingService"))
    assert restricted_eid not in ids


# ── tool_classify_entity ──────────────────────────────────────────────────────


@pytest.mark.phase12
def test_classify_entity_clear(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "clear_svc")
    result = tool_classify_entity({"entity_id": eid}, pg)
    assert result["entity_id"] == eid
    assert result["domain"] == "clear"
    assert "observation_count" in result


@pytest.mark.phase12
def test_classify_entity_complex(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "complex_svc")
    _make_obs(pg, eid, "risk")
    result = tool_classify_entity({"entity_id": eid}, pg)
    assert result["domain"] == "complex"


# ── tool_list_observations ────────────────────────────────────────────────────


@pytest.mark.phase12
def test_list_observations_returns_obs(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "obs_svc")
    _make_obs(pg, eid, "risk")
    result = tool_list_observations({"entity_id": eid}, pg, _INTERNAL)
    assert result["count"] == 1


@pytest.mark.phase12
def test_list_observations_type_filter(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "obs_svc2")
    _make_obs(pg, eid, "risk")
    _make_obs(pg, eid, "maturity")
    result = tool_list_observations({"entity_id": eid, "observation_type": "risk"}, pg, _INTERNAL)
    assert result["count"] == 1
    assert result["observations"][0]["observation_type"] == "risk"


# ── tool_graph_metrics ────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_graph_metrics_returns_expected_fields(pg: psycopg.Connection[dict[str, Any]]) -> None:
    _make_entity(pg, "gm_svc")
    result = tool_graph_metrics({}, pg)
    assert "node_count" in result
    assert "edge_count" in result
    assert "has_cycles" in result
    assert result["node_count"] >= 1


# ── TOOL_REGISTRY ─────────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_tool_registry_contains_expected_tools() -> None:
    expected = {
        "get_entity", "list_entities", "classify_entity", "list_observations",
        "graph_metrics", "find_relationships", "get_neighborhood",
        "entity_cascade_risk", "search", "get_contradictions", "explain_axiom",
    }
    assert expected <= set(TOOL_REGISTRY.keys())


@pytest.mark.phase12
def test_tool_registry_callables() -> None:
    for name, fn in TOOL_REGISTRY.items():
        assert callable(fn), f"{name} should be callable"


# ── tool_find_relationships ───────────────────────────────────────────────────


def _make_rel(
    conn: psycopg.Connection[dict[str, Any]],
    subj: str,
    obj: str,
    predicate: str = "auros:dependsOn",
) -> None:
    RelationshipRepository(conn).insert(
        Relationship(
            subject_id=subj,
            predicate=predicate,
            object_id=obj,
            confidence=0.9,
            source=_source(),
            grounding=_grounding(),
            temporal=Temporal(valid_from=_NOW),
            visibility=Visibility(acl=["internal"], sensitivity="internal"),
            vocabulary_version=_VOCAB,
        )
    )


@pytest.mark.phase12
def test_find_relationships_requires_filter(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_find_relationships({}, pg, _INTERNAL)
    assert result["error"] == "at_least_one_filter_required"


@pytest.mark.phase12
def test_find_relationships_by_subject(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "fr_a")
    b = _make_entity(pg, "fr_b")
    _make_rel(pg, a, b)
    result = tool_find_relationships({"subject_id": a}, pg, _INTERNAL)
    assert result["count"] == 1
    assert result["relationships"][0]["subject_id"] == a


@pytest.mark.phase12
def test_find_relationships_by_predicate(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "fr_pred_a")
    b = _make_entity(pg, "fr_pred_b")
    _make_rel(pg, a, b, "auros:dependsOn")
    result = tool_find_relationships({"predicate": "auros:dependsOn"}, pg, _INTERNAL)
    assert result["count"] >= 1


@pytest.mark.phase12
def test_find_relationships_by_object(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "fr_obj_a")
    b = _make_entity(pg, "fr_obj_b")
    _make_rel(pg, a, b)
    result = tool_find_relationships({"object_id": b}, pg, _INTERNAL)
    assert result["count"] == 1
    assert result["relationships"][0]["object_id"] == b


@pytest.mark.phase12
def test_find_relationships_combination_filter(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "fr_combo_a")
    b = _make_entity(pg, "fr_combo_b")
    c = _make_entity(pg, "fr_combo_c")
    _make_rel(pg, a, b)
    _make_rel(pg, a, c)
    result = tool_find_relationships({"subject_id": a, "object_id": b}, pg, _INTERNAL)
    assert result["count"] == 1


@pytest.mark.phase12
def test_find_relationships_truncation_flag(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_find_relationships({"predicate": "auros:dependsOn", "limit": 1000}, pg, _INTERNAL)
    assert "truncated" in result


# ── tool_get_neighborhood ─────────────────────────────────────────────────────


@pytest.mark.phase12
def test_get_neighborhood_center_exists(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "nb_center")
    b = _make_entity(pg, "nb_neighbor")
    _make_rel(pg, a, b)
    result = tool_get_neighborhood({"entity_id": a, "depth": 1}, pg, _INTERNAL)
    assert result["center"] == a
    node_ids = [n["id"] for n in result["nodes"]]
    assert a in node_ids
    assert b in node_ids


@pytest.mark.phase12
def test_get_neighborhood_empty_graph(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_get_neighborhood({"entity_id": "ghost"}, pg, _INTERNAL)
    assert result["nodes"] == []


@pytest.mark.phase12
def test_get_neighborhood_depth_capped_at_3(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "nb_deep")
    result = tool_get_neighborhood({"entity_id": a, "depth": 99}, pg, _INTERNAL)
    assert result["depth"] == 3


@pytest.mark.phase12
def test_get_neighborhood_predicate_filter(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "nb_pf_a")
    b = _make_entity(pg, "nb_pf_b")
    _make_rel(pg, a, b, "auros:dependsOn")
    result = tool_get_neighborhood(
        {"entity_id": a, "depth": 1, "predicate": "auros:owns"}, pg, _INTERNAL
    )
    assert result["edges"] == []


# ── tool_entity_cascade_risk ──────────────────────────────────────────────────


@pytest.mark.phase12
def test_entity_cascade_risk_not_found(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_entity_cascade_risk({"entity_id": "ghost"}, pg, _INTERNAL)
    assert result["error"] == "not_found"


@pytest.mark.phase12
def test_entity_cascade_risk_isolated_node(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "cr_isolated")
    result = tool_entity_cascade_risk({"entity_id": eid}, pg, _INTERNAL)
    assert result["cascade_risk"] == 0.0
    assert result["downstream_count"] == 0


@pytest.mark.phase12
def test_entity_cascade_risk_with_downstream(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "cr_upstream")
    b = _make_entity(pg, "cr_downstream")
    _make_rel(pg, a, b)
    result = tool_entity_cascade_risk({"entity_id": a}, pg, _INTERNAL)
    assert result["cascade_risk"] > 0.0
    assert result["downstream_count"] >= 1
    assert b in result["downstream_entities"]


# ── tool_search ───────────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_search_finds_by_name(pg: psycopg.Connection[dict[str, Any]]) -> None:
    _make_entity(pg, "SearchableService")
    result = tool_search({"query": "searchable"}, pg, _INTERNAL)
    assert result["count"] >= 1
    names = [r["name"] for r in result["results"]]
    assert any("Searchable" in n for n in names)


@pytest.mark.phase12
def test_search_empty_results(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_search({"query": "zzznomatch999"}, pg, _INTERNAL)
    assert result["count"] == 0


@pytest.mark.phase12
def test_search_respects_acl(pg: psycopg.Connection[dict[str, Any]]) -> None:
    _make_entity(pg, "SecretSearchSvc", acl=["team:secret"])
    result = tool_search({"query": "secretsearch"}, pg, {"unrelated"})
    assert result["count"] == 0


# ── tool_get_contradictions ───────────────────────────────────────────────────


@pytest.mark.phase12
def test_get_contradictions_empty(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_get_contradictions({}, pg, _INTERNAL)
    assert "contradictions" in result
    assert isinstance(result["contradictions"], list)


# ── tool_explain_axiom ────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_explain_axiom_entity_found(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "explain_svc")
    result = tool_explain_axiom({"axiom_id": eid, "kind": "entity"}, pg, _INTERNAL)
    assert "axiom" in result
    assert result["kind"] == "entity"
    assert "grounding_tier" in result
    assert "sources" in result
    assert "wikidata_chain" in result


@pytest.mark.phase12
def test_explain_axiom_entity_not_found(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_explain_axiom({"axiom_id": "ghost-id"}, pg, _INTERNAL)
    assert result["error"] == "not_found"


@pytest.mark.phase12
def test_explain_axiom_entity_forbidden(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "explain_secret", acl=["team:secret"])
    result = tool_explain_axiom({"axiom_id": eid, "kind": "entity"}, pg, {"unrelated"})
    assert result["error"] == "forbidden"


@pytest.mark.phase12
def test_explain_axiom_unknown_kind(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_explain_axiom({"axiom_id": "x", "kind": "dragon"}, pg, _INTERNAL)
    assert result["error"] == "unknown_kind"
    assert result["kind"] == "dragon"


@pytest.mark.phase12
def test_explain_axiom_observation(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "obs_explain_svc")
    _make_obs(pg, eid, "risk")
    row = pg.execute(
        "SELECT id FROM observations WHERE entity_id = %s LIMIT 1", (eid,)
    ).fetchone()
    assert row is not None
    obs_id = str(row["id"])
    result = tool_explain_axiom({"axiom_id": obs_id, "kind": "observation"}, pg, _INTERNAL)
    assert result.get("error") is None
    assert result["kind"] == "observation"
    assert "axiom" in result


@pytest.mark.phase12
def test_explain_axiom_relationship(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "explain_rel_a")
    b = _make_entity(pg, "explain_rel_b")
    _make_rel(pg, a, b)
    row = pg.execute(
        "SELECT id FROM relationships WHERE subject_id = %s LIMIT 1", (a,)
    ).fetchone()
    assert row is not None
    rel_id = str(row["id"])
    result = tool_explain_axiom({"axiom_id": rel_id, "kind": "relationship"}, pg, _INTERNAL)
    assert result.get("error") is None
    assert result["kind"] == "relationship"
    assert "axiom" in result


@pytest.mark.phase12
def test_explain_axiom_invalid_id_for_bigserial(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_explain_axiom({"axiom_id": "not-an-int", "kind": "relationship"}, pg, _INTERNAL)
    assert result["error"] == "not_found"


@pytest.mark.phase12
def test_explain_axiom_default_kind_is_entity(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "default_kind_svc")
    result = tool_explain_axiom({"axiom_id": eid}, pg, _INTERNAL)
    assert result.get("error") is None
    assert result["kind"] == "entity"
