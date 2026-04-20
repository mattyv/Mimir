"""Phase 9 — grounder tests (Wikidata linking + polarity enforcement)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.grounder.polarity import (
    PolarityViolation,
    are_polarity_opposites,
    assert_no_polarity_conflict,
)
from mimir.grounder.wikidata import WikidataMatch, find_wikidata_match, ground_entity
from mimir.models.base import Grounding, GroundingTier, Temporal, Visibility
from mimir.models.nodes import Entity
from mimir.persistence.repository import EntityRepository
from tests.conftest import FakeSPARQL

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_VOCAB = "0.1.0"


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=1, stop_reason="test")


def _make_entity(conn: psycopg.Connection[dict[str, Any]], name: str) -> str:
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}:schema:Organization"))
    EntityRepository(conn).upsert(
        Entity(
            id=eid,
            type="schema:Organization",
            name=name,
            description="",
            created_at=_NOW,
            confidence=0.9,
            grounding=_grounding(),
            temporal=Temporal(valid_from=_NOW),
            visibility=Visibility(acl=["internal"], sensitivity="internal"),
            vocabulary_version=_VOCAB,
        )
    )
    return eid


# ── Wikidata linker ───────────────────────────────────────────────────────────

_WD_RESPONSE: dict[str, Any] = {
    "results": {
        "bindings": [
            {
                "item": {"value": "http://www.wikidata.org/entity/Q42"},
                "itemLabel": {"value": "Douglas Adams"},
                "itemDescription": {"value": "British author"},
            }
        ]
    }
}


@pytest.mark.phase9
def test_find_wikidata_match_exact() -> None:
    sparql = FakeSPARQL()
    from mimir.grounder.wikidata import _LABEL_QUERY_TEMPLATE

    query = _LABEL_QUERY_TEMPLATE.format(name="Douglas Adams")
    sparql.set_response(query, _WD_RESPONSE)

    match = find_wikidata_match("Douglas Adams", sparql)
    assert match is not None
    assert match.qid == "Q42"
    assert match.label == "Douglas Adams"
    assert match.score == 1.0


@pytest.mark.phase9
def test_find_wikidata_match_partial_score() -> None:
    sparql = FakeSPARQL()
    from mimir.grounder.wikidata import _LABEL_QUERY_TEMPLATE

    response = {
        "results": {
            "bindings": [
                {
                    "item": {"value": "http://www.wikidata.org/entity/Q99"},
                    "itemLabel": {"value": "Some Other Name"},
                    "itemDescription": {"value": ""},
                }
            ]
        }
    }
    query = _LABEL_QUERY_TEMPLATE.format(name="my entity")
    sparql.set_response(query, response)

    match = find_wikidata_match("my entity", sparql)
    assert match is not None
    assert match.score == 0.5


@pytest.mark.phase9
def test_find_wikidata_match_no_results() -> None:
    sparql = FakeSPARQL()
    match = find_wikidata_match("nonexistent xyz", sparql)
    assert match is None


@pytest.mark.phase9
def test_wikidata_match_dataclass() -> None:
    m = WikidataMatch(qid="Q1", label="Universe", description="Everything", score=1.0)
    assert m.qid == "Q1"
    assert m.score == 1.0


@pytest.mark.phase9
def test_ground_entity_persists_qid(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "Douglas Adams")
    sparql = FakeSPARQL()
    from mimir.grounder.wikidata import _LABEL_QUERY_TEMPLATE

    query = _LABEL_QUERY_TEMPLATE.format(name="Douglas Adams")
    sparql.set_response(query, _WD_RESPONSE)

    match = ground_entity(eid, "Douglas Adams", sparql, pg)
    assert match is not None
    assert match.qid == "Q42"

    row = pg.execute("SELECT payload FROM entities WHERE id = %s", (eid,)).fetchone()
    assert row is not None
    assert row["payload"]["wikidata_qid"] == "Q42"


@pytest.mark.phase9
def test_ground_entity_no_match_returns_none(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "unknownxyz")
    sparql = FakeSPARQL()
    match = ground_entity(eid, "unknownxyz", sparql, pg)
    assert match is None


@pytest.mark.phase9
def test_ground_entity_sparql_error_returns_none(pg: psycopg.Connection[dict[str, Any]]) -> None:
    class BrokenSPARQL:
        def query(self, _: str) -> dict[str, Any]:
            raise RuntimeError("network error")

    eid = _make_entity(pg, "anysvc")
    match = ground_entity(eid, "anysvc", BrokenSPARQL(), pg)
    assert match is None


# ── polarity enforcement ──────────────────────────────────────────────────────


@pytest.mark.phase9
def test_polarity_opposites_detected() -> None:
    assert are_polarity_opposites("auros:dependsOn", "auros:independentOf")
    assert are_polarity_opposites("auros:independentOf", "auros:dependsOn")


@pytest.mark.phase9
def test_polarity_non_opposites() -> None:
    assert not are_polarity_opposites("auros:dependsOn", "auros:dependsOn")
    assert not are_polarity_opposites("auros:implements", "auros:dependsOn")


@pytest.mark.phase9
def test_assert_no_polarity_conflict_passes() -> None:
    assert_no_polarity_conflict("auros:dependsOn", "auros:implements")


@pytest.mark.phase9
def test_assert_no_polarity_conflict_raises() -> None:
    with pytest.raises(PolarityViolation) as exc_info:
        assert_no_polarity_conflict("auros:dependsOn", "auros:independentOf")
    assert "auros:dependsOn" in str(exc_info.value)
    assert "auros:independentOf" in str(exc_info.value)


@pytest.mark.phase9
def test_polarity_violation_attributes() -> None:
    err = PolarityViolation("auros:dependsOn", "auros:independentOf")
    assert err.predicate_a == "auros:dependsOn"
    assert err.predicate_b == "auros:independentOf"


@pytest.mark.phase9
def test_polarity_pairs_symmetric() -> None:
    assert are_polarity_opposites(
        "auros:dependsOn", "auros:independentOf"
    ) == are_polarity_opposites("auros:independentOf", "auros:dependsOn")
