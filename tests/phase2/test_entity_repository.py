"""Phase 2 — EntityRepository unit and integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity
from mimir.persistence.repository import EntityRepository, _normalize_name

pytestmark = pytest.mark.phase2

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_YESTERDAY = _NOW - timedelta(days=1)


def _entity(
    entity_id: str = "svc_001",
    name: str = "Test Service",
    entity_type: str = "auros:TradingService",
    valid_from: datetime = _NOW,
    valid_until: datetime | None = None,
) -> Entity:
    return Entity(
        id=entity_id,
        type=entity_type,
        name=name,
        description="A test trading service",
        created_at=_NOW,
        confidence=0.9,
        grounding=Grounding(tier=GroundingTier.source_cited, depth=0, stop_reason="test"),
        temporal=Temporal(valid_from=valid_from, valid_until=valid_until),
        visibility=Visibility(acl=["space:test"], sensitivity="internal"),
        vocabulary_version="0.1.0",
    )


# ── normalize_name ─────────────────────────────────────────────────────────────


def test_normalize_name_casefolds() -> None:
    assert _normalize_name("ACME Corp") == "acme corp"


def test_normalize_name_strips_whitespace() -> None:
    assert _normalize_name("  Acme  ") == "acme"


def test_normalize_name_nfc_normalizes() -> None:
    nfd = "A\u0301"  # A + combining acute accent
    assert _normalize_name(nfd) == "\u00e1"  # á (NFC)


# ── upsert ─────────────────────────────────────────────────────────────────────


def test_entity_upsert_inserts_new_row(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    inserted = repo.upsert(_entity())
    assert inserted is True
    assert repo.count() == 1


def test_entity_upsert_returns_false_on_update(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity())
    inserted = repo.upsert(_entity(name="Test Service", entity_type="auros:TradingService"))
    assert inserted is False


def test_entity_upsert_updates_description(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    original = _entity()
    repo.upsert(original)
    updated = Entity(
        **{
            **original.model_dump(),
            "description": "Updated description",
            "temporal": Temporal(valid_from=_NOW),
        }
    )
    repo.upsert(updated)
    row = repo.get("svc_001")
    assert row is not None
    assert row["description"] == "Updated description"


def test_entity_upsert_bumps_graph_version(pg: psycopg.Connection[Any]) -> None:
    from mimir.persistence.graph_version import current_graph_version

    repo = EntityRepository(pg)
    v0 = current_graph_version(pg)
    repo.upsert(_entity())
    assert current_graph_version(pg) == v0 + 1
    repo.upsert(_entity(entity_id="svc_002", name="Second Service"))
    assert current_graph_version(pg) == v0 + 2


# ── get ────────────────────────────────────────────────────────────────────────


def test_entity_get_returns_row(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity())
    row = repo.get("svc_001")
    assert row is not None
    assert row["id"] == "svc_001"
    assert row["entity_type"] == "auros:TradingService"


def test_entity_get_unknown_returns_none(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    assert repo.get("nonexistent") is None


def test_entity_get_as_of_active(pg: psycopg.Connection[Any]) -> None:
    """get returns row when as_of falls within valid_from..valid_until."""
    repo = EntityRepository(pg)
    repo.upsert(_entity(valid_from=_YESTERDAY, valid_until=_NOW + timedelta(days=1)))
    row = repo.get("svc_001", as_of=_NOW)
    assert row is not None


def test_entity_get_as_of_before_valid_from(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity(valid_from=_NOW, valid_until=None))
    row = repo.get("svc_001", as_of=_YESTERDAY)
    assert row is None


def test_entity_get_as_of_after_valid_until(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity(valid_from=_YESTERDAY, valid_until=_NOW - timedelta(hours=1)))
    row = repo.get("svc_001", as_of=_NOW)
    assert row is None


def test_entity_get_at_version(pg: psycopg.Connection[Any]) -> None:
    from mimir.persistence.graph_version import current_graph_version

    repo = EntityRepository(pg)
    repo.upsert(_entity())  # written at v0+1
    v_after_first = current_graph_version(pg)
    repo.upsert(_entity(entity_id="svc_002", name="Second Service"))  # written at v0+2
    # at_version=v_after_first should see svc_001 (written at v_after_first)
    row = repo.get("svc_001", at_version=v_after_first)
    assert row is not None


def test_entity_get_at_version_excludes_future(pg: psycopg.Connection[Any]) -> None:
    from mimir.persistence.graph_version import current_graph_version

    repo = EntityRepository(pg)
    v_before = current_graph_version(pg)
    repo.upsert(_entity())  # written at v_before+1
    # at_version=v_before should not see svc_001 (written at v_before+1 > v_before)
    row = repo.get("svc_001", at_version=v_before)
    assert row is None


# ── list_active ────────────────────────────────────────────────────────────────


def test_list_active_returns_all_current(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("a", "Alpha"))
    repo.upsert(_entity("b", "Beta"))
    rows = repo.list_active()
    assert len(rows) == 2


def test_list_active_filters_by_type(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("a", "Alpha", entity_type="auros:TradingService"))
    repo.upsert(_entity("b", "Beta", entity_type="schema:Organization"))
    rows = repo.list_active(entity_type="auros:TradingService")
    assert len(rows) == 1
    assert rows[0]["entity_type"] == "auros:TradingService"


def test_list_active_excludes_expired(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("a", "Active"))
    repo.upsert(_entity("b", "Expired", valid_from=_YESTERDAY, valid_until=_NOW - timedelta(hours=1)))
    # Default as_of=None → valid_until IS NULL → only "Active" is returned
    rows = repo.list_active()
    assert len(rows) == 1
    assert rows[0]["id"] == "a"


def test_list_active_pagination(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    for i in range(5):
        repo.upsert(_entity(f"svc_{i:02d}", f"Service {i:02d}"))
    page1 = repo.list_active(limit=2, offset=0)
    page2 = repo.list_active(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})


# ── delete ─────────────────────────────────────────────────────────────────────


def test_entity_delete_removes_row(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity())
    deleted = repo.delete("svc_001")
    assert deleted is True
    assert repo.count() == 0


def test_entity_delete_unknown_returns_false(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    assert repo.delete("nonexistent") is False


def test_entity_delete_cascades_to_properties(pg: psycopg.Connection[Any]) -> None:
    from mimir.models.nodes import Property
    from mimir.persistence.repository import PropertyRepository

    repo = EntityRepository(pg)
    repo.upsert(_entity())
    prop_repo = PropertyRepository(pg)
    prop_repo.insert(
        Property(
            entity_id="svc_001",
            key="schema:name",
            value="Acme",
            value_type="str",
            confidence=0.9,
            source=Source(type="github", reference="ref://x", retrieved_at=_NOW),  # type: ignore[arg-type]
            grounding=Grounding(tier=GroundingTier.source_cited, depth=0, stop_reason="test"),
            temporal=Temporal(valid_from=_NOW),
            visibility=Visibility(acl=[], sensitivity="internal"),
            vocabulary_version="0.1.0",
        )
    )
    repo.delete("svc_001")
    # The CASCADE should have removed the property
    rows = pg.execute("SELECT COUNT(*) AS n FROM properties WHERE entity_id = 'svc_001'").fetchone()
    assert rows is not None
    assert rows["n"] == 0


# ── count ──────────────────────────────────────────────────────────────────────


def test_entity_count_empty(pg: psycopg.Connection[Any]) -> None:
    assert EntityRepository(pg).count() == 0


def test_entity_count_after_inserts(pg: psycopg.Connection[Any]) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("a", "A"))
    repo.upsert(_entity("b", "B"))
    assert repo.count() == 2


# ── concurrency: unique constraint ────────────────────────────────────────────


def test_upsert_deduplicates_by_name_type(pg: psycopg.Connection[Any]) -> None:
    """Two entities with the same (name_normalized, entity_type) should collapse to one row."""
    repo = EntityRepository(pg)
    repo.upsert(_entity("id_a", "Acme Corp", entity_type="schema:Organization"))
    repo.upsert(_entity("id_b", "ACME Corp", entity_type="schema:Organization"))
    assert repo.count() == 1
