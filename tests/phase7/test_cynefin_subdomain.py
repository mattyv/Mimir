"""Phase 7 — per-subdomain Cynefin classification tests."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import psycopg
import pytest

from mimir.cynefin.domain import CynefinDomain
from mimir.cynefin.subdomain import SubdomainClassification, _subdomain_of, classify_subdomains

pytestmark = pytest.mark.phase7

_NOW = datetime(2026, 4, 19, tzinfo=UTC)


def test_subdomain_of_explicit_tag() -> None:
    row = {"entity_type": "schema:Organization", "payload": {"subdomain": "trading"}}
    assert _subdomain_of(row) == "trading"


def test_subdomain_of_acl_group() -> None:
    row = {
        "entity_type": "schema:Organization",
        "payload": {"visibility": {"acl": ["team:risk-eng", "internal"]}},
    }
    assert _subdomain_of(row) == "team:risk-eng"


def test_subdomain_of_entity_type_namespace() -> None:
    row = {"entity_type": "auros:TradingService", "payload": {}}
    assert _subdomain_of(row) == "auros"


def test_subdomain_of_no_namespace() -> None:
    row = {"entity_type": "Service", "payload": {}}
    assert _subdomain_of(row) == "Service"


def test_classify_subdomains_empty_db(_pg_schema: None) -> None:
    import os

    with psycopg.connect(
        os.environ.get("DATABASE_URL", "dbname=mimir_test user=root"),
        row_factory=psycopg.rows.dict_row,
        autocommit=True,
    ) as conn:
        results = classify_subdomains(conn)
    assert results == []


def test_classify_subdomains_single_entity(_pg_schema: None) -> None:
    import os

    dsn = os.environ.get("DATABASE_URL", "dbname=mimir_test user=root")
    eid = str(uuid.uuid4())
    payload = json.dumps(
        {
            "subdomain": "test_domain",
            "grounding": {"tier": "source_cited", "depth": 1, "stop_reason": "test"},
            "visibility": {"acl": ["internal"], "sensitivity": "internal"},
        }
    )
    with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO entities
                (id, entity_type, name, name_normalized, description, confidence,
                 valid_from, vocabulary_version, payload, graph_version)
            VALUES (%s, 'schema:Organization', 'TestOrg', 'testorg', '', 0.95,
                    NOW(), '0.1.0', %s::jsonb, 0)
            ON CONFLICT DO NOTHING
            """,
            (eid, payload),
        )
        results = classify_subdomains(conn)
        # Cleanup
        conn.execute("DELETE FROM entities WHERE id = %s", (eid,))

    assert any(r.subdomain == "test_domain" for r in results)
    matching = next(r for r in results if r.subdomain == "test_domain")
    assert matching.entity_count == 1
    assert isinstance(matching.domain, CynefinDomain)
    assert isinstance(matching, SubdomainClassification)
