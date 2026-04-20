"""Per-sub-domain Cynefin classification (§9.1).

A sub-domain is a logical grouping of entities derived from their ACL group,
source reference prefix, or an explicit tag in payload['subdomain'].
This module computes a Cynefin domain for each distinct sub-domain by
aggregating the individual entity classifications within it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg

from mimir.cynefin.domain import CynefinDomain, classify


@dataclass
class SubdomainClassification:
    subdomain: str
    domain: CynefinDomain
    entity_count: int
    avg_confidence: float
    dominant_obs_types: list[str]


def _subdomain_of(row: dict[str, Any]) -> str:
    """Extract sub-domain tag from an entity row.

    Priority:
    1. payload['subdomain'] — explicit tag
    2. First ACL group that isn't 'internal' or 'public'
    3. entity_type namespace (e.g. 'auros' from 'auros:TradingService')
    """
    payload = row.get("payload") or {}
    if isinstance(payload, dict) and payload.get("subdomain"):
        return str(payload["subdomain"])
    vis = payload.get("visibility", {}) if isinstance(payload, dict) else {}
    acl: list[str] = vis.get("acl", [])
    for group in acl:
        if group not in ("internal", "public", "restricted"):
            return group
    entity_type = str(row.get("entity_type") or "unknown")
    return entity_type.split(":")[0] if ":" in entity_type else entity_type


def classify_subdomains(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    limit_per_subdomain: int = 200,
) -> list[SubdomainClassification]:
    """Classify each distinct sub-domain found in active entities.

    Fetches all active entities, groups by sub-domain, then runs the
    per-entity Cynefin heuristic aggregated to the group level.
    """
    entity_rows = conn.execute(
        "SELECT id, entity_type, confidence, payload FROM entities WHERE valid_until IS NULL"
    ).fetchall()

    # Group entity ids by subdomain
    subdomain_entities: dict[str, list[dict[str, Any]]] = {}
    for row in entity_rows:
        sd = _subdomain_of(dict(row))
        subdomain_entities.setdefault(sd, []).append(dict(row))

    results: list[SubdomainClassification] = []

    for subdomain, entities in subdomain_entities.items():
        entity_ids = [e["id"] for e in entities]

        # Fetch observations for these entities (batch)
        if len(entity_ids) > limit_per_subdomain:
            entity_ids = entity_ids[:limit_per_subdomain]

        if not entity_ids:
            continue

        obs_rows = conn.execute(
            """
            SELECT observation_type, entity_id
              FROM observations
             WHERE entity_id = ANY(%s)
               AND valid_until IS NULL
            """,
            (entity_ids,),
        ).fetchall()

        # Fetch relationship count for these entities
        rel_row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM relationships
             WHERE (subject_id = ANY(%s) OR object_id = ANY(%s))
               AND valid_until IS NULL
            """,
            (entity_ids, entity_ids),
        ).fetchone()
        rel_count = int(rel_row["n"]) if rel_row else 0

        observations = [dict(r) for r in obs_rows]
        confidences = [float(e["confidence"]) for e in entities]
        avg_conf = sum(confidences) / len(confidences) if confidences else 1.0

        domain = classify(
            observations,
            relationship_count=rel_count,
            avg_confidence=avg_conf,
        )

        obs_type_counts: dict[str, int] = {}
        for obs in observations:
            t = obs.get("observation_type", "")
            obs_type_counts[t] = obs_type_counts.get(t, 0) + 1
        dominant = sorted(obs_type_counts, key=lambda k: obs_type_counts[k], reverse=True)[:3]

        results.append(
            SubdomainClassification(
                subdomain=subdomain,
                domain=domain,
                entity_count=len(entities),
                avg_confidence=round(avg_conf, 4),
                dominant_obs_types=dominant,
            )
        )

    return sorted(results, key=lambda r: r.entity_count, reverse=True)
