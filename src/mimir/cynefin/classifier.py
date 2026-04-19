"""Entity classifier — pulls data from DB and assigns a Cynefin domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg

from mimir.cynefin.domain import CynefinDomain, classify
from mimir.persistence.repository import ObservationRepository, RelationshipRepository


@dataclass
class ClassificationResult:
    entity_id: str
    domain: CynefinDomain
    observation_count: int
    relationship_count: int
    avg_confidence: float


def classify_entity(
    entity_id: str,
    conn: psycopg.Connection[dict[str, Any]],
) -> ClassificationResult:
    """Fetch entity observations + relationships and return a Cynefin domain."""
    observations = ObservationRepository(conn).list_for_entity(entity_id)
    relationships = RelationshipRepository(conn).list_for_subject(entity_id)
    relationships += RelationshipRepository(conn).list_for_object(entity_id)

    rel_count = len(relationships)

    if observations:
        avg_conf = sum(o["confidence"] for o in observations) / len(observations)
    else:
        row = conn.execute(
            "SELECT confidence FROM entities WHERE id = %s LIMIT 1",
            (entity_id,),
        ).fetchone()
        avg_conf = float(row["confidence"]) if row else 1.0

    domain = classify(observations, relationship_count=rel_count, avg_confidence=avg_conf)

    return ClassificationResult(
        entity_id=entity_id,
        domain=domain,
        observation_count=len(observations),
        relationship_count=rel_count,
        avg_confidence=avg_conf,
    )
