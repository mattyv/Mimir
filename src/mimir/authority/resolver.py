"""Conflict resolution — expire lower-authority properties; flag polarity conflicts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg

from mimir.adapters.base import SourceType
from mimir.authority.conflicts import PolarityConflict, PropertyConflict
from mimir.authority.ranking import trust_score


@dataclass
class ResolutionResult:
    entity_id: str
    key: str
    kept_source: str
    expired_count: int


def resolve_property_conflict(
    conflict: PropertyConflict,
    conn: psycopg.Connection[dict[str, Any]],
) -> ResolutionResult:
    """Expire all property rows except the highest-authority one.

    When trust scores tie, first in the ranked list (highest trust) wins.
    """

    def _score(v: dict[str, Any]) -> float:
        raw = v.get("source_type") or "slack"
        st: SourceType = raw  # type: ignore[assignment]
        return trust_score(st, conflict.key)

    ranked = sorted(conflict.values, key=_score, reverse=True)
    winner = ranked[0]
    losers = ranked[1:]

    now = datetime.now(UTC)
    expired = 0
    for loser in losers:
        prop_id = loser["property_id"]
        result = conn.execute(
            """
            UPDATE properties
            SET valid_until = %s
            WHERE id = %s::bigint
              AND valid_until IS NULL
            """,
            (now, prop_id),
        )
        expired += result.rowcount

    return ResolutionResult(
        entity_id=conflict.entity_id,
        key=conflict.key,
        kept_source=str(winner.get("source_type", "unknown")),
        expired_count=expired,
    )


def combine_corroborated_confidence(confidences: list[float]) -> float:
    """Combine confidences from multiple corroborating sources.

    Uses the Noisy-OR formula: 1 - product(1 - c_i), capped at 0.99.
    With ≥2 sources, the floor is 0.4 to resist over-decay.
    """
    if not confidences:
        return 0.0
    result = 1.0
    for c in confidences:
        result *= 1.0 - c
    combined = 1.0 - result
    return min(combined, 0.99)


def flag_polarity_conflict(
    conflict: PolarityConflict,
    conn: psycopg.Connection[dict[str, Any]],
) -> int:
    """Insert an 'inconsistency' Observation for a polarity conflict.

    Returns the new observation's auto-assigned id.
    """
    from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
    from mimir.models.nodes import Observation
    from mimir.persistence.repository import ObservationRepository

    obs = Observation(
        entity_id=conflict.subject_id,
        type="inconsistency",
        description=(
            f"Polarity conflict: both {conflict.predicates[0]} and "
            f"{conflict.predicates[1]} asserted between "
            f"{conflict.subject_id} and {conflict.object_id}"
        ),
        confidence=1.0,
        source=Source(
            type="confluence",
            reference="system://authority-resolver",
            retrieved_at=datetime.now(UTC),
        ),
        grounding=Grounding(
            tier=GroundingTier.source_cited,
            depth=0,
            stop_reason="polarity_conflict_detected",
        ),
        temporal=Temporal(valid_from=datetime.now(UTC)),
        visibility=Visibility(acl=["internal"], sensitivity="internal"),
        vocabulary_version="0.1.0",
    )
    return ObservationRepository(conn).insert(obs)
