"""Provisional IRI promotion workflow.

A provisional IRI becomes eligible for promotion to a core IRI when:
  - use_count >= 10
  - source_count >= 3
  - approved: bool flag set by a human reviewer

promote_provisional() validates eligibility and updates the vocabulary tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromotionResult:
    iri: str
    promoted: bool
    reason: str


def promote_provisional(
    iri: str,
    *,
    use_count: int,
    source_count: int,
    approved: bool,
    target_iri: str | None = None,
) -> PromotionResult:
    """Evaluate and execute promotion of a provisional IRI.

    Args:
        iri:          The provisional IRI (must start with ``auros:provisional:``).
        use_count:    How many times this IRI has been used.
        source_count: How many distinct sources have referenced it.
        approved:     Whether a human reviewer has approved promotion.
        target_iri:   The core IRI to promote to (e.g. ``auros:MyNewType``).
                      If None, a name is derived from the provisional suffix.

    Returns:
        PromotionResult with promoted=True if all gates passed.
    """
    if not iri.startswith("auros:provisional:"):
        return PromotionResult(iri=iri, promoted=False, reason="not_provisional")

    if use_count < 10:
        return PromotionResult(iri=iri, promoted=False, reason=f"use_count={use_count} < 10")

    if source_count < 3:
        return PromotionResult(iri=iri, promoted=False, reason=f"source_count={source_count} < 3")

    if not approved:
        return PromotionResult(iri=iri, promoted=False, reason="awaiting_approval")

    suffix = iri.split("auros:provisional:", 1)[1]
    resolved = target_iri or f"auros:{suffix}"
    return PromotionResult(iri=resolved, promoted=True, reason="promoted")


def execute_promotion(
    old_iri: str,
    new_iri: str,
    conn: Any,
) -> int:
    """Migrate all axioms using old_iri to new_iri.

    Returns total number of rows updated across entities, relationships,
    and properties tables.
    """
    from mimir.persistence.graph_version import bump_graph_version

    count = 0
    r = conn.execute(
        "UPDATE entities SET entity_type = %s WHERE entity_type = %s",
        (new_iri, old_iri),
    )
    count += int(r.rowcount)
    r = conn.execute(
        "UPDATE relationships SET predicate = %s WHERE predicate = %s",
        (new_iri, old_iri),
    )
    count += int(r.rowcount)
    r = conn.execute(
        "UPDATE properties SET key = %s WHERE key = %s",
        (new_iri, old_iri),
    )
    count += int(r.rowcount)
    if count > 0:
        bump_graph_version(conn)
    return count


def is_provisional(iri: str) -> bool:
    return iri.startswith("auros:provisional:")


def provisional_suffix(iri: str) -> str:
    """Return the local name of a provisional IRI (part after the prefix)."""
    if not is_provisional(iri):
        raise ValueError(f"Not a provisional IRI: {iri!r}")
    return iri.split("auros:provisional:", 1)[1]
