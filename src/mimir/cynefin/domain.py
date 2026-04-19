"""Cynefin domain enumeration and classification logic.

The five Cynefin domains: Clear, Complicated, Complex, Chaotic, Confused.
Classification is heuristic-based: it examines observation types, relationship
counts, and confidence scores attached to an entity to assign a domain.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class CynefinDomain(StrEnum):
    clear = "clear"
    complicated = "complicated"
    complex = "complex"
    chaotic = "chaotic"
    confused = "confused"


# Observation types that push toward higher complexity domains
_CHAOTIC_SIGNALS = frozenset({"inconsistency", "anti_pattern"})
_COMPLEX_SIGNALS = frozenset({"risk", "smell", "opportunity"})
_COMPLICATED_SIGNALS = frozenset({"maturity", "functional_state"})


def classify(
    observations: list[dict[str, Any]],
    *,
    relationship_count: int = 0,
    avg_confidence: float = 1.0,
) -> CynefinDomain:
    """Classify an entity into a Cynefin domain.

    Rules (evaluated in order; first match wins):
    1. Any inconsistency or anti_pattern observation → Chaotic
    2. Low confidence (avg < 0.5) → Chaotic
    3. Any risk, smell, or opportunity observation → Complex
    4. High coupling (relationship_count >= 10) → Complex
    5. Maturity or functional_state observation → Complicated
    6. High relationship count (>= 5) → Complicated
    7. No observations and avg_confidence >= 0.9 → Clear
    8. Default → Confused
    """
    obs_types = {o.get("observation_type") or o.get("type", "") for o in observations}

    if obs_types & _CHAOTIC_SIGNALS:
        return CynefinDomain.chaotic
    if avg_confidence < 0.5:
        return CynefinDomain.chaotic
    if obs_types & _COMPLEX_SIGNALS:
        return CynefinDomain.complex
    if relationship_count >= 10:
        return CynefinDomain.complex
    if obs_types & _COMPLICATED_SIGNALS:
        return CynefinDomain.complicated
    if relationship_count >= 5:
        return CynefinDomain.complicated
    if not observations and avg_confidence >= 0.9:
        return CynefinDomain.clear
    return CynefinDomain.confused
