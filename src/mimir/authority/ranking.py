"""Source authority ranking — trust scores per source type."""

from __future__ import annotations

from mimir.adapters.base import SourceType

_TRUST_SCORES: dict[SourceType, float] = {
    "code_analysis": 1.0,
    "github": 0.9,
    "confluence": 0.8,
    "interview": 0.7,
    "slack": 0.5,
}


def trust_score(source_type: SourceType) -> float:
    """Return a [0, 1] trust score for *source_type*."""
    return _TRUST_SCORES[source_type]


def higher_authority(a: SourceType, b: SourceType) -> SourceType:
    """Return whichever source type has higher trust; *a* wins on tie."""
    return a if trust_score(a) >= trust_score(b) else b
