"""Value objects shared by all Mimir node types.

Every axiom node embeds Grounding, Source, Temporal, and Visibility.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, model_validator

_TIER_ORDER = (
    "ungrounded",
    "source_cited",
    "schema_typed",
    "wikidata_linked",
    "fully_grounded",
)


class GroundingTier(StrEnum):
    """Ordered grounding quality tiers — higher is better."""

    ungrounded = "ungrounded"
    source_cited = "source_cited"
    schema_typed = "schema_typed"
    wikidata_linked = "wikidata_linked"
    fully_grounded = "fully_grounded"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, GroundingTier):
            return NotImplemented
        return _TIER_ORDER.index(self.value) < _TIER_ORDER.index(other.value)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, GroundingTier):
            return NotImplemented
        return _TIER_ORDER.index(self.value) <= _TIER_ORDER.index(other.value)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, GroundingTier):
            return NotImplemented
        return _TIER_ORDER.index(self.value) > _TIER_ORDER.index(other.value)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, GroundingTier):
            return NotImplemented
        return _TIER_ORDER.index(self.value) >= _TIER_ORDER.index(other.value)


class Grounding(BaseModel):
    """Grounding metadata attached to every axiom node."""

    tier: GroundingTier
    depth: int = 0
    stop_reason: str = ""
    wikidata_id: str | None = None
    schema_iri: str | None = None


class Source(BaseModel):
    """Provenance reference for an axiom."""

    type: Literal["confluence", "github", "slack", "interview", "code_analysis"]
    reference: str
    retrieved_at: datetime


class Temporal(BaseModel):
    """Bitemporal validity window for an axiom."""

    valid_from: datetime
    valid_until: datetime | None = None  # None = still true as of now
    superseded_by: str | None = None  # axiom_id that replaced this one

    @model_validator(mode="after")
    def _check_ordering(self) -> Temporal:
        if self.valid_until is not None and self.valid_from > self.valid_until:
            raise ValueError(
                f"valid_from ({self.valid_from}) must not be after "
                f"valid_until ({self.valid_until})"
            )
        return self


class Visibility(BaseModel):
    """ACL and sensitivity metadata propagated from source chunks."""

    acl: list[str]
    sensitivity: Literal["public", "internal", "restricted"]
