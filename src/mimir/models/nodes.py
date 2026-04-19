"""The seven Mimir axiom node types.

Every node carries Grounding, Temporal, Visibility, and vocabulary_version.
IRI-typed fields use the IRI annotated type which enforces namespace validation
at construction time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from mimir.models.base import Grounding, Source, Temporal, Visibility
from mimir.models.iri import IRI


class Entity(BaseModel):
    """A concrete thing that exists in the modelled domain."""

    id: str
    type: IRI
    name: str
    description: str
    created_at: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    grounding: Grounding
    temporal: Temporal
    visibility: Visibility
    vocabulary_version: str


class Property(BaseModel):
    """An attribute of an entity."""

    entity_id: str
    key: IRI
    value: Any
    value_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: Source
    grounding: Grounding
    temporal: Temporal
    visibility: Visibility
    vocabulary_version: str


class Relationship(BaseModel):
    """A directed connection between two entities."""

    subject_id: str
    predicate: IRI
    object_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: Source
    grounding: Grounding
    temporal: Temporal
    visibility: Visibility
    vocabulary_version: str


class Observation(BaseModel):
    """A qualitative characterisation of an entity from a closed set of types."""

    entity_id: str
    type: Literal[
        "strength",
        "risk",
        "anti_pattern",
        "maturity",
        "smell",
        "opportunity",
        "inconsistency",
        "functional_state",
    ]
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: Source
    grounding: Grounding
    temporal: Temporal
    visibility: Visibility
    vocabulary_version: str


class Constraint(BaseModel):
    """A constraint (performance, legal, etc.) on an entity."""

    entity_id: str
    constraint_type: Literal["performance", "availability", "legal", "physical", "social"]
    condition: str
    threshold: Any
    source: Source
    grounding: Grounding
    temporal: Temporal
    visibility: Visibility
    vocabulary_version: str


class Process(BaseModel):
    """A multi-stage process or workflow."""

    id: str
    name: str
    stages: list[str]
    inputs: list[str]
    outputs: list[str]
    slo: str | None = None
    grounding: Grounding
    temporal: Temporal
    visibility: Visibility
    vocabulary_version: str


class Decision(BaseModel):
    """A recorded architectural or operational decision."""

    id: str
    what: str
    why: str
    tradeoffs: list[str]
    when: datetime
    who: list[str]
    source: Source
    grounding: Grounding
    temporal: Temporal
    visibility: Visibility
    vocabulary_version: str
