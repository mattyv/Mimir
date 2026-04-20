"""Load and validate the core vocabulary from vocabulary.yaml.

Provides:
  load_vocabulary(path)  — parse YAML, validate IRIs, return Vocabulary
  Vocabulary             — immutable snapshot of the core IRI set
  ProvisionalTracker     — per-IRI use counter driving promotion eligibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mimir.models.iri import validate_iri

# Minimum thresholds for provisional → core promotion
PROMOTION_MIN_USES = 10
PROMOTION_MIN_SOURCES = 3


@dataclass(frozen=True)
class EntityTypeEntry:
    """One entry from the entity_types section of vocabulary.yaml."""

    iri: str
    label: str
    subclass_of: str | None = None


@dataclass(frozen=True)
class PredicateEntry:
    """One entry from the predicates section of vocabulary.yaml."""

    iri: str
    label: str
    domain: tuple[str, ...] = field(default_factory=tuple)
    range_types: tuple[str, ...] = field(default_factory=tuple)
    polarity: str | None = None
    polarity_opposite: str | None = None


@dataclass(frozen=True)
class Vocabulary:
    """Immutable snapshot of the core vocabulary for a given version."""

    version: str
    entity_types: tuple[EntityTypeEntry, ...]
    predicates: tuple[PredicateEntry, ...]

    # Derived index (built once, not stored in frozen fields)
    _entity_iris: frozenset[str] = field(init=False, compare=False, repr=False)
    _predicate_iris: frozenset[str] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        # Bypass frozen to set derived fields
        object.__setattr__(
            self,
            "_entity_iris",
            frozenset(e.iri for e in self.entity_types),
        )
        object.__setattr__(
            self,
            "_predicate_iris",
            frozenset(p.iri for p in self.predicates),
        )

    def is_core_iri(self, iri: str) -> bool:
        """Return True if *iri* is in the core entity or predicate set."""
        return iri in self._entity_iris or iri in self._predicate_iris

    def is_core_entity_type(self, iri: str) -> bool:
        """Return True if *iri* is a core entity type IRI."""
        return iri in self._entity_iris

    def is_core_predicate(self, iri: str) -> bool:
        """Return True if *iri* is a core predicate IRI."""
        return iri in self._predicate_iris

    def get_polarity_opposite(self, iri: str) -> str | None:
        """Return the IRI of the polarity-opposite predicate, or None."""
        for pred in self.predicates:
            if pred.iri == iri:
                return pred.polarity_opposite
        return None

    def get_predicate(self, iri: str) -> PredicateEntry | None:
        """Return the PredicateEntry for *iri*, or None."""
        for pred in self.predicates:
            if pred.iri == iri:
                return pred
        return None


@dataclass
class ProvisionalTracker:
    """Tracks usage of a single provisional IRI for promotion eligibility."""

    use_count: int = 0
    source_ids: set[str] = field(default_factory=set)

    def record_use(self, source_id: str) -> None:
        """Record one use of the provisional IRI from *source_id*."""
        self.use_count += 1
        self.source_ids.add(source_id)

    @property
    def is_promotion_eligible(self) -> bool:
        """True when use count and source diversity meet promotion thresholds."""
        return (
            self.use_count >= PROMOTION_MIN_USES and len(self.source_ids) >= PROMOTION_MIN_SOURCES
        )


def load_vocabulary(path: str | Path) -> Vocabulary:
    """Parse *path* (vocabulary.yaml), validate all IRIs, and return a Vocabulary.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if any IRI fails validation or polarity pairs are inconsistent.
    """
    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    version: str = raw["version"]

    entity_types: list[EntityTypeEntry] = []
    for entry in raw.get("entity_types", []):
        iri: str = entry["iri"]
        validate_iri(iri)
        subclass_of: str | None = entry.get("subclass_of")
        if subclass_of is not None:
            validate_iri(subclass_of)
        entity_types.append(EntityTypeEntry(iri=iri, label=entry["label"], subclass_of=subclass_of))

    predicates: list[PredicateEntry] = []
    for entry in raw.get("predicates", []):
        iri = entry["iri"]
        validate_iri(iri)
        domain: list[str] = entry.get("domain", [])
        range_raw: list[str] = entry.get("range", [])
        for d in domain:
            validate_iri(d)
        for r in range_raw:
            validate_iri(r)
        polarity_opp: str | None = entry.get("polarity_opposite")
        if polarity_opp is not None:
            validate_iri(polarity_opp)
        predicates.append(
            PredicateEntry(
                iri=iri,
                label=entry["label"],
                domain=tuple(domain),
                range_types=tuple(range_raw),
                polarity=entry.get("polarity"),
                polarity_opposite=polarity_opp,
            )
        )

    _validate_polarity_pairs(predicates)

    return Vocabulary(
        version=version,
        entity_types=tuple(entity_types),
        predicates=tuple(predicates),
    )


def _validate_polarity_pairs(predicates: list[PredicateEntry]) -> None:
    """Verify that polarity_opposite references are symmetric and declared."""
    iri_set = {p.iri for p in predicates}
    for pred in predicates:
        if pred.polarity_opposite is None:
            continue
        if pred.polarity_opposite not in iri_set:
            raise ValueError(
                f"Predicate {pred.iri!r} declares polarity_opposite "
                f"{pred.polarity_opposite!r} which is not in the vocabulary."
            )
