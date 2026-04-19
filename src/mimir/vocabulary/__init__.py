"""Mimir vocabulary — core IRI set and provisional extension tracking."""

from mimir.vocabulary.loader import (
    EntityTypeEntry,
    PredicateEntry,
    ProvisionalTracker,
    Vocabulary,
    load_vocabulary,
)

__all__ = [
    "EntityTypeEntry",
    "PredicateEntry",
    "ProvisionalTracker",
    "Vocabulary",
    "load_vocabulary",
]
