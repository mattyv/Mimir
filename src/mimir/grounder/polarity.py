"""Polarity enforcement — prevents auto-merge of opposite-polarity predicates.

This module provides a guard that raises PolarityViolation when a proposed
merge or write operation would assert both halves of a polarity pair.
"""

from __future__ import annotations

from pathlib import Path


class PolarityViolation(Exception):
    """Raised when an operation would violate a polarity constraint."""

    def __init__(self, predicate_a: str, predicate_b: str) -> None:
        super().__init__(
            f"Polarity violation: {predicate_a!r} and {predicate_b!r} "
            "are opposite-polarity predicates and must not be auto-merged."
        )
        self.predicate_a = predicate_a
        self.predicate_b = predicate_b


def _load_polarity_pairs() -> frozenset[frozenset[str]]:
    from mimir.vocabulary.loader import load_vocabulary

    vocab_path = Path(__file__).parent.parent / "vocabulary" / "vocabulary.yaml"
    vocab = load_vocabulary(vocab_path)
    pairs: set[frozenset[str]] = set()
    for pred in vocab.predicates:
        opp = vocab.get_polarity_opposite(pred.iri)
        if opp:
            pairs.add(frozenset([pred.iri, opp]))
    return frozenset(pairs)


_POLARITY_PAIRS: frozenset[frozenset[str]] | None = None


def _pairs() -> frozenset[frozenset[str]]:
    global _POLARITY_PAIRS
    if _POLARITY_PAIRS is None:
        _POLARITY_PAIRS = _load_polarity_pairs()
    return _POLARITY_PAIRS


def assert_no_polarity_conflict(predicate_a: str, predicate_b: str) -> None:
    """Raise PolarityViolation if *predicate_a* and *predicate_b* are opposites."""
    if frozenset([predicate_a, predicate_b]) in _pairs():
        raise PolarityViolation(predicate_a, predicate_b)


def are_polarity_opposites(predicate_a: str, predicate_b: str) -> bool:
    """Return True if the two predicates form a polarity pair."""
    return frozenset([predicate_a, predicate_b]) in _pairs()
