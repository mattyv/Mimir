"""Phase 1 — Vocabulary loader unit and property tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mimir.vocabulary.loader import (
    PROMOTION_MIN_SOURCES,
    PROMOTION_MIN_USES,
    ProvisionalTracker,
    Vocabulary,
    load_vocabulary,
)

pytestmark = pytest.mark.phase1

_VOCAB_PATH = Path(__file__).parent.parent.parent / "src" / "mimir" / "vocabulary" / "vocabulary.yaml"


# ── load_vocabulary ────────────────────────────────────────────────────────────


def test_vocabulary_yaml_loads_without_error() -> None:
    vocab = load_vocabulary(_VOCAB_PATH)
    assert vocab is not None


def test_vocabulary_has_version() -> None:
    vocab = load_vocabulary(_VOCAB_PATH)
    assert vocab.version == "0.1.0"


def test_vocabulary_contains_expected_entity_types(core_vocabulary: Vocabulary) -> None:
    iris = {e.iri for e in core_vocabulary.entity_types}
    assert "schema:Organization" in iris
    assert "schema:Person" in iris
    assert "auros:TradingService" in iris
    assert "auros:TradingTeam" in iris


def test_vocabulary_contains_expected_predicates(core_vocabulary: Vocabulary) -> None:
    iris = {p.iri for p in core_vocabulary.predicates}
    assert "schema:memberOf" in iris
    assert "auros:dependsOn" in iris
    assert "auros:independentOf" in iris
    assert "auros:quotesOn" in iris


def test_vocabulary_entity_type_count(core_vocabulary: Vocabulary) -> None:
    assert len(core_vocabulary.entity_types) >= 18


def test_vocabulary_predicate_count(core_vocabulary: Vocabulary) -> None:
    assert len(core_vocabulary.predicates) >= 20


def test_vocabulary_is_core_iri_entity(core_vocabulary: Vocabulary) -> None:
    assert core_vocabulary.is_core_iri("schema:Organization") is True


def test_vocabulary_is_core_iri_predicate(core_vocabulary: Vocabulary) -> None:
    assert core_vocabulary.is_core_iri("auros:dependsOn") is True


def test_vocabulary_is_core_iri_false_for_provisional(core_vocabulary: Vocabulary) -> None:
    assert core_vocabulary.is_core_iri("auros:provisional:unknown") is False


def test_vocabulary_is_core_entity_type(core_vocabulary: Vocabulary) -> None:
    assert core_vocabulary.is_core_entity_type("auros:TradingService") is True
    assert core_vocabulary.is_core_entity_type("auros:dependsOn") is False


def test_vocabulary_is_core_predicate(core_vocabulary: Vocabulary) -> None:
    assert core_vocabulary.is_core_predicate("auros:dependsOn") is True
    assert core_vocabulary.is_core_predicate("auros:TradingService") is False


# ── Subclass declarations ──────────────────────────────────────────────────────


def test_trading_service_subclass_of_software_application(core_vocabulary: Vocabulary) -> None:
    entry = next(e for e in core_vocabulary.entity_types if e.iri == "auros:TradingService")
    assert entry.subclass_of == "schema:SoftwareApplication"


def test_trading_team_subclass_of_organization(core_vocabulary: Vocabulary) -> None:
    entry = next(e for e in core_vocabulary.entity_types if e.iri == "auros:TradingTeam")
    assert entry.subclass_of == "schema:Organization"


# ── Polarity declarations ──────────────────────────────────────────────────────


def test_polarity_pair_declared(core_vocabulary: Vocabulary) -> None:
    """auros:dependsOn and auros:independentOf must be declared as opposites."""
    dep = core_vocabulary.get_polarity_opposite("auros:dependsOn")
    indep = core_vocabulary.get_polarity_opposite("auros:independentOf")
    assert dep is None or dep == "auros:independentOf"  # dependsOn may not list its opposite
    assert indep == "auros:dependsOn"


def test_polarity_opposite_never_auto_merged(core_vocabulary: Vocabulary) -> None:
    """Opposite polarity predicates are not equal (sanity check)."""
    assert "auros:dependsOn" != "auros:independentOf"
    assert core_vocabulary.get_polarity_opposite("auros:independentOf") == "auros:dependsOn"


def test_get_polarity_opposite_none_for_no_declaration(core_vocabulary: Vocabulary) -> None:
    assert core_vocabulary.get_polarity_opposite("schema:memberOf") is None


def test_get_polarity_opposite_none_for_unknown_iri(core_vocabulary: Vocabulary) -> None:
    """IRI not in the vocabulary at all returns None (loop completes without match)."""
    assert core_vocabulary.get_polarity_opposite("auros:nonExistentPredicate") is None


def test_get_predicate_returns_none_for_unknown_iri(core_vocabulary: Vocabulary) -> None:
    """IRI not in the vocabulary at all returns None from get_predicate."""
    assert core_vocabulary.get_predicate("auros:nonExistentPredicate") is None


def test_load_vocabulary_bad_polarity_opposite_raises(tmp_path: Path) -> None:
    """load_vocabulary raises ValueError when polarity_opposite references a missing IRI."""
    import yaml as _yaml

    bad_yaml = {
        "version": "0.1.0",
        "entity_types": [{"iri": "schema:Organization", "label": "Org"}],
        "predicates": [
            {
                "iri": "auros:dependsOn",
                "label": "depends on",
                "polarity": "positive",
                "polarity_opposite": "auros:nonExistent",  # not in vocab
            }
        ],
    }
    p = tmp_path / "bad_vocab.yaml"
    p.write_text(_yaml.dump(bad_yaml))
    with pytest.raises(ValueError, match="not in the vocabulary"):
        load_vocabulary(p)


# ── Domain / range on predicates ───────────────────────────────────────────────


def test_member_of_has_person_domain(core_vocabulary: Vocabulary) -> None:
    pred = core_vocabulary.get_predicate("schema:memberOf")
    assert pred is not None
    assert "schema:Person" in pred.domain


def test_quotes_on_has_trading_service_domain(core_vocabulary: Vocabulary) -> None:
    pred = core_vocabulary.get_predicate("auros:quotesOn")
    assert pred is not None
    assert "auros:TradingService" in pred.domain


# ── ProvisionalTracker ─────────────────────────────────────────────────────────


def test_provisional_tracker_initial_state() -> None:
    tracker = ProvisionalTracker()
    assert tracker.use_count == 0
    assert len(tracker.source_ids) == 0
    assert tracker.is_promotion_eligible is False


def test_provisional_tracker_counts_uses() -> None:
    tracker = ProvisionalTracker()
    for i in range(5):
        tracker.record_use(f"source_{i}")
    assert tracker.use_count == 5
    assert len(tracker.source_ids) == 5


def test_provisional_tracker_deduplicates_sources() -> None:
    tracker = ProvisionalTracker()
    for _ in range(10):
        tracker.record_use("source_a")
    assert tracker.use_count == 10
    assert len(tracker.source_ids) == 1  # still one unique source


def test_provisional_tracker_promotion_eligibility_uses_threshold() -> None:
    tracker = ProvisionalTracker()
    # Meet source count but not use count
    for i in range(PROMOTION_MIN_SOURCES + 1):
        tracker.record_use(f"source_{i}")
    assert tracker.use_count < PROMOTION_MIN_USES
    assert tracker.is_promotion_eligible is False


def test_provisional_tracker_promotion_eligibility_sources_threshold() -> None:
    tracker = ProvisionalTracker()
    # Meet use count but not source diversity
    for _ in range(PROMOTION_MIN_USES + 1):
        tracker.record_use("single_source")
    assert tracker.use_count >= PROMOTION_MIN_USES
    assert len(tracker.source_ids) < PROMOTION_MIN_SOURCES
    assert tracker.is_promotion_eligible is False


def test_provisional_tracker_promotion_eligible_when_both_met() -> None:
    tracker = ProvisionalTracker()
    for i in range(PROMOTION_MIN_SOURCES):
        for _ in range(PROMOTION_MIN_USES // PROMOTION_MIN_SOURCES + 1):
            tracker.record_use(f"source_{i}")
    assert tracker.is_promotion_eligible is True


def test_promotion_thresholds_match_spec() -> None:
    assert PROMOTION_MIN_USES == 10
    assert PROMOTION_MIN_SOURCES == 3


# ── Polarity property test ─────────────────────────────────────────────────────


@given(iri=st.sampled_from(["auros:dependsOn", "auros:independentOf"]))
def test_polarity_opposite_never_same_iri(iri: str) -> None:
    """A predicate's polarity opposite is never itself."""
    vocab = load_vocabulary(_VOCAB_PATH)
    opp = vocab.get_polarity_opposite(iri)
    if opp is not None:
        assert opp != iri


# ── Error handling ─────────────────────────────────────────────────────────────


def test_load_vocabulary_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_vocabulary("/nonexistent/vocabulary.yaml")
