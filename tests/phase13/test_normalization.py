"""Phase 13 — normalization and provisional IRI promotion tests."""

from __future__ import annotations

import pytest

from mimir.normalization.normalizer import (
    canonicalize_predicate,
    normalize_entity_name,
    normalize_iri,
)
from mimir.normalization.promotion import (
    PromotionResult,
    is_provisional,
    promote_provisional,
    provisional_suffix,
)

# ── normalize_entity_name ─────────────────────────────────────────────────────


@pytest.mark.phase13
def test_normalize_entity_name_casefolding() -> None:
    assert normalize_entity_name("OMMS") == "omms"


@pytest.mark.phase13
def test_normalize_entity_name_strips_whitespace() -> None:
    assert normalize_entity_name("  hello   world  ") == "hello world"


@pytest.mark.phase13
def test_normalize_entity_name_collapses_spaces() -> None:
    assert normalize_entity_name("a  b   c") == "a b c"


@pytest.mark.phase13
def test_normalize_entity_name_nfc() -> None:
    # é can be NFD (e + combining acute) or NFC (é precomposed)
    nfd = "e\u0301"
    nfc = "\xe9"
    assert normalize_entity_name(nfd) == normalize_entity_name(nfc)


@pytest.mark.phase13
def test_normalize_entity_name_empty() -> None:
    assert normalize_entity_name("") == ""


# ── normalize_iri ─────────────────────────────────────────────────────────────


@pytest.mark.phase13
def test_normalize_iri_leaves_core_unchanged() -> None:
    assert normalize_iri("auros:dependsOn") == "auros:dependsOn"


@pytest.mark.phase13
def test_normalize_iri_lowercases_provisional_prefix() -> None:
    assert normalize_iri("AUROS:PROVISIONAL:trader_state") == "auros:provisional:trader_state"


@pytest.mark.phase13
def test_normalize_iri_strips_whitespace() -> None:
    assert normalize_iri("  auros:dependsOn  ") == "auros:dependsOn"


@pytest.mark.phase13
def test_normalize_iri_provisional_already_lowercase() -> None:
    iri = "auros:provisional:my_type"
    assert normalize_iri(iri) == iri


# ── canonicalize_predicate ────────────────────────────────────────────────────


@pytest.mark.phase13
def test_canonicalize_predicate_maps_alias() -> None:
    aliases = {"depends_on": "auros:dependsOn"}
    assert canonicalize_predicate("depends_on", aliases) == "auros:dependsOn"


@pytest.mark.phase13
def test_canonicalize_predicate_passthrough_unknown() -> None:
    assert canonicalize_predicate("auros:dependsOn", {}) == "auros:dependsOn"


@pytest.mark.phase13
def test_canonicalize_predicate_empty_aliases() -> None:
    assert canonicalize_predicate("schema:name", {}) == "schema:name"


# ── promote_provisional ───────────────────────────────────────────────────────


@pytest.mark.phase13
def test_promote_provisional_success() -> None:
    result = promote_provisional(
        "auros:provisional:trader_state",
        use_count=15,
        source_count=5,
        approved=True,
    )
    assert result.promoted
    assert result.iri == "auros:trader_state"
    assert result.reason == "promoted"


@pytest.mark.phase13
def test_promote_provisional_custom_target_iri() -> None:
    result = promote_provisional(
        "auros:provisional:trader_state",
        use_count=15,
        source_count=5,
        approved=True,
        target_iri="auros:TraderState",
    )
    assert result.iri == "auros:TraderState"


@pytest.mark.phase13
def test_promote_provisional_fails_not_provisional() -> None:
    result = promote_provisional("auros:dependsOn", use_count=100, source_count=10, approved=True)
    assert not result.promoted
    assert result.reason == "not_provisional"


@pytest.mark.phase13
def test_promote_provisional_fails_low_use_count() -> None:
    result = promote_provisional(
        "auros:provisional:x", use_count=5, source_count=5, approved=True
    )
    assert not result.promoted
    assert "use_count=5" in result.reason


@pytest.mark.phase13
def test_promote_provisional_fails_low_source_count() -> None:
    result = promote_provisional(
        "auros:provisional:x", use_count=15, source_count=2, approved=True
    )
    assert not result.promoted
    assert "source_count=2" in result.reason


@pytest.mark.phase13
def test_promote_provisional_fails_not_approved() -> None:
    result = promote_provisional(
        "auros:provisional:x", use_count=15, source_count=5, approved=False
    )
    assert not result.promoted
    assert result.reason == "awaiting_approval"


@pytest.mark.phase13
def test_promote_provisional_result_dataclass() -> None:
    r = PromotionResult(iri="auros:X", promoted=True, reason="promoted")
    assert r.promoted


# ── is_provisional / provisional_suffix ──────────────────────────────────────


@pytest.mark.phase13
def test_is_provisional_true() -> None:
    assert is_provisional("auros:provisional:my_type")


@pytest.mark.phase13
def test_is_provisional_false() -> None:
    assert not is_provisional("auros:dependsOn")


@pytest.mark.phase13
def test_provisional_suffix_extracts_correctly() -> None:
    assert provisional_suffix("auros:provisional:trader_state") == "trader_state"


@pytest.mark.phase13
def test_provisional_suffix_raises_on_non_provisional() -> None:
    with pytest.raises(ValueError):
        provisional_suffix("auros:dependsOn")
