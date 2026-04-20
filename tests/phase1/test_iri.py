"""Phase 1 — IRI validation unit and property tests."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mimir.models.iri import (
    CORE_NAMESPACES,
    PROVISIONAL_PREFIX,
    extract_namespace,
    iri_roundtrip,
    is_provisional,
    validate_iri,
)

pytestmark = pytest.mark.phase1


# ── Valid IRI acceptance ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "iri",
    [
        "schema:Organization",
        "schema:Person",
        "schema:SoftwareApplication",
        "schema:Service",
        "auros:TradingService",
        "auros:TradingTeam",
        "auros:Strategy",
        "auros:dependsOn",
        "auros:provisional:trader_state",
        "auros:provisional:order_flow_metric",
        "auros:provisional:unknown_concept123",
    ],
)
def test_valid_iris_accepted(iri: str) -> None:
    assert validate_iri(iri) == iri


# ── Invalid IRI rejection ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_value",
    [
        "Organization",  # free string, no prefix
        "Organization:Person",  # unknown namespace
        "http://schema.org/Person",  # full HTTP IRI — not compact notation
        "",  # empty
        "schema:",  # empty local part
        "unknown:Thing",  # namespace not in core set
        "foo:bar",  # namespace not in core set
        "SCHEMA:Organization",  # uppercase namespace
        " schema:Organization",  # leading space
        "schema:Organization ",  # trailing space
    ],
)
def test_invalid_iris_rejected(bad_value: str) -> None:
    with pytest.raises(ValueError):
        validate_iri(bad_value)


def test_free_string_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid IRI syntax"):
        validate_iri("TradingService")


def test_unknown_namespace_rejected() -> None:
    with pytest.raises(ValueError, match="not in core namespaces"):
        validate_iri("owl:Class")


def test_empty_iri_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_iri("")


# ── Provisional IRI detection ──────────────────────────────────────────────────


def test_is_provisional_true_for_provisional_iri() -> None:
    assert is_provisional("auros:provisional:trader_state") is True


def test_is_provisional_false_for_core_auros() -> None:
    assert is_provisional("auros:TradingService") is False


def test_is_provisional_false_for_schema() -> None:
    assert is_provisional("schema:Organization") is False


def test_provisional_prefix_constant() -> None:
    assert PROVISIONAL_PREFIX == "auros:provisional:"


# ── Namespace extraction ───────────────────────────────────────────────────────


def test_extract_namespace_schema() -> None:
    assert extract_namespace("schema:Organization") == "schema"


def test_extract_namespace_auros() -> None:
    assert extract_namespace("auros:TradingService") == "auros"


def test_extract_namespace_provisional() -> None:
    assert extract_namespace("auros:provisional:foo") == "auros"


def test_extract_namespace_invalid_raises() -> None:
    with pytest.raises(ValueError):
        extract_namespace("NotAnIRI")


# ── Core namespaces constant ───────────────────────────────────────────────────


def test_core_namespaces_contains_schema_and_auros() -> None:
    assert "schema" in CORE_NAMESPACES
    assert "auros" in CORE_NAMESPACES


def test_core_namespaces_is_frozenset() -> None:
    assert isinstance(CORE_NAMESPACES, frozenset)


# ── Property-based tests ───────────────────────────────────────────────────────


_VALID_LOCAL = st.from_regex(r"[A-Za-z][A-Za-z0-9_]{0,30}", fullmatch=True)

_VALID_IRI_STRATEGY = st.one_of(
    st.builds(lambda local: f"schema:{local}", _VALID_LOCAL),
    st.builds(lambda local: f"auros:{local}", _VALID_LOCAL),
    st.builds(lambda local: f"auros:provisional:{local}", _VALID_LOCAL),
)


@given(iri=_VALID_IRI_STRATEGY)
@settings(max_examples=200)
def test_iri_roundtrip(iri: str) -> None:
    """validate_iri(iri) == iri for all valid IRIs; iri_roundtrip is idempotent."""
    validated = validate_iri(iri)
    assert validated == iri
    assert iri_roundtrip(validated) == iri


@given(iri=_VALID_IRI_STRATEGY)
def test_extract_namespace_always_in_core(iri: str) -> None:
    """extract_namespace always returns a core namespace for valid IRIs."""
    ns = extract_namespace(iri)
    assert ns in CORE_NAMESPACES


@given(iri=_VALID_IRI_STRATEGY)
def test_is_provisional_consistent_with_prefix(iri: str) -> None:
    """is_provisional is consistent with the PROVISIONAL_PREFIX string constant."""
    assert is_provisional(iri) == iri.startswith(PROVISIONAL_PREFIX)
