"""Phase 1 — SHACL shape generation and validation tests.

Uses rdflib to build minimal RDF graphs and validates them against generated
SHACL shapes.  Does NOT go through Pydantic models — tests the SHACL layer
directly.

RDF namespace conventions (must match shacl.py constants):
  SCHEMA = Namespace("https://schema.org/")
  AUROS  = Namespace("https://auros.internal/vocab/")
  MIMIR  = Namespace("https://mimir.internal/")
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from rdflib import XSD, Graph, Literal, Namespace, URIRef

from mimir.vocabulary.loader import Vocabulary, load_vocabulary
from mimir.vocabulary.shacl import (
    AUROS_NS,
    MIMIR_NS,
    SCHEMA_NS,
    ValidationResult,
    expand_iri,
    generate_shapes,
    validate_graph,
)

pytestmark = pytest.mark.phase1

_VOCAB_PATH = (
    Path(__file__).parent.parent.parent / "src" / "mimir" / "vocabulary" / "vocabulary.yaml"
)

SCHEMA = Namespace(SCHEMA_NS)
AUROS = Namespace(AUROS_NS)
MIMIR = Namespace(MIMIR_NS)

_VALID_FROM_LITERAL = Literal(
    datetime(2026, 4, 19, tzinfo=UTC).isoformat(),
    datatype=XSD.dateTime,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    return load_vocabulary(_VOCAB_PATH)


@pytest.fixture(scope="module")
def shapes_str(vocab: Vocabulary) -> str:
    return generate_shapes(vocab)


# ── Shape generation ───────────────────────────────────────────────────────────


def test_generate_shapes_with_provisional_entity_type() -> None:
    """generate_shapes uses full URI syntax for provisional entity types (shacl.py:_turtle_iri)."""
    from mimir.vocabulary.loader import EntityTypeEntry, PredicateEntry, Vocabulary

    vocab = Vocabulary(
        version="test",
        entity_types=(
            EntityTypeEntry(iri="schema:Organization", label="Org"),
            EntityTypeEntry(iri="auros:provisional:CustomThing", label="Custom"),
        ),
        predicates=(
            PredicateEntry(iri="schema:memberOf", label="member of", domain=("schema:Person",)),
        ),
    )
    shapes = generate_shapes(vocab)
    # Provisional IRIs must appear as full URIs (angle-bracket form) in Turtle
    assert f"<{AUROS_NS}provisional/CustomThing>" in shapes
    # The shapes must still be parseable Turtle
    g = Graph()
    g.parse(data=shapes, format="turtle")
    assert len(g) > 0


def test_generate_shapes_returns_string(shapes_str: str) -> None:
    assert isinstance(shapes_str, str)
    assert len(shapes_str) > 100


def test_shapes_parseable_as_turtle(shapes_str: str) -> None:
    """Generated SHACL shapes must be valid Turtle."""
    g = Graph()
    g.parse(data=shapes_str, format="turtle")
    assert len(g) > 0


def test_shapes_contains_entity_shape(shapes_str: str) -> None:
    assert "EntityShape" in shapes_str
    assert "sh:NodeShape" in shapes_str


def test_shapes_contains_valid_from_constraint(shapes_str: str) -> None:
    assert "mimir:validFrom" in shapes_str


def test_shacl_validates_all_core_predicates_have_shapes(
    vocab: Vocabulary, shapes_str: str
) -> None:
    """Every predicate with a non-empty domain has a corresponding shape."""
    for pred in vocab.predicates:
        if pred.domain:
            safe_local = pred.iri.replace(":", "_")
            assert f"Shape_{safe_local}" in shapes_str, (
                f"Expected SHACL shape for predicate {pred.iri!r} but not found in shapes."
            )


# ── IRI expansion ──────────────────────────────────────────────────────────────


def test_expand_iri_schema() -> None:
    assert expand_iri("schema:Organization") == f"{SCHEMA_NS}Organization"


def test_expand_iri_auros() -> None:
    assert expand_iri("auros:TradingService") == f"{AUROS_NS}TradingService"


def test_expand_iri_provisional() -> None:
    assert expand_iri("auros:provisional:foo") == f"{AUROS_NS}provisional/foo"


def test_expand_iri_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Cannot expand"):
        expand_iri("owl:Class")


# ── EntityShape: valid entities ────────────────────────────────────────────────


def _well_formed_entity_graph(entity_type_uri: URIRef) -> Graph:
    g = Graph()
    entity = URIRef(f"{MIMIR_NS}entity/test_org")
    g.add((entity, MIMIR.entityType, entity_type_uri))
    g.add((entity, MIMIR.validFrom, _VALID_FROM_LITERAL))
    return g


def test_shacl_accepts_well_formed_entity(shapes_str: str) -> None:
    g = _well_formed_entity_graph(SCHEMA.Organization)
    result = validate_graph(g, shapes_str)
    assert result.conforms, f"Well-formed entity should pass SHACL:\n{result.report}"


def test_shacl_accepts_auros_entity_type(shapes_str: str) -> None:
    g = _well_formed_entity_graph(AUROS.TradingService)
    result = validate_graph(g, shapes_str)
    assert result.conforms, f"auros:TradingService should pass SHACL:\n{result.report}"


# ── EntityShape: invalid type ──────────────────────────────────────────────────


def test_shacl_rejects_entity_with_unknown_iri_type(shapes_str: str) -> None:
    """An entity whose mimir:entityType is outside the core vocab must fail."""
    g = Graph()
    entity = URIRef(f"{MIMIR_NS}entity/bad_entity")
    # Use an IRI that is not in the vocabulary
    g.add((entity, MIMIR.entityType, URIRef("http://example.com/UnknownType")))
    g.add((entity, MIMIR.validFrom, _VALID_FROM_LITERAL))
    result = validate_graph(g, shapes_str)
    assert not result.conforms, "Unknown entity type should fail SHACL"


def test_shacl_rejects_free_string_type(shapes_str: str) -> None:
    """A Literal type value (free string) is not a valid IRI type."""
    g = Graph()
    entity = URIRef(f"{MIMIR_NS}entity/string_typed")
    # Literal "Organization" instead of the IRI schema:Organization
    g.add((entity, MIMIR.entityType, Literal("Organization")))
    g.add((entity, MIMIR.validFrom, _VALID_FROM_LITERAL))
    result = validate_graph(g, shapes_str)
    assert not result.conforms, "Free string entity type should fail SHACL"


# ── EntityShape: missing validFrom ────────────────────────────────────────────


def test_shacl_enforces_temporal_valid_from(shapes_str: str) -> None:
    """An entity without mimir:validFrom must fail validation."""
    g = Graph()
    entity = URIRef(f"{MIMIR_NS}entity/no_temporal")
    g.add((entity, MIMIR.entityType, SCHEMA.Organization))
    # Deliberately omit validFrom
    result = validate_graph(g, shapes_str)
    assert not result.conforms, "Entity missing validFrom should fail SHACL"


# ── Predicate domain shape ────────────────────────────────────────────────────


def test_shacl_accepts_person_as_memberOf_subject(shapes_str: str) -> None:
    """schema:Person as subject of schema:memberOf must pass domain check."""
    g = Graph()
    person = URIRef(f"{MIMIR_NS}entity/alice")
    org = URIRef(f"{MIMIR_NS}entity/acme")

    g.add((person, MIMIR.entityType, SCHEMA.Person))
    g.add((person, MIMIR.validFrom, _VALID_FROM_LITERAL))
    g.add((person, SCHEMA.memberOf, org))
    result = validate_graph(g, shapes_str)
    assert result.conforms, f"Person → memberOf should pass domain check:\n{result.report}"


def test_shacl_rejects_mismatched_predicate_endpoints(shapes_str: str) -> None:
    """schema:Organization as subject of schema:memberOf must fail domain check.

    schema:memberOf declares domain [schema:Person]; an Organization as subject
    violates that constraint.
    """
    g = Graph()
    org = URIRef(f"{MIMIR_NS}entity/acme")
    parent_org = URIRef(f"{MIMIR_NS}entity/holdco")

    g.add((org, MIMIR.entityType, SCHEMA.Organization))
    g.add((org, MIMIR.validFrom, _VALID_FROM_LITERAL))
    # Organization used as subject of memberOf (should be Person only)
    g.add((org, SCHEMA.memberOf, parent_org))
    result = validate_graph(g, shapes_str)
    assert not result.conforms, (
        "Organization as subject of schema:memberOf should fail domain check"
    )


# ── ValidationResult ──────────────────────────────────────────────────────────


def test_validation_result_is_frozen(shapes_str: str) -> None:
    g = _well_formed_entity_graph(SCHEMA.Organization)
    result = validate_graph(g, shapes_str)
    assert isinstance(result, ValidationResult)
    # frozen=True means we cannot reassign
    with pytest.raises((AttributeError, TypeError)):
        result.conforms = False  # type: ignore[misc]
