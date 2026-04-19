"""Generate SHACL shapes from the core vocabulary and validate RDF graphs.

The generated shapes enforce:
  1. EntityShape  — every node with mimir:entityType must have a core-tier type
                    and mimir:validFrom.
  2. Per-predicate domain shapes — subjects of predicates with declared domains
     must have an entity type in that domain.

RDF namespace conventions (must match data graph builders):
  schema:  https://schema.org/
  auros:   https://auros.internal/vocab/
  mimir:   https://mimir.internal/
"""

from __future__ import annotations

from dataclasses import dataclass

from pyshacl import validate as pyshacl_validate
from rdflib import Graph

from mimir.vocabulary.loader import Vocabulary

# Canonical namespace URIs — callers building data graphs must use these.
SCHEMA_NS = "https://schema.org/"
AUROS_NS = "https://auros.internal/vocab/"
MIMIR_NS = "https://mimir.internal/"

_PREFIXES = f"""\
@prefix sh:     <http://www.w3.org/ns/shacl#> .
@prefix schema: <{SCHEMA_NS}> .
@prefix auros:  <{AUROS_NS}> .
@prefix mimir:  <{MIMIR_NS}> .
@prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .
"""


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a SHACL validation run."""

    conforms: bool
    report: str


def expand_iri(iri: str) -> str:
    """Expand a compact IRI (schema:X, auros:X, auros:provisional:X) to a full URI."""
    if iri.startswith("schema:"):
        return SCHEMA_NS + iri[7:]
    if iri.startswith("auros:provisional:"):
        return AUROS_NS + "provisional/" + iri[18:]
    if iri.startswith("auros:"):
        return AUROS_NS + iri[6:]
    raise ValueError(f"Cannot expand IRI with unknown namespace: {iri!r}")


def _turtle_iri(iri: str) -> str:
    """Return the Turtle compact-IRI form suitable for inclusion in generated shapes."""
    # Provisional IRIs can't be expressed as a single compact IRI in Turtle
    # (auros:provisional:X would require a second prefix), so we use full URI syntax.
    if iri.startswith("auros:provisional:"):
        return f"<{expand_iri(iri)}>"
    return iri  # schema:X or auros:X — covered by declared prefixes


def generate_shapes(vocabulary: Vocabulary) -> str:
    """Return a Turtle-format SHACL shapes document derived from *vocabulary*.

    The resulting string is suitable for passing to pyshacl as the shacl_graph.
    """
    entity_iris = " ".join(_turtle_iri(e.iri) for e in vocabulary.entity_types)

    lines: list[str] = [_PREFIXES, ""]

    # ── EntityShape ───────────────────────────────────────────────────────────
    lines += [
        "mimir:EntityShape",
        "    a sh:NodeShape ;",
        "    sh:targetSubjectsOf mimir:entityType ;",
        "    sh:property [",
        "        sh:path mimir:entityType ;",
        f"        sh:in ({entity_iris}) ;",
        "        sh:minCount 1 ;",
        '        sh:message "Entity type must be a core-tier IRI" ;',
        "    ] ;",
        "    sh:property [",
        "        sh:path mimir:validFrom ;",
        "        sh:minCount 1 ;",
        "        sh:datatype xsd:dateTime ;",
        '        sh:message "mimir:validFrom (temporal.valid_from) is required" ;',
        "    ] ;",
        ".",
        "",
    ]

    # ── Per-predicate domain shapes ───────────────────────────────────────────
    for pred in vocabulary.predicates:
        if not pred.domain:
            continue
        pred_turtle = _turtle_iri(pred.iri)
        # Shape ID: replace colons with underscores for a valid prefixed name
        safe_local = pred.iri.replace(":", "_")
        shape_id = f"mimir:Shape_{safe_local}"
        domain_iris = " ".join(_turtle_iri(d) for d in pred.domain)

        lines += [
            f"{shape_id}",
            "    a sh:NodeShape ;",
            f"    sh:targetSubjectsOf {pred_turtle} ;",
            "    sh:property [",
            "        sh:path mimir:entityType ;",
            f"        sh:in ({domain_iris}) ;",
            f'        sh:message "Subject of {pred.iri} must have a valid domain type" ;',
            "    ] ;",
            ".",
            "",
        ]

    return "\n".join(lines)


def validate_graph(data_graph: Graph, shapes_str: str) -> ValidationResult:
    """Validate *data_graph* against the SHACL shapes in *shapes_str*.

    Args:
        data_graph: An rdflib Graph containing the axiom data.
        shapes_str: A Turtle-format SHACL shapes document (from generate_shapes).

    Returns:
        ValidationResult with conforms=True if the graph satisfies all shapes.
    """
    shapes_graph = Graph().parse(data=shapes_str, format="turtle")
    conforms, _, report_text = pyshacl_validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
        meta_shacl=False,
    )
    return ValidationResult(conforms=bool(conforms), report=str(report_text))
