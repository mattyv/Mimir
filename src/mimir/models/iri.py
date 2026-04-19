"""IRI validation and utilities for Mimir's two-tier vocabulary.

Accepted forms:
  schema:<LocalName>          — schema.org namespace
  auros:<LocalName>           — core auros namespace
  auros:provisional:<name>    — provisional/quarantined extension
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator

CORE_NAMESPACES: frozenset[str] = frozenset({"schema", "auros"})
PROVISIONAL_PREFIX = "auros:provisional:"

# Matches "prefix:rest" where prefix is lowercase alpha+digits and rest is non-empty
_PREFIX_RE = re.compile(r"^([a-z][a-z0-9]*):(.+)$", re.ASCII)


def validate_iri(value: str) -> str:
    """Validate that *value* is an acceptable IRI in our two-tier vocabulary.

    Raises ValueError for bare strings, unknown namespaces, or empty local parts.
    """
    if not value:
        raise ValueError("IRI cannot be empty")
    if value != value.strip():
        raise ValueError(f"IRI must not have leading or trailing whitespace: {value!r}")
    m = _PREFIX_RE.match(value)
    if not m:
        raise ValueError(
            f"Invalid IRI syntax (expected 'prefix:local'): {value!r}"
        )
    ns = m.group(1)
    if ns not in CORE_NAMESPACES:
        raise ValueError(
            f"IRI namespace {ns!r} not in core namespaces "
            f"{sorted(CORE_NAMESPACES)}. "
            f"Use auros:provisional:<name> for extension concepts."
        )
    return value


def is_provisional(iri: str) -> bool:
    """Return True if *iri* is a provisional extension IRI."""
    return iri.startswith(PROVISIONAL_PREFIX)


def extract_namespace(iri: str) -> str:
    """Return the namespace prefix of a validated IRI (e.g. 'schema' or 'auros')."""
    m = _PREFIX_RE.match(iri)
    if not m:
        raise ValueError(f"Cannot extract namespace from invalid IRI: {iri!r}")
    return m.group(1)


def iri_roundtrip(iri: str) -> str:
    """Parse and re-serialize an IRI. Used in property-based tests."""
    return validate_iri(iri)


#: Annotated str type accepted by Pydantic that enforces IRI validation.
IRI = Annotated[str, AfterValidator(validate_iri)]
