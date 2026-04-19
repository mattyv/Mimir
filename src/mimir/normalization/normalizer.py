"""Text and IRI normalization utilities.

normalize_entity_name: strip whitespace, collapse runs, casefold for dedup index.
normalize_iri: ensure auros:provisional: prefix is lowercase; validate structure.
canonicalize_predicate: map aliases to canonical IRIs via the vocabulary.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_entity_name(name: str) -> str:
    """Return NFC-casefolded entity name for dedup index lookups."""
    nfc = unicodedata.normalize("NFC", name)
    collapsed = _WHITESPACE_RE.sub(" ", nfc).strip()
    return collapsed.casefold()


def normalize_iri(iri: str) -> str:
    """Lowercase the auros:provisional: prefix; leave other IRIs unchanged."""
    lower = iri.strip()
    if lower.startswith("AUROS:PROVISIONAL:") or lower.lower().startswith("auros:provisional:"):
        tail = lower[lower.index(":") + 1 :]
        tail = tail[tail.index(":") + 1 :]
        return f"auros:provisional:{tail}"
    return lower


def canonicalize_predicate(predicate: str, aliases: dict[str, str]) -> str:
    """Return the canonical IRI for *predicate*, falling back to the input."""
    return aliases.get(predicate, predicate)
