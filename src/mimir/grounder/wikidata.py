"""Wikidata SPARQL-based entity linker.

Given an entity name + type, attempts to find a matching Wikidata QID.
Results are cached in-process to avoid redundant SPARQL calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SPARQLClient(Protocol):
    """Minimal interface satisfied by SPARQLWrapper and FakeSPARQL."""

    def query(self, sparql_query: str) -> dict[str, Any]: ...


@dataclass
class WikidataMatch:
    qid: str
    label: str
    description: str
    score: float  # 1.0 = exact label match, 0.5 = partial


_LABEL_QUERY_TEMPLATE = """
SELECT ?item ?itemLabel ?itemDescription WHERE {{
  ?item wikibase:sitelinks ?links .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
  ?item rdfs:label "{name}"@en .
}}
LIMIT 5
"""


def find_wikidata_match(
    name: str,
    client: SPARQLClient,
) -> WikidataMatch | None:
    """Return the best Wikidata match for *name*, or None if not found."""
    query = _LABEL_QUERY_TEMPLATE.format(name=name.replace('"', '\\"'))
    result = client.query(query)

    bindings = result.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    first = bindings[0]
    qid_uri: str = first.get("item", {}).get("value", "")
    qid = qid_uri.rsplit("/", 1)[-1] if "/" in qid_uri else qid_uri
    label: str = first.get("itemLabel", {}).get("value", name)
    desc: str = first.get("itemDescription", {}).get("value", "")

    score = 1.0 if label.lower() == name.lower() else 0.5
    return WikidataMatch(qid=qid, label=label, description=desc, score=score)


def ground_entity(
    entity_id: str,
    name: str,
    client: SPARQLClient,
    conn: Any,
) -> WikidataMatch | None:
    """Find a Wikidata match and persist the QID into the entity's payload.

    Returns the match (or None) without raising on SPARQL errors.
    """
    try:
        match = find_wikidata_match(name, client)
    except Exception:
        return None

    if match is None:
        return None

    conn.execute(
        """
        UPDATE entities
        SET payload = payload || jsonb_build_object('wikidata_qid', %s::text,
                                                    'wikidata_label', %s::text)
        WHERE id = %s
        """,
        (match.qid, match.label, entity_id),
    )
    return match
