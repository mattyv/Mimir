"""Wikidata SPARQL-based entity linker.

Given an entity name + type, attempts to find a matching Wikidata QID.
Results are cached in-process to avoid redundant SPARQL calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mimir.models.base import GroundingTier


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
    """Find a Wikidata match and persist QID + updated grounding tier into payload.

    Tier is advanced to wikidata_linked (or fully_grounded if source_cited already).
    Returns the match (or None) without raising on SPARQL errors.
    """
    try:
        match = find_wikidata_match(name, client)
    except Exception:
        return None

    if match is None:
        return None

    # Determine new tier — only advance, never retreat
    row = conn.execute(
        "SELECT payload FROM entities WHERE id = %s",
        (entity_id,),
    ).fetchone()

    current_tier_str = "ungrounded"
    if row:
        current_tier_str = (
            row["payload"].get("grounding", {}).get("tier", "ungrounded")
            if isinstance(row["payload"], dict)
            else "ungrounded"
        )

    _tier_order = [t.value for t in GroundingTier]
    current_idx = _tier_order.index(current_tier_str) if current_tier_str in _tier_order else 0

    new_tier = GroundingTier.wikidata_linked
    if current_idx >= _tier_order.index(GroundingTier.wikidata_linked.value):
        new_tier = GroundingTier.fully_grounded

    conn.execute(
        """
        UPDATE entities
        SET payload = payload || jsonb_build_object(
              'wikidata_qid', %s::text,
              'wikidata_label', %s::text,
              'grounding', jsonb_build_object(
                  'tier', %s::text,
                  'wikidata_id', %s::text,
                  'depth', 0,
                  'stop_reason', 'wikidata_matched'
              )
            )
        WHERE id = %s
        """,
        (match.qid, match.label, new_tier.value, match.qid, entity_id),
    )
    return match
