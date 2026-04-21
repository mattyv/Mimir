"""Wikidata SPARQL-based entity linker.

Given an entity name + type, attempts to find a matching Wikidata QID.
Results are cached in-process to avoid redundant SPARQL calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from mimir.grounder.cache import get_cached, put_cached
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


_ANCESTOR_QUERY_TEMPLATE = """
SELECT ?ancestor ?ancestorLabel WHERE {{
  {{ <{qid_uri}> wdt:P31 ?ancestor . }}
  UNION
  {{ <{qid_uri}> wdt:P279 ?ancestor . }}
  UNION
  {{ <{qid_uri}> wdt:P361 ?ancestor . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT 10
"""

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
    hit, cached_qid, cached_label = get_cached(name)
    if hit:
        if cached_qid is None:
            return None
        return WikidataMatch(qid=cached_qid, label=cached_label, description="", score=1.0)

    query = _LABEL_QUERY_TEMPLATE.format(name=name.replace('"', '\\"'))
    result = client.query(query)

    bindings = result.get("results", {}).get("bindings", [])
    if not bindings:
        put_cached(name, None)
        return None

    first = bindings[0]
    qid_uri: str = first.get("item", {}).get("value", "")
    qid = qid_uri.rsplit("/", 1)[-1] if "/" in qid_uri else qid_uri
    label: str = first.get("itemLabel", {}).get("value", name)
    desc: str = first.get("itemDescription", {}).get("value", "")

    score = 1.0 if label.lower() == name.lower() else 0.5
    put_cached(name, qid, label)
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


def find_ancestor_qids(
    qid: str,
    client: SPARQLClient,
    *,
    depth_cap: int = 4,
    budget: int = 25,
    _seen: set[str] | None = None,
    _depth: int = 0,
) -> list[tuple[str, str]]:
    """Recursively fetch parent concepts (instance_of / subclass_of / part_of).

    Returns list of (qid, label) tuples, excluding already-seen QIDs.
    Stops at depth_cap or when budget is exhausted.
    Cycle detection via _seen set of QIDs.
    """
    if _seen is None:
        _seen = {qid}
    if _depth >= depth_cap or budget <= 0:
        return []

    qid_uri = f"http://www.wikidata.org/entity/{qid}"
    try:
        result = client.query(_ANCESTOR_QUERY_TEMPLATE.format(qid_uri=qid_uri))
    except Exception:
        return []

    bindings = result.get("results", {}).get("bindings", [])
    ancestors: list[tuple[str, str]] = []
    remaining_budget = budget - 1

    for b in bindings:
        uri = b.get("ancestor", {}).get("value", "")
        label = b.get("ancestorLabel", {}).get("value", "")
        ancestor_qid = uri.rsplit("/", 1)[-1] if "/" in uri else uri
        if not ancestor_qid or ancestor_qid in _seen:
            continue
        _seen.add(ancestor_qid)
        ancestors.append((ancestor_qid, label))
        sub = find_ancestor_qids(
            ancestor_qid,
            client,
            depth_cap=depth_cap,
            budget=remaining_budget,
            _seen=_seen,
            _depth=_depth + 1,
        )
        remaining_budget -= len(sub)
        ancestors.extend(sub)
        if remaining_budget <= 0:
            break

    return ancestors


def ground_entity_recursive(
    entity_id: str,
    name: str,
    client: SPARQLClient,
    conn: Any,
    *,
    depth_cap: int = 4,
    budget: int = 25,
) -> list[tuple[str, str]]:
    """Ground entity and recursively fetch ancestor concepts.

    Calls ground_entity() first, then fetches ancestors and stores them
    in the entity payload as 'wikidata_ancestors' list.
    Returns list of (qid, label) ancestor tuples added.
    """
    match = ground_entity(entity_id, name, client, conn)
    if match is None:
        return []

    ancestors = find_ancestor_qids(
        match.qid,
        client,
        depth_cap=depth_cap,
        budget=budget,
    )
    if not ancestors:
        return []

    ancestor_list = [{"qid": q, "label": lbl} for q, lbl in ancestors]
    conn.execute(
        "UPDATE entities SET payload = payload || jsonb_build_object('wikidata_ancestors', %s::jsonb) WHERE id = %s",
        (json.dumps(ancestor_list), entity_id),
    )
    return ancestors
