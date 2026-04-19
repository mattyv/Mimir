"""Extraction prompt templates for the crawler."""

from __future__ import annotations

_EXTRACTION_SYSTEM = """\
You are a knowledge-graph extraction engine.  Given a text chunk from a \
technical document, extract structured facts as JSON.

Return a JSON object with these keys (omit any that have no findings):
  entities   : list of {name, type, description}
  properties : list of {entity_name, key, value}
  relationships: list of {subject, predicate, object}
  observations : list of {entity_name, type, description}

Rules:
- entity type must be one of: schema:Organization, schema:Person,
  schema:SoftwareApplication, auros:TradingService, auros:TradingTeam,
  auros:Venue, auros:Strategy, auros:RiskSystem, auros:Connector
- predicate must be one of: schema:memberOf, auros:dependsOn,
  auros:independentOf, auros:owns, auros:connects, schema:name
- observation type must be one of: strength, risk, anti_pattern,
  maturity, smell, opportunity, inconsistency, functional_state
- Be conservative: only extract facts explicitly stated in the text.
- Return ONLY the JSON object, no markdown fences.
"""


def build_extraction_prompt(chunk_content: str) -> str:
    """Return the full prompt string for entity/relationship extraction."""
    return f"{_EXTRACTION_SYSTEM}\n\nText:\n{chunk_content}\n\nJSON:"
