"""LLM-based entity/relationship extractor.

Takes a Chunk, calls the LLM, parses the JSON response, and returns
ExtractionResult — a structured collection of raw extracted facts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from mimir.adapters.base import Chunk
from mimir.crawler.prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


@dataclass
class RawEntity:
    name: str
    type: str
    description: str = ""


@dataclass
class RawProperty:
    entity_name: str
    key: str
    value: Any


@dataclass
class RawRelationship:
    subject: str
    predicate: str
    object: str


@dataclass
class RawObservation:
    entity_name: str
    type: str
    description: str


@dataclass
class ExtractionResult:
    chunk_id: str
    source_type: str
    entities: list[RawEntity] = field(default_factory=list)
    properties: list[RawProperty] = field(default_factory=list)
    relationships: list[RawRelationship] = field(default_factory=list)
    observations: list[RawObservation] = field(default_factory=list)
    parse_error: str | None = None


def extract(chunk: Chunk, llm: Any) -> ExtractionResult:
    """Call the LLM on *chunk.content* and parse the JSON extraction result.

    If the LLM returns invalid JSON, the result's ``parse_error`` field is
    set and all fact lists are empty — the caller decides how to handle this.
    """
    prompt = build_extraction_prompt(chunk.content)
    raw = llm.complete(prompt, temperature=0.0)
    result = ExtractionResult(chunk_id=chunk.id, source_type=chunk.source_type)
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        result.parse_error = str(exc)
        logger.warning("LLM returned non-JSON for chunk %s: %s", chunk.id, exc)
        return result

    for e in data.get("entities", []):
        result.entities.append(
            RawEntity(
                name=str(e.get("name", "")),
                type=str(e.get("type", "")),
                description=str(e.get("description", "")),
            )
        )
    for p in data.get("properties", []):
        result.properties.append(
            RawProperty(
                entity_name=str(p.get("entity_name", "")),
                key=str(p.get("key", "")),
                value=p.get("value"),
            )
        )
    for r in data.get("relationships", []):
        result.relationships.append(
            RawRelationship(
                subject=str(r.get("subject", "")),
                predicate=str(r.get("predicate", "")),
                object=str(r.get("object", "")),
            )
        )
    for o in data.get("observations", []):
        result.observations.append(
            RawObservation(
                entity_name=str(o.get("entity_name", "")),
                type=str(o.get("type", "")),
                description=str(o.get("description", "")),
            )
        )
    return result
