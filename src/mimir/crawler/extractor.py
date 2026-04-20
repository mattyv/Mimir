"""LLM-based entity/relationship/observation extractor.

Two extraction modes:
  extract()            — single-pass combined prompt (backward compatible)
  extract_three_pass() — three focused LLM calls: spine -> grounding -> observations

Three-pass mode is the preferred pipeline path. Passes 2 and 3 degrade
gracefully on LLM errors — only Pass 1 is fatal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from mimir.adapters.base import Chunk
from mimir.crawler.prompts import (
    build_extraction_prompt,
    build_grounding_candidates_prompt,
    build_observations_prompt,
    build_spine_prompt,
)

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
class RawGroundingCandidate:
    entity_name: str
    wikidata_qid: str
    label: str
    category: str


@dataclass
class ExtractionResult:
    chunk_id: str
    source_type: str
    entities: list[RawEntity] = field(default_factory=list)
    properties: list[RawProperty] = field(default_factory=list)
    relationships: list[RawRelationship] = field(default_factory=list)
    observations: list[RawObservation] = field(default_factory=list)
    grounding_candidates: list[RawGroundingCandidate] = field(default_factory=list)
    parse_error: str | None = None


def _parse_spine_data(data: dict[str, Any], result: ExtractionResult) -> None:
    """Populate entities, properties, relationships from parsed JSON."""
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


# -- Three-pass functions ------------------------------------------------------


def extract_spine(chunk: Chunk, llm: Any) -> ExtractionResult:
    """Pass 1: extract entities, properties, relationships. Fatal on parse error."""
    prompt = build_spine_prompt(chunk.content)
    raw = llm.complete(prompt, temperature=0.0)
    result = ExtractionResult(chunk_id=chunk.id, source_type=chunk.source_type)
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        result.parse_error = str(exc)
        logger.warning("LLM spine pass returned non-JSON for chunk %s: %s", chunk.id, exc)
        return result
    _parse_spine_data(data, result)
    return result


def extract_grounding_candidates(
    chunk: Chunk,
    llm: Any,
    entities: list[RawEntity],
) -> list[RawGroundingCandidate]:
    """Pass 2: match entities to external concepts. Returns empty list on any error."""
    if not entities:
        return []
    entity_list = "\n".join(f"- {e.name} ({e.type})" for e in entities)
    prompt = build_grounding_candidates_prompt(chunk.content, entity_list)
    raw = llm.complete(prompt, temperature=0.0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [
        RawGroundingCandidate(
            entity_name=str(c.get("entity_name", "")),
            wikidata_qid=str(c.get("wikidata_qid", "")),
            label=str(c.get("label", "")),
            category=str(c.get("category", "")),
        )
        for c in data.get("candidates", [])
    ]


def extract_observations(chunk: Chunk, llm: Any) -> list[RawObservation]:
    """Pass 3: extract qualitative observations. Returns empty list on any error."""
    prompt = build_observations_prompt(chunk.content)
    raw = llm.complete(prompt, temperature=0.0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [
        RawObservation(
            entity_name=str(o.get("entity_name", "")),
            type=str(o.get("type", "")),
            description=str(o.get("description", "")),
        )
        for o in data.get("observations", [])
    ]


def extract_three_pass(chunk: Chunk, llm: Any) -> ExtractionResult:
    """Three-pass extraction: spine -> grounding candidates -> observations.

    Pass 1 (spine) is fatal on parse error; passes 2 and 3 degrade gracefully.
    """
    result = extract_spine(chunk, llm)
    if result.parse_error:
        return result
    result.grounding_candidates = extract_grounding_candidates(chunk, llm, result.entities)
    result.observations = extract_observations(chunk, llm)
    return result


# -- Single-pass (backward compatible) -----------------------------------------


def extract(chunk: Chunk, llm: Any) -> ExtractionResult:
    """Single-pass extraction using the combined prompt. Prefer extract_three_pass()."""
    prompt = build_extraction_prompt(chunk.content)
    raw = llm.complete(prompt, temperature=0.0)
    result = ExtractionResult(chunk_id=chunk.id, source_type=chunk.source_type)
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        result.parse_error = str(exc)
        logger.warning("LLM returned non-JSON for chunk %s: %s", chunk.id, exc)
        return result
    _parse_spine_data(data, result)
    for o in data.get("observations", []):
        result.observations.append(
            RawObservation(
                entity_name=str(o.get("entity_name", "")),
                type=str(o.get("type", "")),
                description=str(o.get("description", "")),
            )
        )
    return result
