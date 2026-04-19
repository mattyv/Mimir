"""Crawler pipeline: chunks → extraction → persistence.

Orchestrates: LLM extraction → node construction → upsert.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import psycopg

from mimir.adapters.base import Chunk
from mimir.adapters.pii import scan_chunk
from mimir.crawler.extractor import ExtractionResult, extract
from mimir.grounder.wikidata import SPARQLClient, ground_entity
from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Observation, Property, Relationship
from mimir.observability.logging import get_logger, log_pipeline_event
from mimir.observability.metrics import get_registry
from mimir.persistence.repository import (
    EntityRepository,
    ObservationRepository,
    PropertyRepository,
    RelationshipRepository,
)

_logger = get_logger("mimir.pipeline")
_reg = get_registry()

_VOCAB_VERSION = "0.1.0"
_DEFAULT_VISIBILITY = Visibility(acl=["internal"], sensitivity="internal")


def _grounding(tier: GroundingTier = GroundingTier.source_cited) -> Grounding:
    return Grounding(tier=tier, depth=1, stop_reason="llm_extraction")


def _temporal(chunk: Chunk) -> Temporal:
    return Temporal(valid_from=chunk.retrieved_at)


def _source(chunk: Chunk) -> Source:
    return Source(
        type=chunk.source_type,
        reference=chunk.reference,
        retrieved_at=chunk.retrieved_at,
    )


def _sensitivity_from_acl(acl: list[str]) -> str:
    if any(a.startswith(("restricted:", "private:", "confidential:")) for a in acl):
        return "restricted"
    if not acl or "public" in acl:
        return "public"
    return "internal"


def _visibility(chunk: Chunk) -> Visibility:
    if not chunk.acl:
        return _DEFAULT_VISIBILITY
    sensitivity = _sensitivity_from_acl(chunk.acl)
    return Visibility(acl=chunk.acl, sensitivity=sensitivity)  # type: ignore[arg-type,unused-ignore]


@dataclass
class PipelineResult:
    chunk_id: str
    parse_error: str | None = None
    entities_upserted: int = 0
    entities_grounded: int = 0
    properties_inserted: int = 0
    relationships_inserted: int = 0
    observations_inserted: int = 0
    unknown_entity_refs: list[str] = field(default_factory=list)
    pii_skipped: bool = False
    pii_findings: list[str] = field(default_factory=list)


def process_chunk(
    chunk: Chunk,
    llm: Any,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    sparql_client: SPARQLClient | None = None,
) -> PipelineResult:
    """Run one chunk through the full pipeline inside the caller's transaction.

    Steps:
    1. LLM extraction — call the model, parse JSON.
    2. Entity upsert — create/update Entity rows.
    3. Relationship insert — link entities (skip if either endpoint unknown).
    4. Observation insert — attach observations to known entities.
    """
    result = PipelineResult(chunk_id=chunk.id)

    _reg.counter("chunks_processed", source_type=chunk.source_type).inc()
    _reg.histogram("chunk_content_length").observe(float(len(chunk.content)))

    # PII pre-scan — skip chunks that contain secrets
    pii = scan_chunk(chunk)
    if pii.has_secrets:
        result.pii_skipped = True
        result.pii_findings = [f.secret_type for f in pii.findings]
        log_pipeline_event(_logger, "chunk_skipped_pii", chunk.id, findings=result.pii_findings)
        _reg.counter("chunks_pii_skipped", source_type=chunk.source_type).inc()
        return result

    extraction: ExtractionResult = extract(chunk, llm)
    if extraction.parse_error:
        result.parse_error = extraction.parse_error
        _reg.counter("extraction_errors", source_type=chunk.source_type).inc()
        log_pipeline_event(_logger, "extraction_error", chunk.id, error=extraction.parse_error)
        return result

    entity_repo = EntityRepository(conn)
    prop_repo = PropertyRepository(conn)
    rel_repo = RelationshipRepository(conn)
    obs_repo = ObservationRepository(conn)

    name_to_id: dict[str, str] = {}

    for e_raw in extraction.entities:
        entity_type = e_raw.type or "schema:SoftwareApplication"
        e_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{e_raw.name}:{entity_type}"))
        entity = Entity(
            id=e_id,
            type=entity_type,
            name=e_raw.name,
            description=e_raw.description,
            created_at=datetime.now(UTC),
            confidence=0.8,
            grounding=_grounding(),
            temporal=_temporal(chunk),
            visibility=_visibility(chunk),
            vocabulary_version=_VOCAB_VERSION,
        )
        entity_repo.upsert(entity)
        name_to_id[e_raw.name] = e_id
        result.entities_upserted += 1

    for p_raw in extraction.properties:
        entity_id = name_to_id.get(p_raw.entity_name)
        if entity_id is None:
            result.unknown_entity_refs.append(p_raw.entity_name)
            continue
        prop = Property(
            entity_id=entity_id,
            key=p_raw.key,
            value=p_raw.value,
            value_type=type(p_raw.value).__name__,
            confidence=0.8,
            source=_source(chunk),
            grounding=_grounding(),
            temporal=_temporal(chunk),
            visibility=_visibility(chunk),
            vocabulary_version=_VOCAB_VERSION,
        )
        prop_repo.insert(prop)
        result.properties_inserted += 1

    for r_raw in extraction.relationships:
        subj_id = name_to_id.get(r_raw.subject)
        obj_id = name_to_id.get(r_raw.object)
        if subj_id is None:
            result.unknown_entity_refs.append(r_raw.subject)
            continue
        if obj_id is None:
            result.unknown_entity_refs.append(r_raw.object)
            continue
        rel = Relationship(
            subject_id=subj_id,
            predicate=r_raw.predicate,
            object_id=obj_id,
            confidence=0.8,
            source=_source(chunk),
            grounding=_grounding(),
            temporal=_temporal(chunk),
            visibility=_visibility(chunk),
            vocabulary_version=_VOCAB_VERSION,
        )
        rel_repo.insert(rel)
        result.relationships_inserted += 1

    for o_raw in extraction.observations:
        o_eid = name_to_id.get(o_raw.entity_name)
        if o_eid is None:
            result.unknown_entity_refs.append(o_raw.entity_name)
            continue
        obs = Observation(
            entity_id=o_eid,
            type=o_raw.type,  # type: ignore[arg-type,unused-ignore]
            description=o_raw.description,
            confidence=0.8,
            source=_source(chunk),
            grounding=_grounding(),
            temporal=_temporal(chunk),
            visibility=_visibility(chunk),
            vocabulary_version=_VOCAB_VERSION,
        )
        obs_repo.insert(obs)
        result.observations_inserted += 1

    # Optional Wikidata grounding pass
    if sparql_client is not None:
        for entity_name, entity_id in name_to_id.items():
            match = ground_entity(entity_id, entity_name, sparql_client, conn)
            if match:
                result.entities_grounded += 1

    _reg.counter("entities_upserted", source_type=chunk.source_type).inc(float(result.entities_upserted))
    _reg.counter("properties_inserted", source_type=chunk.source_type).inc(float(result.properties_inserted))
    _reg.counter("relationships_inserted").inc(float(result.relationships_inserted))
    _reg.counter("observations_inserted").inc(float(result.observations_inserted))
    log_pipeline_event(_logger, "chunk_processed", chunk.id,
                       entities=result.entities_upserted,
                       properties=result.properties_inserted,
                       relationships=result.relationships_inserted,
                       observations=result.observations_inserted)

    return result
