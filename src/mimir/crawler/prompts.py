"""Extraction prompt templates for the crawler.

Three focused prompts for the three-pass extraction pipeline:
  build_spine_prompt                — entities, properties, relationships (Pass 1)
  build_grounding_candidates_prompt — Wikidata concept matching (Pass 2)
  build_observations_prompt         — qualitative observations (Pass 3)

build_extraction_prompt is the original combined prompt kept for backward
compatibility (single-pass extract() and prompt regression tests).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_VOCAB_PATH = Path(__file__).parent.parent / "vocabulary" / "vocabulary.yaml"


@lru_cache(maxsize=1)
def _vocabulary_strings() -> tuple[str, str]:
    """Return (entity_type_list, predicate_list) strings from vocabulary.yaml."""
    from mimir.vocabulary.loader import load_vocabulary

    vocab = load_vocabulary(_VOCAB_PATH)
    entity_types = ", ".join(e.iri for e in vocab.entity_types)
    predicates = ", ".join(p.iri for p in vocab.predicates)
    return entity_types, predicates


# ── Pass 1: spine (entities, properties, relationships) ───────────────────────

_SPINE_HEADER = """\
You are a knowledge-graph extraction engine.  Given a text chunk from a \
technical document, extract structural facts as JSON.

Return a JSON object with these keys (omit any that have no findings):
  entities     : list of {{name, type, description}}
  properties   : list of {{entity_name, key, value}}
  relationships: list of {{subject, predicate, object}}

Rules:
- entity type must be one of: {entity_types}
- predicate must be one of: {predicates}
- Be conservative: only extract facts explicitly stated in the text.
- If no core IRI fits, use auros:provisional:<snake_name>.
- Return ONLY the JSON object, no markdown fences.
"""

# ── Pass 2: grounding candidates (Wikidata concept matching) ──────────────────

_GROUNDING_HEADER = """\
For each entity listed below, determine whether it clearly matches a \
well-known external concept from one of these categories:
  - design_pattern (Factory Method, Observer, Circuit Breaker, Saga, ...)
  - methodology (TDD, Agile, Scrum, Kanban, Waterfall, ...)
  - mathematical_construct (Bayesian inference, FFT, Kalman filter, ...)
  - technical_concept (OAuth 2, REST, gRPC, async I/O, ...)
  - domain_concept (market making, delta hedging, clearing house, ...)

For each clear match include: entity_name, wikidata_qid (e.g. "Q5292"), \
label (English Wikidata label), and category (one of the snake_case values above).
Only include entities with a confident match — omit uncertain ones.

Entities to evaluate:
{entity_list}

Chunk context:
{chunk_content}

Return JSON: {{"candidates": \
[{{"entity_name": "...", "wikidata_qid": "Q...", "label": "...", "category": "..."}}]}}
Return ONLY the JSON object, no markdown fences.
"""

# ── Pass 3: observations ──────────────────────────────────────────────────────

_OBSERVATIONS_HEADER = """\
You are a knowledge-graph observation extractor.  Given a text chunk from a \
technical document, extract qualitative observations about named entities.

Return a JSON object with this key (omit if no observations):
  observations : list of {{entity_name, type, description}}

Rules:
- observation type must be one of: \
strength, risk, anti_pattern, maturity, smell, opportunity, inconsistency, functional_state
- Be conservative: only extract observations explicitly supported by the text.
- entity_name must match a name that appears in the chunk.
- description is a concise natural-language sentence.
- Return ONLY the JSON object, no markdown fences.
"""

# ── Original combined prompt (backward compat) ────────────────────────────────

_COMBINED_HEADER = """\
You are a knowledge-graph extraction engine.  Given a text chunk from a \
technical document, extract structured facts as JSON.

Return a JSON object with these keys (omit any that have no findings):
  entities   : list of {{name, type, description}}
  properties : list of {{entity_name, key, value}}
  relationships: list of {{subject, predicate, object}}
  observations : list of {{entity_name, type, description}}

Rules:
- entity type must be one of: {entity_types}
- predicate must be one of: {predicates}
- observation type must be one of: strength, risk, anti_pattern, \
maturity, smell, opportunity, inconsistency, functional_state
- Be conservative: only extract facts explicitly stated in the text.
- If no core IRI fits, use auros:provisional:<snake_name>.
- Return ONLY the JSON object, no markdown fences.
"""


def build_spine_prompt(chunk_content: str) -> str:
    """Pass 1 prompt: extract entities, properties, and relationships."""
    entity_types, predicates = _vocabulary_strings()
    header = _SPINE_HEADER.format(entity_types=entity_types, predicates=predicates)
    return f"{header}\nText:\n{chunk_content}\n\nJSON:"


def build_grounding_candidates_prompt(chunk_content: str, entity_list: str) -> str:
    """Pass 2 prompt: match entities to known external concepts (Wikidata hints)."""
    return _GROUNDING_HEADER.format(
        entity_list=entity_list or "(none)",
        chunk_content=chunk_content,
    )


def build_observations_prompt(chunk_content: str) -> str:
    """Pass 3 prompt: extract qualitative observations."""
    return f"{_OBSERVATIONS_HEADER}\nText:\n{chunk_content}\n\nJSON:"


def build_extraction_prompt(chunk_content: str) -> str:
    """Combined single-pass prompt (original; kept for backward compatibility).

    Used by the single-pass extract() function and prompt regression tests.
    Prefer the three focused prompts for new callers.
    """
    entity_types, predicates = _vocabulary_strings()
    header = _COMBINED_HEADER.format(entity_types=entity_types, predicates=predicates)
    return f"{header}\nText:\n{chunk_content}\n\nJSON:"
