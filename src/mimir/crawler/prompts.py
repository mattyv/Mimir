"""Extraction prompt templates for the crawler.

The prompt is built dynamically from the current vocabulary so it never
drifts out of sync with vocabulary.yaml.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_VOCAB_PATH = Path(__file__).parent.parent / "vocabulary" / "vocabulary.yaml"

_HEADER = """\
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


@lru_cache(maxsize=1)
def _vocabulary_strings() -> tuple[str, str]:
    """Return (entity_type_list, predicate_list) strings from vocabulary.yaml."""
    from mimir.vocabulary.loader import load_vocabulary

    vocab = load_vocabulary(_VOCAB_PATH)
    entity_types = ", ".join(e.iri for e in vocab.entity_types)
    predicates = ", ".join(p.iri for p in vocab.predicates)
    return entity_types, predicates


def build_extraction_prompt(chunk_content: str) -> str:
    """Return the full prompt string for entity/relationship extraction."""
    entity_types, predicates = _vocabulary_strings()
    system = _HEADER.format(entity_types=entity_types, predicates=predicates)
    return f"{system}\nText:\n{chunk_content}\n\nJSON:"
