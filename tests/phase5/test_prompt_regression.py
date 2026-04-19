"""Phase 5 — prompt regression: prompt must include all core vocabulary IRIs."""

from __future__ import annotations

import pytest

from mimir.crawler.prompts import build_extraction_prompt


@pytest.mark.phase5
def test_extraction_prompt_includes_entity_types(core_vocabulary: object) -> None:
    """Every core entity type IRI must appear in the extraction prompt."""
    from mimir.vocabulary.loader import Vocabulary
    vocab: Vocabulary = core_vocabulary  # type: ignore[assignment]
    prompt = build_extraction_prompt("test content")
    missing = [e.iri for e in vocab.entity_types if e.iri not in prompt]
    assert not missing, f"Entity type IRIs missing from prompt: {missing}"


@pytest.mark.phase5
def test_extraction_prompt_includes_predicates(core_vocabulary: object) -> None:
    """Every core predicate IRI must appear in the extraction prompt."""
    from mimir.vocabulary.loader import Vocabulary
    vocab: Vocabulary = core_vocabulary  # type: ignore[assignment]
    prompt = build_extraction_prompt("test content")
    missing = [p.iri for p in vocab.predicates if p.iri not in prompt]
    assert not missing, f"Predicate IRIs missing from prompt: {missing}"


@pytest.mark.phase5
def test_extraction_prompt_includes_observation_types(core_vocabulary: object) -> None:
    """All 8 closed observation types must appear in the prompt."""
    obs_types = [
        "strength", "risk", "anti_pattern", "maturity",
        "smell", "opportunity", "inconsistency", "functional_state",
    ]
    prompt = build_extraction_prompt("test content")
    missing = [t for t in obs_types if t not in prompt]
    assert not missing, f"Observation types missing from prompt: {missing}"


@pytest.mark.phase5
def test_extraction_prompt_has_json_keys(core_vocabulary: object) -> None:
    prompt = build_extraction_prompt("some content")
    for key in ("entities", "properties", "relationships", "observations"):
        assert key in prompt
