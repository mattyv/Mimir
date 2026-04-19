"""Phase 0 — Scaffolding & Contracts tests.

Verifies that the test harness, fixtures, and eval file integrity are all
in place before any business logic is implemented.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.conftest import FakeEmbedder, FakeLLM, FakeSPARQL

pytestmark = pytest.mark.phase0

_REPO_ROOT = Path(__file__).parent.parent.parent
_EVAL_FILE = _REPO_ROOT / "eval" / "frozen_questions.yaml"
_CHECKSUM_FILE = _REPO_ROOT / ".eval_checksum"


# ── Fixture smoke tests ────────────────────────────────────────────────────────


def test_fixtures_load(
    fake_llm: FakeLLM,
    fake_embedder: FakeEmbedder,
    fake_sparql: FakeSPARQL,
    sample_chunks: list,
    core_vocabulary: object,
) -> None:
    """Every non-infrastructure fixture instantiates without error."""
    assert fake_llm is not None
    assert fake_embedder is not None
    assert fake_sparql is not None
    assert len(sample_chunks) > 0
    assert core_vocabulary is not None


def test_fake_llm_determinism(fake_llm: FakeLLM) -> None:
    """Same prompt hash → same response, across 10 calls."""
    prompt = "What is the SLO for order submission?"
    responses = [fake_llm.complete(prompt) for _ in range(10)]
    assert len(set(responses)) == 1, "FakeLLM must return the same response for the same prompt"


def test_fake_llm_scripted_response(fake_llm: FakeLLM) -> None:
    """Scripted response is returned when prompt matches."""
    prompt = "Who owns OMMS?"
    fake_llm.set_response(prompt, "The APAC team owns OMMS.")
    assert fake_llm.complete(prompt) == "The APAC team owns OMMS."


def test_fake_embedder_determinism(fake_embedder: FakeEmbedder) -> None:
    """Same text → same embedding vector, across 10 calls."""
    text = "options market making service"
    vectors = [fake_embedder.embed(text) for _ in range(10)]
    assert all(v == vectors[0] for v in vectors), "FakeEmbedder must be deterministic"


def test_fake_embedder_unit_vector(fake_embedder: FakeEmbedder) -> None:
    """Returned vector has unit magnitude (within floating-point tolerance)."""
    import math

    vec = fake_embedder.embed("risk engine")
    magnitude = math.sqrt(sum(v * v for v in vec))
    assert abs(magnitude - 1.0) < 1e-6, f"Expected unit vector, got magnitude {magnitude}"


def test_fake_embedder_different_texts_differ(fake_embedder: FakeEmbedder) -> None:
    """Different inputs produce different embedding vectors."""
    v1 = fake_embedder.embed("risk engine")
    v2 = fake_embedder.embed("order book")
    assert v1 != v2


def test_fake_sparql_deterministic(fake_sparql: FakeSPARQL) -> None:
    """Same SPARQL query → same response."""
    query = "SELECT ?x WHERE { ?x a <https://schema.org/Organization> }"
    r1 = fake_sparql.query(query)
    r2 = fake_sparql.query(query)
    assert r1 == r2


def test_fake_sparql_scripted_response(fake_sparql: FakeSPARQL) -> None:
    """Scripted response is returned for a known query."""
    query = "SELECT ?label WHERE { wd:Q5292 rdfs:label ?label }"
    fake_sparql.set_response(query, {"results": {"bindings": [{"label": {"value": "Factory"}}]}})
    result = fake_sparql.query(query)
    assert result["results"]["bindings"][0]["label"]["value"] == "Factory"


def test_sample_chunks_cover_all_source_types(sample_chunks: list) -> None:
    """Fixture includes at least one chunk per source type."""
    from tests.conftest import Chunk

    assert all(isinstance(c, Chunk) for c in sample_chunks)
    source_types = {c.source_type for c in sample_chunks}
    expected = {"confluence", "github", "slack", "interview", "code_analysis"}
    assert source_types == expected


def test_sample_chunks_have_acl(sample_chunks: list) -> None:
    """Every chunk carries at least one ACL entry."""
    for chunk in sample_chunks:
        assert chunk.acl, f"Chunk {chunk.id!r} has no ACL"


# ── Eval file integrity ────────────────────────────────────────────────────────


def test_frozen_questions_file_exists() -> None:
    """eval/frozen_questions.yaml must exist in the repository root."""
    assert _EVAL_FILE.exists(), (
        f"eval/frozen_questions.yaml not found at {_EVAL_FILE}. "
        "Create it before closing Phase 0."
    )


def test_frozen_questions_checksum_file_exists() -> None:
    """`.eval_checksum` must exist alongside the questions file."""
    assert _CHECKSUM_FILE.exists(), (
        f".eval_checksum not found at {_CHECKSUM_FILE}. "
        "Run `sha256sum eval/frozen_questions.yaml` and store the hash."
    )


def test_frozen_questions_checksum_matches() -> None:
    """The stored SHA-256 checksum must match the current file contents."""
    content = _EVAL_FILE.read_bytes()
    computed = hashlib.sha256(content).hexdigest()
    stored = _CHECKSUM_FILE.read_text().strip()
    assert computed == stored, (
        f"eval/frozen_questions.yaml has been modified!\n"
        f"  Stored  : {stored}\n"
        f"  Computed: {computed}\n"
        "Do not edit the frozen questions after Phase 0."
    )


def test_frozen_questions_has_twenty_questions() -> None:
    """The eval file must contain exactly 20 questions."""
    import yaml

    data = yaml.safe_load(_EVAL_FILE.read_text())
    questions = data.get("questions", [])
    assert len(questions) == 20, (
        f"Expected 20 frozen eval questions, found {len(questions)}."
    )
