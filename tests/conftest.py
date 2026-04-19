"""Shared fixtures for all Mimir test phases.

Fixtures defined here are available to every test module without import.
Heavy infrastructure fixtures (pg, fake_llm, etc.) are defined here but
only exercised by the phases that need them.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

# ── Vocabulary path ────────────────────────────────────────────────────────────

_VOCAB_PATH = Path(__file__).parent.parent / "src" / "mimir" / "vocabulary" / "vocabulary.yaml"


# ── Stub classes ───────────────────────────────────────────────────────────────


class FakeLLM:
    """Deterministic LLM stub.

    Responses are keyed by SHA-256 of the prompt string.  Scripts responses
    via set_response(); unscripted prompts return a stable placeholder.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses: dict[str, str] = responses or {}

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def complete(self, prompt: str) -> str:
        key = self._key(prompt)
        return self._responses.get(key, f"FAKE_RESPONSE_{key[:8]}")

    def set_response(self, prompt: str, response: str) -> None:
        self._responses[self._key(prompt)] = response


class FakeEmbedder:
    """Deterministic sentence-embedding stub.

    Returns a normalised vector whose components are derived from the SHA-256
    of the input text.  Same text always produces the same vector.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        # Build dim floats by cycling through digest bytes
        vals = [float(digest[i % len(digest)]) - 127.5 for i in range(self.dim)]
        magnitude = math.sqrt(sum(v * v for v in vals))
        if magnitude > 0:
            vals = [v / magnitude for v in vals]
        return vals


class FakeSPARQL:
    """Wikidata SPARQL stub with recorded fixture responses."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None) -> None:
        self._responses: dict[str, dict[str, Any]] = responses or {}

    def query(self, sparql_query: str) -> dict[str, Any]:
        return self._responses.get(sparql_query, {"results": {"bindings": []}})

    def set_response(self, sparql_query: str, response: dict[str, Any]) -> None:
        self._responses[sparql_query] = response


@dataclass
class Chunk:
    """A raw content chunk from a source system, pre-extraction."""

    id: str
    source_type: str  # confluence | github | slack | interview | code_analysis
    content: str
    acl: list[str] = field(default_factory=list)
    retrieved_at: datetime = field(
        default_factory=lambda: datetime(2026, 4, 19, tzinfo=UTC)
    )
    reference: str = ""


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_sparql() -> FakeSPARQL:
    return FakeSPARQL()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Corpus of test chunks covering all source types with ACL metadata."""
    return [
        Chunk(
            id="confluence_001",
            source_type="confluence",
            content=(
                "The options market making service (OMMS) is owned by the APAC team. "
                "It connects to venues CME and ICE via the FIX connector. "
                "The SLO for order submission is <5ms p99."
            ),
            acl=["space:trading-eng"],
            reference="https://wiki.example.com/spaces/trading-eng/OMMS-overview",
        ),
        Chunk(
            id="github_001",
            source_type="github",
            content=(
                "# panic_server\n\n"
                "Safety circuit breaker for the trading system. "
                "Implemented in Rust. Depends on the risk engine via gRPC.\n"
                "Maintained by the risk-infra team."
            ),
            acl=["repo:risk-infra/panic_server"],
            reference="https://github.com/example/panic_server/README.md",
        ),
        Chunk(
            id="slack_001",
            source_type="slack",
            content=(
                "dmitri_v: PAN-12445 is blocked on the sub-account consolidation. "
                "We need hawkeye to expose the new API first."
            ),
            acl=["channel:trading-risk"],
            reference="https://slack.example.com/archives/trading-risk/p1713523200",
        ),
        Chunk(
            id="interview_001",
            source_type="interview",
            content=(
                "Interviewer: How does the hedge book feed connect to the clearing house?\n"
                "Engineer: It goes through the FIX connector to CME clearing. "
                "The dependency is hard — if the feed goes down, we stop quoting."
            ),
            acl=["internal"],
            reference="interview://2026-04-15/risk-architecture-review",
        ),
        Chunk(
            id="code_001",
            source_type="code_analysis",
            content=(
                "Module: risk_engine.py\n"
                "Language: Python 3.12\n"
                "Dependencies: numpy, pandas, asyncio\n"
                "Exports: RiskCalculator, PositionLimiter\n"
                "Cyclomatic complexity: 14 (high)"
            ),
            acl=["repo:risk-infra/risk-engine"],
            reference="code://risk-infra/risk-engine/risk_engine.py",
        ),
    ]


@pytest.fixture
def core_vocabulary() -> Any:
    """Frozen snapshot of the core IRI set loaded from vocabulary.yaml."""
    from mimir.vocabulary.loader import load_vocabulary

    return load_vocabulary(_VOCAB_PATH)


@pytest.fixture
def pg() -> Any:
    """Per-test PostgreSQL instance (Phase 2+).

    Requires pytest-postgresql and a running PostgreSQL server.
    Skipped in Phase 0/1 since no persistence is needed yet.
    """
    pytest.skip("PostgreSQL fixture not configured until Phase 2")
