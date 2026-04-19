"""Phase 4 — PII / secret scanner tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mimir.adapters.base import Chunk
from mimir.adapters.pii import scan_chunk

_NOW = datetime(2026, 4, 19, tzinfo=UTC)


def _chunk(content: str, chunk_id: str = "test_chunk") -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type="confluence",
        content=content,
        retrieved_at=_NOW,
    )


@pytest.mark.phase4
def test_scan_clean_content_has_no_secrets() -> None:
    chunk = _chunk("The OMMS service is owned by the APAC team.")
    result = scan_chunk(chunk)
    assert result.has_secrets is False
    assert result.findings == []


@pytest.mark.phase4
def test_scan_aws_key_detected() -> None:
    chunk = _chunk("config: AKIAIOSFODNN7EXAMPLE")
    result = scan_chunk(chunk)
    assert result.has_secrets is True
    assert any("AWS" in f.secret_type for f in result.findings)


@pytest.mark.phase4
def test_scan_result_chunk_id_matches() -> None:
    chunk = _chunk("normal content", chunk_id="my_chunk_123")
    result = scan_chunk(chunk)
    assert result.chunk_id == "my_chunk_123"


@pytest.mark.phase4
def test_scan_findings_include_line_number() -> None:
    chunk = _chunk("line one\nAKIAIOSFODNN7EXAMPLE\nline three")
    result = scan_chunk(chunk)
    assert result.has_secrets is True
    assert result.findings[0].line_number == 2


@pytest.mark.phase4
def test_scan_multiline_clean_content() -> None:
    content = (
        "# Architecture Overview\n"
        "The trading system consists of three core services:\n"
        "1. Options Market Maker\n"
        "2. Risk Engine\n"
        "3. Panic Server\n"
    )
    result = scan_chunk(_chunk(content))
    assert result.has_secrets is False
