"""Phase 4 — InterviewAdapter tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mimir.adapters.interview import InterviewAdapter

_VALID_YAML = textwrap.dedent("""\
    date: "2026-04-15"
    topic: "risk architecture review"
    participants:
      - Interviewer
      - Engineer
    acl:
      - internal
    reference: "interview://2026-04-15/risk-architecture-review"
    transcript:
      - speaker: Interviewer
        text: How does the hedge book feed connect to the clearing house?
      - speaker: Engineer
        text: It goes through the FIX connector to CME clearing.
""")


@pytest.fixture
def interview_file(tmp_path: Path) -> Path:
    f = tmp_path / "interview.yaml"
    f.write_text(_VALID_YAML)
    return f


@pytest.mark.phase4
def test_load_returns_one_chunk(interview_file: Path) -> None:
    chunks = InterviewAdapter().load(interview_file)
    assert len(chunks) == 1
    assert chunks[0].source_type == "interview"


@pytest.mark.phase4
def test_load_content_includes_transcript(interview_file: Path) -> None:
    chunks = InterviewAdapter().load(interview_file)
    assert "FIX connector" in chunks[0].content
    assert "Interviewer:" in chunks[0].content


@pytest.mark.phase4
def test_load_acl_from_yaml(interview_file: Path) -> None:
    chunks = InterviewAdapter().load(interview_file)
    assert "internal" in chunks[0].acl


@pytest.mark.phase4
def test_load_reference_from_yaml(interview_file: Path) -> None:
    chunks = InterviewAdapter().load(interview_file)
    assert chunks[0].reference == "interview://2026-04-15/risk-architecture-review"


@pytest.mark.phase4
def test_load_metadata(interview_file: Path) -> None:
    chunks = InterviewAdapter().load(interview_file)
    meta = chunks[0].metadata
    assert meta["topic"] == "risk architecture review"
    assert "Interviewer" in meta["participants"]


@pytest.mark.phase4
def test_load_missing_transcript_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("date: 2026-04-15\ntopic: test\n")
    with pytest.raises(ValueError, match="transcript"):
        InterviewAdapter().load(bad)


@pytest.mark.phase4
def test_load_missing_speaker_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("transcript:\n  - text: hello\n")
    with pytest.raises(ValueError, match="speaker"):
        InterviewAdapter().load(bad)


@pytest.mark.phase4
def test_load_file_not_found_raises() -> None:
    with pytest.raises(FileNotFoundError):
        InterviewAdapter().load("/nonexistent/path/interview.yaml")
