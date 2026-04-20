"""Phase 9 — grounder cache tests."""
from __future__ import annotations

import pytest

from mimir.grounder.cache import clear_cache, get_cached, put_cached

pytestmark = pytest.mark.phase9


@pytest.fixture(autouse=True)
def _clear() -> None:
    """Ensure a clean cache before each test."""
    clear_cache()


def test_cache_miss_initially() -> None:
    hit, qid, label = get_cached("Douglas Adams")
    assert hit is False
    assert qid is None
    assert label == ""


def test_cache_hit_after_put() -> None:
    put_cached("Douglas Adams", qid="Q42", label="Douglas Adams")
    hit, qid, label = get_cached("Douglas Adams")
    assert hit is True
    assert qid == "Q42"
    assert label == "Douglas Adams"


def test_cache_stores_none_for_no_match() -> None:
    put_cached("UnknownEntity", qid=None, label="")
    hit, qid, label = get_cached("UnknownEntity")
    assert hit is True
    assert qid is None


def test_clear_cache_empties() -> None:
    put_cached("SomeName", qid="Q1", label="SomeName")
    clear_cache()
    hit, qid, label = get_cached("SomeName")
    assert hit is False
