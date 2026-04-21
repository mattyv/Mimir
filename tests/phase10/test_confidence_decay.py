"""Phase 10 — confidence decay tests (§7.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mimir.persistence.repository import _decayed_confidence, apply_confidence_decay

pytestmark = pytest.mark.phase10


def test_decay_freshly_written_unchanged() -> None:
    now = datetime.now(UTC)
    # 0 days old → 2^0 = 1.0 multiplier
    result = _decayed_confidence(0.9, now, "confluence")
    assert result == pytest.approx(0.9, abs=0.01)


def test_decay_one_half_life_halves_confidence() -> None:
    half_life = 180  # confluence half-life in days
    past = datetime.now(UTC) - timedelta(days=half_life)
    result = _decayed_confidence(0.8, past, "confluence")
    assert result == pytest.approx(0.4, abs=0.01)  # exactly at floor


def test_decay_slack_faster_than_confluence() -> None:
    one_month_ago = datetime.now(UTC) - timedelta(days=30)
    slack_decay = _decayed_confidence(0.9, one_month_ago, "slack")
    conf_decay = _decayed_confidence(0.9, one_month_ago, "confluence")
    assert slack_decay < conf_decay


def test_decay_floor_enforced() -> None:
    very_old = datetime.now(UTC) - timedelta(days=3650)
    result = _decayed_confidence(0.9, very_old, "slack")
    assert result >= 0.4


def test_decay_no_valid_from_returns_base() -> None:
    result = _decayed_confidence(0.7, None, "github")
    assert result == pytest.approx(0.7, abs=0.001)


def test_apply_confidence_decay_disabled() -> None:
    row = {"confidence": 0.9, "valid_from": datetime.now(UTC), "payload": {"source": {"type": "slack"}}}
    result = apply_confidence_decay(row)
    assert "decayed_confidence" not in result


def test_apply_confidence_decay_enabled() -> None:
    row = {
        "confidence": 0.9,
        "valid_from": datetime.now(UTC),
        "payload": {"source": {"type": "confluence"}},
    }
    result = apply_confidence_decay(row, apply_decay=True)
    assert "decayed_confidence" in result
    assert 0.4 <= result["decayed_confidence"] <= 0.9


def test_decay_unknown_source_uses_default() -> None:
    one_month = datetime.now(UTC) - timedelta(days=30)
    result = _decayed_confidence(0.9, one_month, "unknown_source_type")
    assert 0.4 <= result <= 0.9
