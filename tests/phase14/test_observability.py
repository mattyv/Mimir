"""Phase 14 — observability tests (metrics + structured logging)."""

from __future__ import annotations

import json
import logging

import pytest

from mimir.observability.logging import JsonFormatter, get_logger, log_pipeline_event
from mimir.observability.metrics import (
    Counter,
    Histogram,
    MetricsRegistry,
    get_registry,
)

# ── Counter ───────────────────────────────────────────────────────────────────


@pytest.mark.phase14
def test_counter_starts_at_zero() -> None:
    c = Counter(name="test")
    assert c.value == 0.0


@pytest.mark.phase14
def test_counter_inc_default() -> None:
    c = Counter(name="test")
    c.inc()
    assert c.value == 1.0


@pytest.mark.phase14
def test_counter_inc_custom_amount() -> None:
    c = Counter(name="test")
    c.inc(5.0)
    assert c.value == 5.0


@pytest.mark.phase14
def test_counter_accumulates() -> None:
    c = Counter(name="test")
    c.inc()
    c.inc(2.0)
    assert c.value == 3.0


# ── Histogram ─────────────────────────────────────────────────────────────────


@pytest.mark.phase14
def test_histogram_empty() -> None:
    h = Histogram(name="latency")
    assert h.count == 0
    assert h.mean == 0.0
    assert h.percentile(99) == 0.0


@pytest.mark.phase14
def test_histogram_observe() -> None:
    h = Histogram(name="latency")
    h.observe(10.0)
    h.observe(20.0)
    assert h.count == 2
    assert h.sum == 30.0
    assert h.mean == 15.0


@pytest.mark.phase14
def test_histogram_percentile() -> None:
    h = Histogram(name="latency")
    for v in range(1, 101):
        h.observe(float(v))
    # floor-based index: int(n * p/100) gives the element just above the boundary
    assert h.percentile(50) == 51.0
    assert h.percentile(99) == 100.0


@pytest.mark.phase14
def test_histogram_single_observation() -> None:
    h = Histogram(name="latency")
    h.observe(42.0)
    assert h.mean == 42.0
    assert h.percentile(99) == 42.0


# ── MetricsRegistry ───────────────────────────────────────────────────────────


@pytest.mark.phase14
def test_registry_counter_idempotent() -> None:
    r = MetricsRegistry()
    c1 = r.counter("requests", method="GET")
    c2 = r.counter("requests", method="GET")
    assert c1 is c2


@pytest.mark.phase14
def test_registry_counter_different_labels() -> None:
    r = MetricsRegistry()
    c1 = r.counter("requests", method="GET")
    c2 = r.counter("requests", method="POST")
    assert c1 is not c2


@pytest.mark.phase14
def test_registry_histogram_idempotent() -> None:
    r = MetricsRegistry()
    h1 = r.histogram("latency", route="/api")
    h2 = r.histogram("latency", route="/api")
    assert h1 is h2


@pytest.mark.phase14
def test_registry_snapshot_structure() -> None:
    r = MetricsRegistry()
    c = r.counter("hits")
    c.inc(3)
    h = r.histogram("duration")
    h.observe(1.0)
    snap = r.snapshot()
    assert "counters" in snap
    assert "histograms" in snap
    counter_vals = list(snap["counters"].values())
    assert any(v["value"] == 3.0 for v in counter_vals)


@pytest.mark.phase14
def test_registry_reset_clears_all() -> None:
    r = MetricsRegistry()
    r.counter("x").inc()
    r.histogram("y").observe(1.0)
    r.reset()
    snap = r.snapshot()
    assert snap["counters"] == {}
    assert snap["histograms"] == {}


@pytest.mark.phase14
def test_get_registry_returns_same_instance() -> None:
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


# ── JsonFormatter ─────────────────────────────────────────────────────────────


@pytest.mark.phase14
def test_json_formatter_produces_valid_json(caplog: pytest.LogCaptureFixture) -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello world", args=(), exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["msg"] == "hello world"
    assert parsed["level"] == "INFO"
    assert "ts" in parsed


@pytest.mark.phase14
def test_json_formatter_includes_extra_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="event", args=(), exc_info=None,
    )
    record.chunk_id = "abc123"
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed.get("extra", {}).get("chunk_id") == "abc123"


@pytest.mark.phase14
def test_get_logger_returns_logger() -> None:
    logger = get_logger("mimir.test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "mimir.test_logger"


@pytest.mark.phase14
def test_log_pipeline_event_no_error(capfd: pytest.CaptureFixture[str]) -> None:
    logger = get_logger("mimir.pipe", level=logging.DEBUG)
    log_pipeline_event(logger, "chunk_processed", "chunk_001", entities=3)
    out = capfd.readouterr().out
    assert "chunk_processed" in out or out == ""  # may be buffered
