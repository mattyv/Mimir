"""In-process metrics registry — counters and histograms.

This is a deliberately lightweight implementation that avoids external
dependencies (no prometheus_client, no statsd).  It is suitable for
embedding in tests and for basic operational visibility.

For production use, replace or wrap with your preferred metrics library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class Counter:
    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


@dataclass
class Histogram:
    name: str
    observations: list[float] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    def observe(self, value: float) -> None:
        self.observations.append(value)

    @property
    def count(self) -> int:
        return len(self.observations)

    @property
    def sum(self) -> float:
        return sum(self.observations)

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0–100) of recorded observations."""
        if not self.observations:
            return 0.0
        sorted_obs = sorted(self.observations)
        idx = int(len(sorted_obs) * p / 100)
        return sorted_obs[min(idx, len(sorted_obs) - 1)]


class MetricsRegistry:
    """Thread-safe in-process metrics registry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str, **labels: str) -> Counter:
        key = f"{name}|{sorted(labels.items())}"
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(name=name, labels=dict(labels))
            return self._counters[key]

    def histogram(self, name: str, **labels: str) -> Histogram:
        key = f"{name}|{sorted(labels.items())}"
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(name=name, labels=dict(labels))
            return self._histograms[key]

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all metrics."""
        with self._lock:
            return {
                "counters": {
                    k: {"name": c.name, "value": c.value, "labels": c.labels}
                    for k, c in self._counters.items()
                },
                "histograms": {
                    k: {
                        "name": h.name,
                        "count": h.count,
                        "sum": h.sum,
                        "mean": h.mean,
                        "p99": h.percentile(99),
                        "labels": h.labels,
                    }
                    for k, h in self._histograms.items()
                },
            }

    def reset(self) -> None:
        """Clear all metrics (useful in tests)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# Module-level default registry
_DEFAULT_REGISTRY = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _DEFAULT_REGISTRY
