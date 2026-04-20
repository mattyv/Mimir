"""In-process 24h TTL cache for Wikidata grounding results.

Avoids repeated SPARQL calls for the same entity name within a session.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

_TTL_SECONDS = 24 * 3600  # 24 hours


@dataclass
class _CacheEntry:
    qid: str | None  # None = "no match found"
    label: str
    cached_at: float = field(default_factory=time.monotonic)

    def is_expired(self, ttl: float = _TTL_SECONDS) -> bool:
        return (time.monotonic() - self.cached_at) > ttl


_cache: dict[str, _CacheEntry] = {}


def get_cached(name: str) -> tuple[bool, str | None, str]:
    """Return (hit, qid, label). hit=False means cache miss or expired."""
    entry = _cache.get(name.casefold())
    if entry is None or entry.is_expired():
        return False, None, ""
    return True, entry.qid, entry.label


def put_cached(name: str, qid: str | None, label: str = "") -> None:
    """Store a result (qid=None means no match). TTL starts now."""
    _cache[name.casefold()] = _CacheEntry(qid=qid, label=label)


def cache_size() -> int:
    """Return current number of cached entries (including expired)."""
    return len(_cache)


def clear_cache() -> None:
    """Evict all entries. Used in tests."""
    _cache.clear()
