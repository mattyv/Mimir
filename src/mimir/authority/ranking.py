"""Source authority ranking — trust scores per source type, with per-property overrides."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from mimir.adapters.base import SourceType

_DEFAULTS: dict[SourceType, float] = {
    "code_analysis": 1.0,
    "github": 0.9,
    "confluence": 0.8,
    "interview": 0.7,
    "slack": 0.5,
}

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "source_priority.yaml"


@lru_cache(maxsize=1)
def _load_per_property() -> dict[str, dict[str, float]]:
    """Load per-property priority tables from source_priority.yaml."""
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return {k: dict(v) for k, v in (data.get("per_property") or {}).items()}


def trust_score(source_type: SourceType, property_key: str | None = None) -> float:
    """Return a [0, 1] trust score for *source_type*, optionally scoped to a property key."""
    if property_key:
        per_prop = _load_per_property()
        if property_key in per_prop:
            return per_prop[property_key].get(source_type, _DEFAULTS[source_type])
    return _DEFAULTS[source_type]


def higher_authority(a: SourceType, b: SourceType, property_key: str | None = None) -> SourceType:
    """Return whichever source type has higher trust; *a* wins on tie."""
    return a if trust_score(a, property_key) >= trust_score(b, property_key) else b
