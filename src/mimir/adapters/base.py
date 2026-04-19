"""Shared data model for all source adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["confluence", "github", "slack", "interview", "code_analysis"]


class Chunk(BaseModel):
    """A normalized content unit produced by a source adapter.

    All adapters return Chunk objects.  The crawler (Phase 5) consumes them
    to extract entities and relationships.
    """

    id: str
    source_type: SourceType
    content: str
    acl: list[str] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reference: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
