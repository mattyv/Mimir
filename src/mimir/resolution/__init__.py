"""Mimir entity resolution — embedding, candidate detection, and merging."""

from mimir.resolution.candidates import (
    MergeCandidate,
    find_merge_candidates,
    find_similar_by_embedding,
)
from mimir.resolution.embedder import (
    EMBEDDING_DIM,
    Embedder,
    compute_embedding,
    update_entity_embedding,
)
from mimir.resolution.merger import MergeResult, merge_entities

__all__ = [
    "EMBEDDING_DIM",
    "Embedder",
    "MergeCandidate",
    "MergeResult",
    "compute_embedding",
    "find_merge_candidates",
    "find_similar_by_embedding",
    "merge_entities",
    "update_entity_embedding",
]
