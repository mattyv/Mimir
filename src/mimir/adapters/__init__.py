"""Mimir source adapters — fetch raw content from external systems."""

from mimir.adapters.base import Chunk, SourceType
from mimir.adapters.code_analysis import CodeAnalysisAdapter
from mimir.adapters.confluence import ConfluenceAdapter
from mimir.adapters.github import GitHubAdapter
from mimir.adapters.interview import InterviewAdapter
from mimir.adapters.pii import PIIFinding, PIIScanResult, scan_chunk
from mimir.adapters.slack import SlackAdapter

__all__ = [
    "Chunk",
    "CodeAnalysisAdapter",
    "ConfluenceAdapter",
    "GitHubAdapter",
    "InterviewAdapter",
    "PIIFinding",
    "PIIScanResult",
    "SlackAdapter",
    "SourceType",
    "scan_chunk",
]
