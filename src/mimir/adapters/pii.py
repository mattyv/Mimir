"""PII / secret scanning gate for Chunk content.

Uses detect-secrets to flag chunks that contain credentials, tokens, or
other sensitive strings before they enter the extraction pipeline.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

from detect_secrets import SecretsCollection  # type: ignore[attr-defined]
from detect_secrets.settings import transient_settings

from mimir.adapters.base import Chunk

_PLUGINS: list[dict[str, str]] = [
    {"name": "AWSKeyDetector"},
    {"name": "BasicAuthDetector"},
    {"name": "KeywordDetector"},
    {"name": "PrivateKeyDetector"},
    {"name": "SlackDetector"},
]


@dataclass
class PIIFinding:
    secret_type: str
    line_number: int


@dataclass
class PIIScanResult:
    chunk_id: str
    has_secrets: bool
    findings: list[PIIFinding] = field(default_factory=list)


def scan_chunk(chunk: Chunk) -> PIIScanResult:
    """Scan *chunk.content* for secrets using detect-secrets.

    Writes content to a temp file (detect-secrets works file-based),
    runs the configured plugins, then removes the temp file.
    """
    findings: list[PIIFinding] = []
    suffix = ".txt"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(chunk.content)
        with transient_settings({"plugins_used": _PLUGINS}):
            sc = SecretsCollection()
            sc.scan_file(tmp_path)
            for _path, secret in sc:
                findings.append(
                    PIIFinding(
                        secret_type=secret.type,
                        line_number=secret.line_number,
                    )
                )
    finally:
        os.unlink(tmp_path)
    return PIIScanResult(
        chunk_id=chunk.id,
        has_secrets=bool(findings),
        findings=findings,
    )
