"""Prefect flow: ingest — pull source chunks and run pipeline (§12.1)."""

from __future__ import annotations

import os
from typing import Any

from prefect import flow, task

from mimir.adapters.base import Chunk
from mimir.adapters.confluence import ConfluenceAdapter
from mimir.adapters.github import GitHubAdapter
from mimir.adapters.version_store import has_changed, set_version
from mimir.crawler.pipeline import PipelineResult, process_chunk
from mimir.persistence.connection import get_pool, init_pool, transaction
from mimir.temporal.retraction import retract_by_source


@task(retries=2, retry_delay_seconds=10)
def run_chunk(chunk: Chunk, llm: Any) -> PipelineResult:
    """Process one chunk inside its own transaction."""
    pool = get_pool()
    with transaction(pool, isolation="read committed") as conn:
        return process_chunk(chunk, llm, conn)


@task(retries=2, retry_delay_seconds=10)
def retract_and_reingest_page(
    page_id: str,
    adapter: ConfluenceAdapter,
    llm: Any,
) -> PipelineResult | None:
    """Re-ingest a Confluence page only if its version number changed.

    Flow:
    1. Cheap metadata fetch → compare version against source_versions table.
    2. Unchanged → return None (skip).
    3. Changed → retract all axioms from the old reference, fetch full content,
       run pipeline, record new version.
    """
    pool = get_pool()
    with transaction(pool, isolation="read committed") as conn:
        version_result = adapter.fetch_page_version(page_id)
        if version_result is None:
            return None
        version_number, reference = version_result
        if not has_changed("confluence", reference, str(version_number), conn):
            return None
        retract_by_source(reference, conn)
        chunk = adapter.fetch_page(page_id)
        if chunk is None:
            return None
        result = process_chunk(chunk, llm, conn)
        set_version("confluence", reference, str(version_number), conn)
        return result


@task(retries=2, retry_delay_seconds=10)
def retract_and_reingest_file(
    owner: str,
    repo: str,
    path: str,
    adapter: GitHubAdapter,
    llm: Any,
    ref: str = "HEAD",
) -> PipelineResult | None:
    """Re-ingest a GitHub file only if its blob SHA changed.

    Flow:
    1. Cheap SHA fetch → compare against source_versions table.
    2. Unchanged → return None (skip).
    3. Changed → retract axioms from the old reference, fetch full content,
       run pipeline, record new SHA.
    """
    pool = get_pool()
    with transaction(pool, isolation="read committed") as conn:
        sha_result = adapter.fetch_file_sha(owner, repo, path, ref)
        if sha_result is None:
            return None
        sha, html_url = sha_result
        if not has_changed("github", html_url, sha, conn):
            return None
        retract_by_source(html_url, conn)
        chunk = adapter.fetch_file(owner, repo, path, ref)
        if chunk is None:
            return None
        result = process_chunk(chunk, llm, conn)
        set_version("github", html_url, sha, conn)
        return result


@flow(name="mimir-ingest", log_prints=True)
def ingest_flow(chunks: list[Chunk], llm: Any) -> list[PipelineResult]:
    """Ingest a batch of chunks into the Mimir knowledge graph."""
    dsn = os.environ.get("DATABASE_URL", "dbname=mimir user=root")
    init_pool(dsn, min_size=2, max_size=8)
    results = []
    for chunk in chunks:
        result = run_chunk(chunk, llm)
        results.append(result)
        print(
            f"chunk={chunk.id} entities={result.entities_upserted} "
            f"rels={result.relationships_inserted} pii_skip={result.pii_skipped}"
        )
    return results
