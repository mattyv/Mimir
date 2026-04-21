"""Prefect flow: ingest — pull source chunks and run pipeline (§12.1)."""

from __future__ import annotations

import os
from typing import Any

from prefect import flow, task

from mimir.adapters.base import Chunk
from mimir.crawler.pipeline import PipelineResult, process_chunk
from mimir.persistence.connection import get_pool, init_pool, transaction


@task(retries=2, retry_delay_seconds=10)
def run_chunk(chunk: Chunk, llm: Any) -> PipelineResult:
    """Process one chunk inside its own transaction."""
    pool = get_pool()
    with transaction(pool, isolation="read committed") as conn:
        return process_chunk(chunk, llm, conn)


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
