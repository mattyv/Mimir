"""Prefect flow: normalize — run entity resolution on the current graph (§12.1)."""

from __future__ import annotations

import os

from prefect import flow, task

from mimir.persistence.connection import get_pool, init_pool, transaction
from mimir.resolution.candidates import find_merge_candidates, get_thresholds
from mimir.resolution.merger import merge_entities
from mimir.resolution.queue import enqueue_candidate


@task
def resolve_batch() -> tuple[int, int]:
    """Find candidates, auto-merge above threshold, enqueue borderline pairs."""
    pool = get_pool()
    auto_merged = 0
    enqueued = 0

    with transaction(pool, isolation="repeatable read") as conn:
        auto_threshold, review_threshold = get_thresholds(conn)
        candidates = find_merge_candidates(conn, threshold=review_threshold, limit=200)

    for candidate in candidates:
        with transaction(pool, isolation="repeatable read") as conn:
            try:
                if candidate.similarity >= auto_threshold:
                    merge_entities(candidate.entity_a_id, candidate.entity_b_id, conn)
                    auto_merged += 1
                else:
                    enqueue_candidate(
                        candidate.entity_a_id,
                        candidate.entity_b_id,
                        candidate.similarity,
                        conn,
                    )
                    enqueued += 1
            except (ValueError, Exception):
                pass

    return auto_merged, enqueued


@flow(name="mimir-normalize", log_prints=True)
def normalize_flow() -> None:
    """Run entity resolution: merge duplicates and queue borderline pairs."""
    dsn = os.environ.get("DATABASE_URL", "dbname=mimir user=root")
    init_pool(dsn, min_size=2, max_size=4)
    auto_merged, enqueued = resolve_batch()
    print(f"normalize_flow complete: auto_merged={auto_merged} enqueued={enqueued}")
