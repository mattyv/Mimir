"""Prefect flow: retract — expire axioms whose source chunks no longer exist (§12.1)."""

from __future__ import annotations

import os
from typing import Any

from prefect import flow, task

from mimir.persistence.connection import get_pool, init_pool, transaction
from mimir.temporal.retraction import (
    RetractionResult,
    list_active_source_refs,
    retract_by_source,
)


@task
def check_and_retract(source_ref: str, checker: Any) -> RetractionResult | None:
    if checker.exists(source_ref):
        return None
    pool = get_pool()
    with transaction(pool, isolation="read committed") as conn:
        return retract_by_source(source_ref, conn)


@flow(name="mimir-retract", log_prints=True)
def retract_flow(checker: Any, *, source_type: str | None = None) -> int:
    """Scan for deleted source references and expire their axioms."""
    dsn = os.environ.get("DATABASE_URL", "dbname=mimir user=root")
    init_pool(dsn, min_size=2, max_size=4)
    pool = get_pool()
    with pool.connection() as conn:
        refs = list_active_source_refs(conn, source_type=source_type)

    total_expired = 0
    for ref in refs:
        result = check_and_retract(ref, checker)
        if result is not None:
            total_expired += result.entities_expired
            print(
                f"retracted source={ref!r} "
                f"entities={result.entities_expired} "
                f"rels={result.relationships_expired}"
            )

    print(f"retract_flow complete: {total_expired} entities expired across {len(refs)} refs checked")
    return total_expired
