"""Prefect flow: ground — run Wikidata grounding on ungrounded entities (§12.1)."""

from __future__ import annotations

import os
from typing import Any

from prefect import flow, task

from mimir.grounder.wikidata import ground_entity_recursive
from mimir.persistence.connection import get_pool, init_pool, transaction


@task(retries=2, retry_delay_seconds=30)
def ground_entity_task(entity_id: str, name: str, sparql_client: Any) -> bool:
    pool = get_pool()
    with transaction(pool, isolation="read committed") as conn:
        result = ground_entity_recursive(entity_id, name, sparql_client, conn)
        return len(result) > 0


@flow(name="mimir-ground", log_prints=True)
def ground_flow(sparql_client: Any) -> int:
    """Ground all ungrounded entities via Wikidata SPARQL."""
    dsn = os.environ.get("DATABASE_URL", "dbname=mimir user=root")
    init_pool(dsn, min_size=2, max_size=4)
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name FROM entities
             WHERE valid_until IS NULL
               AND (payload->>'wikidata_qid') IS NULL
            ORDER BY name
            LIMIT 500
            """
        ).fetchall()

    grounded = 0
    for row in rows:
        success = ground_entity_task(row["id"], row["name"], sparql_client)
        if success:
            grounded += 1
            print(f"grounded entity={row['id']} name={row['name']!r}")

    print(f"ground_flow complete: {grounded}/{len(rows)} entities grounded")
    return grounded
