# Mimir — Axiom-Grounded World Model

Mimir is a batched backend service that crawls a domain (Confluence, GitHub, Slack, code repos, interviews) and builds a shared **world model** expressed as axioms — grounded, time-stamped facts about entities, their relationships, and observations about them. The world model is exposed over the Model Context Protocol (MCP) so any LLM client can query it.

---

## How It Works

### The Data Model (Phases 0–1)

Everything in Mimir is a typed **node** carrying four cross-cutting value objects:

| Value object | Purpose |
|---|---|
| `Grounding` | How well the fact is anchored: `ungrounded` → `source_cited` → `schema_typed` → `wikidata_linked` → `fully_grounded` |
| `Temporal` | When the fact was true (`valid_from` / `valid_until`); `None` means still true |
| `Visibility` | ACL list + sensitivity label (`public` / `internal` / `restricted`) |
| `Source` | Where the fact came from (Confluence page, GitHub repo, Slack channel, interview, code analysis) |

The seven node types are:

- **Entity** — a named thing in the domain (trading service, team, person, venue…)
- **Property** — a key/value attribute of an entity (e.g. `schema:name = "OMMS"`)
- **Relationship** — a directed edge between two entities (`subject → predicate → object`)
- **Observation** — a qualitative finding attached to an entity (strength, risk, anti-pattern, maturity…)
- **Constraint** — a bound on an entity (performance SLO, legal, availability…)
- **Process** — a sequence of stages with inputs, outputs, and an optional SLO
- **Decision** — an architectural decision with rationale, trade-offs, and participants

All entity types and predicates must be IRIs from the core vocabulary (`schema:` or `auros:` namespace) or provisional IRIs (`auros:provisional:*`). Free strings are rejected at model construction time. SHACL shapes generated from the vocabulary enforce domain/range constraints.

**Polarity system**: predicates can be declared as positive (`auros:dependsOn`) or negative (`auros:independentOf`) with explicit opposites. The grounder (Phase 9) uses this to detect contradictions.

### The Persistence Layer (Phase 2)

All nodes are stored in PostgreSQL with JSONB payloads. The schema supports two independent time axes:

**Bitemporal queries**

- `as_of: datetime` — filters by the *real-world* validity interval (`valid_from ≤ as_of < valid_until`). Pass `None` to get only currently-active rows.
- `at_version: int` — filters by the *graph version* stamp written at commit time. Pass a past version number to replay the graph as it looked after that write.

This means you can ask: *"What did the graph say about team X as of last Tuesday, based on data ingested before version 500?"*

**Graph version counter**

Every write transaction atomically increments a single row in `graph_meta` using `UPDATE … RETURNING version`. Writers serialize on this row-level lock; no two transactions get the same version. This gives a total ordering of all writes without a distributed clock.

**Concurrency-safe deduplication**

Entities are deduplicated at insert time via a `UNIQUE INDEX ON entities (name_normalized, entity_type)`. The upsert uses `ON CONFLICT … DO UPDATE`, so concurrent crawlers writing the same entity from different sources converge to a single row.

**NetworkX projection**

`build_graph(conn, as_of, at_version)` projects the Postgres rows into a NetworkX `MultiDiGraph` for in-memory graph algorithms (ego-graphs, path queries, centrality). All bitemporal filters apply before projection.

### Entity Resolution (Phase 3)

After ingestion, the same real-world thing may appear under slightly different names across sources ("Options Market Making Service" vs "Options MM Service"). Phase 3 resolves these duplicates.

**Embeddings**

`compute_embedding(text, embedder)` runs text through a sentence-transformer (production: `all-MiniLM-L6-v2`, tests: `FakeEmbedder`) and returns a unit-normalized 384-dim vector. `update_entity_embedding(entity_id, embedding, conn)` stores the vector in the `entities.embedding` column (a `VECTOR(384)` pgvector column) using a string-literal cast — no Python pgvector registration required.

**Candidate detection**

- `find_similar_by_embedding(embedding, entity_type, conn, threshold=0.85)` — searches active entities of the given type using pgvector's cosine-distance operator (`<=>`) and returns rows above the similarity threshold, sorted by descending similarity.
- `find_merge_candidates(conn, entity_type=None, threshold=0.85)` — self-join across all active entities sharing the same type and both having stored embeddings. Returns `MergeCandidate` dataclasses with entity pair IDs, similarity score, and detection method.

**Merge execution**

`merge_entities(kept_id, dropped_id, conn)` performs an atomic merge inside the caller's transaction:

1. Validates both entities exist and are active.
2. Reroutes all **properties**, **observations**, and **relationships** from the dropped entity to the kept entity (simple `UPDATE … SET entity_id/subject_id/object_id`).
3. Deletes any **self-referential relationships** created when a relationship that connected the two entities becomes a loop after rerouting.
4. Expires the dropped entity: sets `valid_until = NOW()` and records `superseded_by = kept_id` in the JSONB payload.
5. Bumps the **graph version** once.

Returns a `MergeResult` with counts of rerouted rows and the new graph version.

---

## Project Structure

```
src/mimir/
├── models/
│   ├── base.py          # Grounding, Source, Temporal, Visibility
│   ├── nodes.py         # Entity, Property, Relationship, Observation, Constraint, Process, Decision
│   └── iri.py           # IRI validation, provisional IRIs, namespace extraction
├── vocabulary/
│   ├── vocabulary.yaml  # ~60 core IRIs (entity types + predicates + polarity pairs)
│   ├── loader.py        # load_vocabulary(), ProvisionalTracker
│   └── shacl.py         # SHACL shape generation + pyshacl validation
├── persistence/
│   ├── schema.py        # DDL (CREATE TABLE, pgvector extension)
│   ├── connection.py    # ConnectionPool, transaction() context manager
│   ├── graph_version.py # bump_graph_version(), current_graph_version()
│   ├── repository.py    # EntityRepository, PropertyRepository, RelationshipRepository, ObservationRepository
│   └── graph_projection.py  # build_graph(), subgraph_for_entity()
└── resolution/
    ├── embedder.py      # Embedder protocol, compute_embedding(), update_entity_embedding()
    ├── candidates.py    # find_similar_by_embedding(), find_merge_candidates(), MergeCandidate
    └── merger.py        # merge_entities(), MergeResult
```

---

## Development

**Prerequisites**: Python 3.11+, PostgreSQL with pgvector extension, a `mimir_test` database owned by the current user.

```bash
pip install -e ".[persistence,dev]"
```

**Gates (must all pass before committing)**:

```bash
ruff check src/ tests/          # lint
python -m mypy --strict src/    # types
python -m pytest -m "phase0 or phase1 or phase2 or phase3" \
    --cov=src/mimir --cov-fail-under=95
```

**Alembic migrations**:

```bash
MIMIR_DATABASE_URL=postgresql+psycopg://user@/mimir alembic upgrade head
```

---

## Phases

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ | Scaffolding, CI, fixtures, frozen eval questions |
| 1 | ✅ | Schema, vocabulary, temporal model, SHACL |
| 2 | ✅ | Persistence layer — repositories, migrations, bitemporal queries |
| 3 | ✅ | Entity resolution — embeddings, candidate detection, merge |
| 4 | 🔲 | Source adapters (Confluence, GitHub, Slack, interview, code) |
| 5 | 🔲 | Crawler extraction (Gemma 4 via OpenRouter) |
| 6 | 🔲 | Source authority and conflict resolution |
| 7 | 🔲 | Cynefin classifier |
| 8 | 🔲 | Complexity analyzer |
| 9 | 🔲 | Grounder (Wikidata linking, polarity enforcement) |
| 10 | 🔲 | Temporal management |
| 11 | 🔲 | Permissions and ACL |
| 12 | 🔲 | MCP server |
| 13 | 🔲 | Normalization and promotion |
| 14 | 🔲 | Observability |
| 15 | 🔲 | Eval harness |
