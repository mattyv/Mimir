# Mimir — Axiom-Grounded World Model Service

Mimir is a batched backend service that crawls internal knowledge sources (Confluence, GitHub, Slack, interviews, code analysis) and builds a shared **world model** expressed as axioms — entities, relationships, observations, and constraints — exposed via the Model Context Protocol (MCP).

## Architecture

```
Sources → Adapters → Extraction Pipeline → Persistence (Postgres + pgvector)
                                               ↓
                            Resolution ← Authority ← Temporal Manager
                                               ↓
                            Cynefin Classifier → Complexity Analyzer
                                               ↓
                                     MCP Tool Layer
```

### Modules

| Module | Phase | Purpose |
|--------|-------|---------|
| `models/` | 1 | Pydantic node types: Entity, Property, Relationship, Observation, Constraint, Process, Decision |
| `vocabulary/` | 1 | ~60 core IRIs (entity types + predicates) in YAML; SHACL shape generation |
| `persistence/` | 2 | Postgres schema, repositories, bitemporal queries, graph version counter, NetworkX projection |
| `resolution/` | 3 | Embedding-based merge candidate detection; atomic entity merger |
| `adapters/` | 4 | Confluence, GitHub, Slack, Interview (YAML), CodeAnalysis (AST) source adapters |
| `crawler/` | 5 | LLM extraction pipeline: prompts → JSON parsing → entity/relationship/observation upsert |
| `authority/` | 6 | Trust scores per source type; property conflict detection and resolution; polarity conflict flagging |
| `cynefin/` | 7 | Cynefin domain classifier: Clear / Complicated / Complex / Chaotic / Confused |
| `complexity/` | 8 | Graph metrics: fan-in/out, cascade risk, density, cycle detection |
| `grounder/` | 9 | Wikidata SPARQL entity linking; polarity enforcement (PolarityViolation guard) |
| `temporal/` | 10 | Expiry, supersession, point-in-time snapshot queries |
| `permissions/` | 11 | ACL evaluation: caller groups vs node visibility; filter_entities; can_write |
| `mcp/` | 12 | MCP tool layer: get_entity, list_entities, classify_entity, list_observations, graph_metrics |
| `normalization/` | 13 | Entity name normalization; provisional IRI promotion (use≥10, sources≥3, approved) |
| `observability/` | 14 | Thread-safe metrics registry (Counter, Histogram); JSON structured logging |
| `eval/` | 15 | Eval harness: load frozen questions, run through LLM, score with judge function |

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.11+ |
| LLM | Gemma 4 via OpenRouter (OpenAI-compatible API) |
| Graph (in-memory) | NetworkX MultiDiGraph |
| Persistence | PostgreSQL + JSONB (psycopg3) |
| Vector | pgvector (VECTOR(384)) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Validation | pyshacl |
| Wikidata | SPARQLWrapper |
| MCP | modelcontextprotocol Python SDK |
| Tests | pytest + hypothesis + respx |
| Lint / type | ruff + mypy --strict |

## Setup

```bash
pip install -e ".[dev,persistence,ml,wikidata,llm,adapters]"
createdb mimir_test
```

## Running Tests

```bash
# All phases with coverage gate (≥95%)
pytest -m "phase0 or phase1 or ... or phase15" --cov=src/mimir --cov-fail-under=95

# Single phase
pytest -m phase5 -v

# Watch mode (requires watchdog)
python watch.py --phases phase5 phase6
```

## Key Design Decisions

**Deterministic entity IDs**: `uuid.uuid5(NAMESPACE_DNS, f"{name}:{type}")` — same entity from any source always maps to the same UUID, enabling idempotent upserts via `ON CONFLICT`.

**Bitemporal data**: Every row carries `valid_from` / `valid_until` columns plus a `graph_version` counter. Point-in-time queries use `as_of: datetime` and `at_version: int`.

**Source authority**: Trust scores (code_analysis=1.0 → slack=0.5) govern conflict resolution. Polarity-opposite predicates (`auros:dependsOn` / `auros:independentOf`) are never auto-merged; conflicts become `inconsistency` Observations.

**Provisional IRIs**: Unknown entity types get `auros:provisional:*` IRIs tracked by use count and source count. Promotion to core IRIs requires ≥10 uses, ≥3 sources, and human approval.

**Test isolation**: Session-scoped schema creation + per-test `ROLLBACK` gives full isolation at ~11× the speed of per-test DDL.
