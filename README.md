# Mimir

Mimir crawls your internal knowledge sources — Confluence pages, GitHub repos, Slack channels, engineering interviews, and static code analysis — and builds a **shared world model** of your organisation: what services exist, who owns them, how they depend on each other, what risks they carry, and why decisions were made.

The world model is a typed knowledge graph stored in Postgres. It is exposed to LLMs via the Model Context Protocol (MCP), so Claude (or any MCP-aware tool) can answer questions like *"what does the options market-making service depend on?"*, *"which services have unresolved SLO-breach risks?"*, or *"what changed in the risk architecture last quarter?"* — grounded in your actual internal documentation rather than hallucinated from training data.

---

## How it works

### 1. Crawling: turning documents into chunks

Each source adapter fetches raw content and wraps it in a `Chunk`:

```python
Chunk(
    id="confluence_001",
    source_type="confluence",
    content="The OMMS service handles options market making...",
    acl=["space:trading-eng"],
    retrieved_at=datetime(...),
    reference="https://wiki.example.com/...",
)
```

Five adapters are supported: **Confluence** (REST API, HTML-stripped), **GitHub** (file/README fetch, base64-decoded), **Slack** (channel history, chronological), **Interview** (YAML transcripts), **Code analysis** (Python AST — classes, functions, imports, cyclomatic complexity).

### 2. Extraction: LLM turns chunks into structured facts

Each chunk is sent to an LLM (Gemma 4 via OpenRouter) with a constrained extraction prompt. The LLM must return JSON with four keys:

```json
{
  "entities":      [{"name": "OMMS", "type": "auros:TradingService", "description": "..."}],
  "properties":    [{"entity_name": "OMMS", "key": "schema:name", "value": "OMMS"}],
  "relationships": [{"subject": "OMMS", "predicate": "auros:dependsOn", "object": "FIX Connector"}],
  "observations":  [{"entity_name": "OMMS", "type": "risk", "description": "SLO breach risk at p99"}]
}
```

Entity types and predicates are constrained to the core vocabulary (see below), so the graph stays clean. Unknown concepts get provisional IRIs (`auros:provisional:*`) which can be promoted later.

### 3. The world model: a typed knowledge graph

Everything extracted lands in Postgres as one of seven node types:

| Node | What it represents |
|------|--------------------|
| `Entity` | A named thing: a service, team, person, venue, strategy, risk system, … |
| `Property` | A key/value fact about an entity (owner, language, SLO threshold, …) |
| `Relationship` | A typed, directed edge between two entities (`auros:dependsOn`, `auros:owns`, …) |
| `Observation` | A qualitative judgement: `risk`, `anti_pattern`, `maturity`, `opportunity`, `inconsistency`, … |
| `Constraint` | A hard rule on an entity (performance, availability, legal, …) |
| `Process` | A multi-stage workflow with inputs and outputs |
| `Decision` | A recorded architectural decision — what, why, tradeoffs, who, when |

Every node carries:
- **Grounding** — how trustworthy is this fact? (`source_cited` → `wikidata_linked` → `fully_grounded`)
- **Temporal** — `valid_from` / `valid_until` so the graph is a time-series, not a snapshot
- **Visibility** — ACL list + sensitivity level; MCP queries are filtered per-caller
- **Source** — back-link to the original chunk and its URL

### 4. Entity resolution: deduplication across sources

The same service might be called "OMMS" in Confluence, "omms" in GitHub, and "options MM" in a Slack message. The resolver:

1. Computes sentence embeddings for each entity (all-MiniLM-L6-v2, 384 dims stored as pgvector)
2. Finds merge candidates via cosine similarity self-join (`(1 - embedding <=> other) >= 0.85`)
3. Merges confirmed pairs atomically — rerouting all properties, observations, and relationships to the surviving entity

Entity IDs are deterministic (`uuid5(NAMESPACE_DNS, f"{name}:{type}")`), so the same name+type from any source always produces the same UUID, and re-ingesting a chunk is always idempotent.

### 5. Source authority and conflict resolution

Different sources are trusted differently:

| Source | Trust score |
|--------|-------------|
| Code analysis | 1.0 |
| GitHub | 0.9 |
| Confluence | 0.8 |
| Interview | 0.7 |
| Slack | 0.5 |

When two sources disagree on a property value, the lower-authority row is expired (its `valid_until` is set). When two sources assert opposite-polarity predicates simultaneously — e.g. both `auros:dependsOn` and `auros:independentOf` between the same pair — the conflict is flagged as an `inconsistency` Observation rather than silently discarded, and polarity merges are blocked at the code level.

### 6. Classification and complexity

Every entity is automatically classified into a **Cynefin domain** based on its observations and relationship count:

- **Chaotic** — has `inconsistency` or `anti_pattern` observations, or very low confidence
- **Complex** — has `risk`, `smell`, or `opportunity` observations, or highly coupled (≥10 relationships)
- **Complicated** — has `maturity` or `functional_state` observations, or moderately coupled (≥5)
- **Clear** — no observations, high confidence
- **Confused** — everything else

The graph complexity analyser adds whole-graph metrics: fan-in/out per node, cascade risk (fraction of the graph reachable downstream from a node), cycle detection, density, and strongly-connected component count.

### 7. Temporal management

The graph is fully bitemporal. Every write records `valid_from`. Expiry sets `valid_until`. Supersession records the replacement entity ID in `payload.superseded_by`. You can ask:

- *What did the graph look like on a specific date?* (`active_entities_at(conn, as_of=date)`)
- *What has been expired?* (`expired_entities(conn)`)
- *What was the graph at version N?* (`at_version=N` on any query)

### 8. MCP exposure

Five tools are exposed to LLM clients over MCP:

| Tool | What it does |
|------|-------------|
| `get_entity` | Fetch a single entity by ID, ACL-filtered |
| `list_entities` | List active entities, optionally filtered by type, ACL-filtered |
| `classify_entity` | Return the Cynefin domain + metrics for an entity |
| `list_observations` | Return qualitative observations for an entity, optionally by type |
| `graph_metrics` | Return whole-graph complexity metrics |

Every response is filtered against the caller's group memberships before being returned.

---

## Vocabulary

The core vocabulary lives in `src/mimir/vocabulary/vocabulary.yaml`. It defines ~20 entity types and ~30 predicates, all as IRIs in either the `schema:` (schema.org) or `auros:` (domain-specific) namespaces.

Unknown concepts extracted by the LLM are assigned provisional IRIs (`auros:provisional:<name>`). These are tracked by use count and source count. A provisional IRI can be promoted to a core IRI once it has been seen ≥10 times across ≥3 distinct sources, with human approval.

SHACL shapes are generated from the vocabulary and used to validate axioms before they are written.

---

## Setup

```bash
# Install
pip install -e ".[dev,persistence,ml,wikidata,llm,adapters]"

# Create the test database
createdb mimir_test

# Run all tests
pytest --cov=src/mimir --cov-fail-under=95

# Watch mode during development
python watch.py --phases phase5 phase6
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | API key for LLM extraction (Gemma 4 via OpenRouter) |
| `DATABASE_URL` | Postgres connection string (default: `dbname=mimir user=root`) |

---

## Project layout

```
src/mimir/
├── models/          # Pydantic node types + IRI validation
├── vocabulary/      # Core IRI vocabulary (YAML) + SHACL shapes
├── persistence/     # Postgres schema, repositories, graph projection (NetworkX)
├── adapters/        # Source adapters: Confluence, GitHub, Slack, Interview, Code
├── crawler/         # LLM client, extraction prompt, pipeline orchestration
├── resolution/      # Embedding computation, merge candidate detection, merger
├── authority/       # Source trust scores, conflict detection + resolution
├── cynefin/         # Domain classifier (heuristic, pure function)
├── complexity/      # Graph metrics: cascade risk, density, coupling
├── grounder/        # Wikidata entity linking, polarity enforcement
├── temporal/        # Expiry, supersession, point-in-time queries
├── permissions/     # ACL evaluation, entity filtering
├── mcp/             # MCP tool definitions
├── normalization/   # Name normalisation, IRI normalisation, provisional promotion
├── observability/   # Metrics registry, JSON structured logging
└── eval/            # Eval harness for frozen question set
```
