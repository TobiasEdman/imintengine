# TrustGraph Assessment — ImintEngine + Space Ecosystem

*Date: 2025-03-25*
*Repo: https://github.com/trustgraph-ai/trustgraph*

---

## Summary

TrustGraph is an open-source "Context Development Platform" built on Apache Cassandra + Qdrant + Neo4j + Apache Pulsar. It provides automated knowledge graph construction, GraphRAG, OntologyRAG, provenance tracking, and multi-agent orchestration.

**Verdict: Interesting but premature for integration. Worth watching.**

---

## Assessment for ImintEngine (des-agent RAG)

### Potential Benefits
- Knowledge graphs could model Sentinel-2 sensor/band/index/use-case relationships better than flat vector embeddings
- Provenance tracking maps naturally to geospatial data lineage
- OntologyRAG could improve retrieval precision for DES-specific queries
- MCP integration available for easy bridging

### Why Not Now
- **No stable releases** — zero version tags, running from master
- **Heavy infrastructure** — Cassandra + Pulsar + Qdrant + Garage would 3-5x operational footprint
- **Single-maintainer risk** — concentrated development (user "cybermaggedon")
- des-agent RAG already works — incremental benefit may not justify integration cost

### Recommendation
Monitor the project (star it, join Discord). Consider a lightweight KG experiment (e.g. just Neo4j + des-agent) to test if graph-based retrieval actually improves domain-specific queries before adopting the full platform.

---

## Assessment for swedish-space-ecosystem-v2

### Context
The space-ecosystem-v2 project **already IS a knowledge graph** on Neo4j with:
- ~5000 publications, ~2000 researchers, ~80 companies, ~100 institutions
- 12 node types, 9+ relationship types
- Full graph traversal via Cypher
- 7 visualization views with Pydantic-validated export pipeline
- Clean ETL: extractors -> loaders -> exporters

### Why TrustGraph Would Be Overkill
1. **Would trade Neo4j + Cypher for Cassandra** — losing 47K+ bytes of carefully crafted Cypher queries in `view_exporter.py`
2. **6-7 extra containers** (Cassandra, Pulsar, Qdrant, Garage, Prometheus, Grafana) for a graph with ~5000 nodes
3. **Full ETL rewrite** — current pipeline does direct MERGE into Neo4j; TrustGraph wants everything via Pulsar topics
4. **No stable releases** — risky for a Vinnova proposal deliverable

### Lighter Alternatives (Recommended)

#### Option A: Neo4j + LLM-to-Cypher Layer (Minimal Change) ~1-2 days
- Keep Neo4j exactly as-is
- Add a thin Python service that uses an LLM (Claude/GPT) to translate natural language questions into Cypher queries
- Use `neo4j-graphrag` (official Neo4j Python library for GraphRAG)
- Infrastructure: zero new services

#### Option B: Neo4j Built-in Vector Search (Small Addition) ~2-3 days
- Neo4j 5.11+ has built-in vector search support
- Generate embeddings for entity descriptions and store as node properties
- Enables hybrid graph+semantic search without additional infrastructure
- Infrastructure: zero new services

#### Option C: Extend des-agent Qdrant (Already Available) ~3-5 days
- des-agent already runs Qdrant
- Extend the existing Qdrant instance to index space ecosystem entities alongside code
- Build a simple query interface that searches both Qdrant (semantic) and Neo4j (graph)
- Infrastructure: already running

#### Option D: LlamaIndex Knowledge Graph Integration (Moderate) ~1 week
- des-agent already uses LlamaIndex
- LlamaIndex has `KnowledgeGraphIndex` that can sit on top of Neo4j
- Would provide GraphRAG-like capabilities with the existing infrastructure
- Infrastructure: existing des-agent + existing Neo4j

#### Option E: CrewAI Domain Agents (Already Have Foundation) ~1-2 weeks
- des-agent already uses CrewAI
- Build domain-specific agents that can query Neo4j, interpret results, and answer questions about the ecosystem
- Could answer questions like "What are Sweden's key capabilities in satellite propulsion?"
- Infrastructure: existing

### Bottom Line
**Option A or B** would give 80% of TrustGraph's value with ~5% of the infrastructure complexity. Start there.

---

## TrustGraph Repo Health (March 2025)

| Metric | Status |
|--------|--------|
| Last commit | March 21, 2026 (active) |
| Commit frequency | Daily, multiple per day |
| Open issues | 13 (manageable) |
| Formal releases | **None** (concerning) |
| Stars / Forks | 1,500 / 137 |
| License | Apache 2.0 |
| Test infrastructure | Present |
| Documentation | External site + 46 tech specs |

### Key Features
- Multi-model DB: Cassandra
- Vector Store: Qdrant (also Milvus, Pinecone, FAISS)
- Graph Store: Neo4j, FalkorDB
- Message Bus: Apache Pulsar
- LLM Support: Claude, OpenAI, Gemini, Ollama, vLLM, Bedrock, Azure, Mistral, Cohere
- 3 RAG pipelines: DocumentRAG, GraphRAG, OntologyRAG
- "Context Cores": portable versioned knowledge bundles
- MCP server integration (`trustgraph-mcp`)
- Provenance tracking (active development)
