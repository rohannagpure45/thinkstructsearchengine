# Hybrid Patent Search Starter

A patent search engine using Elasticsearch's hybrid search capabilities, combining BM25 lexical search with both dense vector (kNN) and sparse vector (ELSER v2) semantic search.


## Problem statement and Goal

1. **Goals**
- Build a patent search that “gets meaning,” not just words: Supports Natural Langage queries and claim-text queries over Titles/Abstracts/Claims/Desc.
- Hybrid ranking that merges BM25 lexical hits with semantic candidates.
- If time permits, add reranking later via Elastic’s Inference API (Cohere/Voyage/HF/Jina) and add image/structure vectors for diagrams.

2. **Enhancement**
- Cross‑encoder semantic re‑ranking via Elastic’s Inference API.
- Boosts Top‑K precision by letting a model re‑read each (query, doc) pair and reorder candidates

3. **How the Code Addresses the Problem**
- Hybrid retrieval: BM25 + dense + ELSER using Reciprocal Rank Fusion (RRF)
- Multi‑index mapping built for claims & paragraphs
- ELSER pipelines per section

## Features

- **Multi-index architecture**: Separate indices for patents, claims, and descriptions
- **Triple fusion search**: BM25 + Dense Vectors + ELSER v2 using RRF (Reciprocal Rank Fusion)
- **ML-powered**: Leverages Elasticsearch's ML capabilities for ELSER and external sentence transformers
- **Production-ready**: Docker-based setup with proper configurations and error handling
- **Extensible**: Ready for reranking, filtering, and image search in Phase 2

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- 8GB RAM minimum (4GB for Elasticsearch)

## Quick Start

1. **Clone and setup environment**
   ```bash
   cp env.example .env
   # Edit .env with your configurations
   ```

2. **Start Elasticsearch**
   ```bash
   docker-compose up -d
   # Wait for Elasticsearch to be healthy
   docker-compose logs -f elasticsearch
   ```

3. **Install Python dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Setup ML models and indices**
   ```bash
   # Deploy ELSER model
   python scripts/setup_ml_models.py
   
   # Create indices
   ./scripts/create_indices.sh
   
   # Register ELSER pipeline
   ./scripts/register_elser_pipeline.sh
   ```

5. **Ingest sample data**
   ```bash
   python scripts/ingest_data.py --input /path/to/patent/data.json
   ```

6. **Start the API**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Usage

### Basic Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "anti-rotation composite spoke wheel rim",
    "top_k": 20
  }'
```

### Search with Reranking (Phase 2)
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "molecular structure detection optical",
    "top_k": 20,
    "rerank": true
  }'
```

### Inspecting Ranked Results with `jq`

The `/search` endpoint aggregates hits by patent by default (`aggregate_by_patent: true`), so the response contains an `aggregated_results` array ordered by total score. Use `jq` to pretty-print the ranking and top-level scores:

```bash
curl -s "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
        "q": "anti-rotation composite spoke wheel rim",
        "top_k": 10,
        "search_scope": "all",
        "use_embeddings": true,
        "filters": null,
        "aggregate_by_patent": true
      }' \
  | jq '{total_hits,
         search_time_ms,
         aggregated_results: [((.aggregated_results // []) | to_entries[] |
           {rank: (.key + 1),
            patent_id: .value.patent_id,
            total_score: .value.total_score,
            best_score: .value.best_score,
            hit_count: .value.hit_count,
            title: .value.title})]}'
```

- Change `aggregate_by_patent` to `false` if you want the raw `hits` array instead of aggregated patent groups.
- Pipe the JSON through `python -m json.tool` or another formatter if `jq` is unavailable.

## Architecture

### Indices
- **patents_core**: Patent metadata, titles, and abstracts
- **claims_chunks**: Individual patent claims with embeddings
- **desc_chunks**: Description paragraphs with embeddings

### Search Strategy
1. **BM25**: Traditional text matching on all text fields
2. **Dense Vectors**: Semantic similarity using sentence-transformers
3. **ELSER v2**: Elastic's sparse vector semantic search
4. **RRF Fusion**: Combines all three approaches for optimal ranking

## Configuration

See `env.example` for all configuration options including:
- Elasticsearch connection settings
- Index names
- Model configurations
- Search parameters (RRF constants, window sizes)

## Development

### Running Tests
```bash
pytest tests/
```

### Linting
```bash
black app/ scripts/ tests/
isort app/ scripts/ tests/
flake8 app/ scripts/ tests/
mypy app/ scripts/ tests/
```

## Monitoring

Access Kibana at http://localhost:5601 for:
- Index management
- Search analytics
- ML model monitoring
- Query debugging

## Roadmap

- [x] Phase 1: Basic hybrid search with BM25 + ELSER + Dense vectors
- [ ] Phase 2: Cross-encoder reranking
- [ ] Phase 2: Advanced filtering and faceting
- [ ] Phase 3: Image similarity search
- [ ] Phase 3: Chemical structure search
