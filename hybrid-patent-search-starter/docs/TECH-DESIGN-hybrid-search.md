Below is a **technical foundation document** you can drop into your repo’s `/docs/` folder as `TECH-DESIGN-hybrid-search.md`. It specifies the **tech stack, data model, index mappings, query plans, and operational knobs** for a hybrid search engine that **combines semantic transformer-based retrieval with BM25** using Elastic’s current, production-grade capabilities (Elasticsearch + Elastic Labs patterns). I’ve also included **version notes**, **code-ready JSON payloads**, and **cut-over options** for the newest features (ELSER v2, `semantic_text`, RRF retrievers, inference/rerankers).

---

# Hybrid Search in Elasticsearch (BM25 + Semantic) — Technical Design

**Audience:** engineers building the patent search MVP
**Scope:** natural-language & claim-text queries; returns patents and/or claims
**Targets:** fast baseline relevance, explainability, and a path to image & molecule expansion
**Elastic versions:** 8.19+ (works today); notes for 8.16+ retrievers & 9.0+ `semantic_text` GA

---

## 1) Goals & Non‑Goals

**Goals**

* Support **NL queries** and **claim-text queries** over Titles/Abstracts/Claims/Desc.
* **Hybrid ranking** that merges BM25 lexical hits with semantic candidates.
* **Production-minded**: one index per “chunk type” (claims, description paragraphs) to support highlighting/snippets.
* **Extendable**: add **reranking** later via Elastic’s Inference API (Cohere/Voyage/HF/Jina) and add **image/structure** vectors for diagrams.

**Non‑Goals (Phase 1)**

* Full visual OCSR pipeline for molecules (outline provided, not implemented here).
* Custom embedding model training; we use off‑the‑shelf ELSER v2 or public text‑embedding models.

---

## 2) Why Elastic + Labs Patterns

* **RRF hybrid search built-in:** Elasticsearch exposes **Reciprocal Rank Fusion (RRF)** directly in the Search API to fuse **BM25** and **kNN (dense)** or **sparse semantic** result sets into one ranked list, with **`rank_constant`** and **`rank_window_size`** knobs. This avoids manual score blending and is robust across signals. ([Elastic][1])
* **kNN vector search native:** `dense_vector` with HNSW index, **`num_candidates`** tuning, quantization (byte/INT8) for memory, and **filtered kNN**—all documented and supported. ([Elastic][2])
* **ELSER v2 (sparse) and `text_expansion`/`sparse_vector`:** Elastic’s **ELSER v2** enables semantic retrieval **without managing dense embeddings**; tokens+weights are stored in a sparse field, queried via **`sparse_vector`** (or legacy `text_expansion`). ([Elastic][3])
* **`semantic_text` (new, GA) simplification:** a single mapping type that **auto-embeds at index & query time** via an inference endpoint; it supports **BM25, kNN, and sparse** queries on the same field for future-proof hybrid. (Use if your stack is ≥ 9.0). ([Elastic][4])
* **ESRE** (Elasticsearch Relevance Engine): a product umbrella for vector DB, ML inference, ELSER, connectors, and LLM integrations—this design uses the same building blocks ESRE formalizes. ([Elastic][5])
* **Elastic Labs tutorials & notebooks** document the exact hybrid patterns, including **RRF** and **ELSER** examples. ([Elastic][6])

**Industry context:** Hybrid search is the pragmatic “best of both worlds,” balancing BM25’s precision with semantic recall; see Aman’s systems write‑up for the conceptual rationale & fusion patterns. ([Aman AI][7])

---

## 3) Architecture (MVP)

```
                +---------------------------+
Query (NL or    |  App/API (FastAPI/Node)  |
claim text) --->|  - Query builder         |
                |  - Retrievers (ES client)|
                +---------------------------+
                           |
                           v
                 +---------------------+
                 |  Elasticsearch      |
                 |  Patent Indexes:    |
                 |   - patents_core    |  (one doc per patent)
                 |   - claims_chunks   |  (one doc per claim)
                 |   - desc_chunks     |  (one doc per paragraph)
                 |                     |
                 |  Fields:            |
                 |   - text: BM25      |
                 |   - vectors:        |
                 |       dense (kNN)   |
                 |       or ELSER v2   |
                 |   - metadata, codes |
                 +----------+----------+
                            ^
   Ingestion (JSON)         |
   - Parse PDFs/JSON        |
   - Chunk (claims/paras)   |
   - Ingest pipelines:      |
     - optional inference   |
       (ELSER or embeddings)|
       -> write sparse/dense|
```

**Hybrid request path**

1. **Lexical retriever** (BM25) over targets (claims_chunks + desc_chunks, optionally patents_core)
2. **Semantic retriever** (either **kNN** on `dense_vector` or **ELSER sparse** on `sparse_vector` / `semantic_text`)
3. **RRF** fuse lists → ranked results + highlights (lexical-only) ([Elastic][1])

---

## 4) Stack & Frameworks

* **Elasticsearch** 8.19+ (Cloud or self-managed).

  * Vector search (HNSW), **kNN**, **filtered kNN**, quantized vectors. ([Elastic][2])
  * **RRF** hybrid fusion (retriever API). ([Elastic][1])
  * **ELSER v2** (`sparse_vector` / `text_expansion`) or **`semantic_text`** (9.0+). ([Elastic][3])
  * **Inference API** for embeddings & **Rerank** (Cohere, HuggingFace, Voyage, Jina, etc.). ([Elastic][8])
* **Elastic Labs** examples, tutorials & notebooks for **hybrid**, **ELSER**, and **LangChain**. ([Elastic][6])
* **Client SDKs**: `@elastic/elasticsearch` (Node) or `elasticsearch` (Python).
* **Optional orchestration**: **LangChain**’s `ElasticsearchStore` & `ElasticsearchRetriever` for quick app glue. ([Langchain Python Docs][9])
* **App runtime**: FastAPI or Node/Express + simple UI (Next.js) for inspection/snippets.

---

## 5) Data Model & Indexing

**Documents**

* **`patents_core`**: one doc per patent (metadata, title, abstract, CPC/IPC, filing date).
* **`claims_chunks`**: one doc per claim; fields: `patent_id`, `claim_num`, `claim_text`, normalized text, **vector/sparse**, metadata.
* **`desc_chunks`**: one doc per paragraph (optional initially; add when you want deeper recall).

**Why chunked indexes?**

* **Claim-level recall & highlighting** (patent examiners care about claim lines).
* Avoids oversized vectors; yields tighter candidate sets & better snippet UX.

**Option A — Dense embeddings + BM25 (works on 8.x)**

* Generate embeddings externally (HuggingFace E5/BGE, or Elastic Inference).
* Map a `dense_vector` (e.g., 384/768 dims).
* **Approximate kNN**: set **`similarity`** (cosine/dot) and **HNSW `index_options`**; tune **`num_candidates`**. ([Elastic][2])

```json
PUT claims_chunks
{
  "mappings": {
    "properties": {
      "patent_id":   { "type": "keyword" },
      "claim_num":   { "type": "integer" },
      "claim_text":  { "type": "text" },                 // BM25
      "claim_vec":   { "type": "dense_vector",
                       "dims": 768,
                       "similarity": "cosine",
                       "index": true,
                       "index_options": { "type": "hnsw", "m": 32, "ef_construction": 100 } },
      "cpc":         { "type": "keyword" },
      "filed_date":  { "type": "date" }
    }
  }
}
```

**Option B — ELSER v2 sparse + BM25 (fastest to stand up on Elastic Cloud)**

* Use ELSER v2 with an **ingest inference pipeline** to write tokens+weights into a **`sparse_vector`** field (or use **`text_expansion`** at query time). ([Elastic][3])
* Mapping fragment:

```json
PUT claims_chunks
{
  "mappings": {
    "properties": {
      "claim_text": { "type": "text" },
      "elser_vec":  { "type": "sparse_vector" }    // ELSER tokens+weights
    }
  }
}
```

> **Note**: If you’re on Elastic **9.x**, you can replace both with a single `semantic_text` field that auto-embeds at index + query time via an **inference endpoint** (defaults to ELSER v2 if unspecified). This simplifies setup and still allows BM25/knn/sparse on the same field. ([Elastic][4])

---

## 6) Ingestion & Inference

* **BM25**: store raw `claim_text`, `abstract`, `title` with an English analyzer + character filters for chemical tokens; add keyword-normalized fields for exact matching of IDs.
* **Dense vectors**:

  * Generate via **Elastic Inference API** (OpenAI, Mistral, HF, VoyageAI, etc.) or client-side HF; write to `dense_vector`. ([Elastic][10])
  * For big data, consider **byte quantization** to reduce memory. ([Elastic][2])
* **ELSER v2 sparse**:

  * Create an **ingest pipeline** with an **`inference` processor** that targets `claim_text` and writes tokens to `elser_vec`. ([Elastic][11])
  * You can also **exclude tokens from `_source`** (space savings) and rely on rehydration. ([Elastic][12])

---

## 7) Query Plans (Hybrid)

### A) RRF fusion of BM25 and kNN (dense)

Use the **retriever API** with an **RRF** parent that runs a `standard` (BM25) and a `knn` retriever, then fuses top‑N results with **RRF** (default `rank_constant=60`). ([Elastic][1])

```json
GET claims_chunks/_search
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "multi_match": {
                "query": "anti-rotation composite spoke loop at rim",
                "fields": ["claim_text^2", "title", "abstract"]
              }
            }
          }
        },
        {
          "knn": {
            "field": "claim_vec",
            "query_vector": [/* 768-d query vector */],
            "k": 50,
            "num_candidates": 200
          }
        }
      ],
      "rank_window_size": 50,
      "rank_constant": 60
    }
  },
  "size": 20,
  "highlight": { "fields": { "claim_text": {} } }
}
```

> **Why RRF?** It’s well-studied, **doesn’t require weighting calibration**, and is exposed natively in Elasticsearch’s Search API. ([Elastic][1])

### B) RRF fusion of BM25 and ELSER (sparse)

If using ELSER v2, you can RRF fuse **BM25** with a **sparse_vector** (ELSER) search: ([Elastic][1])

```json
GET claims_chunks/_search
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": { "match": { "claim_text": "wheel spoke loop outer circumference" } }
          }
        },
        {
          "standard": {
            "query": {
              "sparse_vector": {
                "field": "elser_vec",
                "inference_id": ".elser_model_2",
                "query": "wheel spoke loop outer circumference"
              }
            }
          }
        }
      ]
    }
  }
}
```

> You may also use the **legacy** `text_expansion` query against a **rank_features/sparse** field if you prefer that style. ([Elastic][13])

### C) (Alt) `rank` section (tutorial style)

Elastic Labs tutorials also show hybrid by combining a normal `query` + `knn` with a `rank`/`rrf` section (older style). Use the **retriever API** for forward compatibility, but the tutorial explains the same concept. ([Elastic][6])

---

## 8) Key Tuning Knobs

* **kNN**

  * **`num_candidates`** ↑ improves recall at added latency; main knob for latency/accuracy trade-off. ([Elastic][2])
  * **HNSW index**: set `m` and `ef_construction` at index time; larger values = more accurate but heavier index build. ([Elastic][2])
  * **Memory**: vector data should **fit page cache** for best performance; consider **byte vectors (INT8)** if tight. ([Elastic][2])
  * **Filtered kNN**: supported, but filtering can decrease performance (HNSW explores more graph). Pre‑filter with metadata when possible. ([Elastic][2])
* **RRF**

  * **`rank_window_size`** controls how many top docs from each child list are considered; **`rank_constant`** controls degree of long-tail influence. Defaults are sensible. ([Elastic][1])
* **ELSER v2**

  * Sparse vectors require an **ingest pipeline** or **query-time expansion**; v2 brings sizeable **inference speedups** vs v1. ([Elastic][14])

---

## 9) Version Notes

* **Retrievers / RRF** landed 8.14 (GA 8.16). Using it requires a compatible ES version; else, fall back to `query`+`knn`+`rank` pattern. ([Elastic][15])
* **`semantic_text`** is GA in the newer stacks (9.x) and simplifies semantic indexing + querying drastically. ([Elastic][16])

---

## 10) Reranking (Phase‑2 Ready)

Add a **cross‑encoder reranker** on the **top‑K** (20–100) candidates after RRF. Use Elastic’s **Inference API** to attach a **`rerank`** provider (Cohere, VoyageAI, HuggingFace/Jina/Vertex). This keeps latency manageable while improving precision on the top of the list. ([Elastic][8])

> **How:** call Inference API to score (query, text) pairs for each candidate’s `claim_text` (and optionally abstract), then **re-sort** by the reranker score before returning results.

---

## 11) Image & Molecule Roadmap (Phase‑2/3)

1. **Figure vectors**: Extract figure captions + **image embeddings** (e.g., CLIP) per figure → store in a `figures` index with `dense_vector`. Enable **image similarity** or multi-modal hybrid (BM25 captions + kNN image vectors) via RRF. Elastic has image-similarity examples & docs for vector search on images. ([Elastic][17])
2. **Chemistry (drug patents)**: When available, push OCSR (e.g., DECIMER/OSRA) to convert structure diagrams → SMILES; index SMILES / InChI and use **substructure search** via RDKit offline, or store graph fingerprints as bit vectors for approximate lookups; fuse with text/image retrieval in UI.

---

## 12) Minimal End‑to‑End Build Recipes

### A) Dense (E5/BGE) + BM25 + RRF

1. **Create index** with `dense_vector` + `text` (mapping above). ([Elastic][2])
2. **Embed & ingest** claims: either client‑side HF or Elastic Inference API. ([Elastic][10])
3. **Search**: use **RRF retriever** to fuse BM25 `standard` + `knn`. ([Elastic][1])

### B) ELSER v2 (sparse) + BM25 + RRF (fastest path on Elastic Cloud)

1. Create `sparse_vector` field, set **ingest pipeline** with **inference** to write `elser_vec`. ([Elastic][18])
2. **Reindex** through the pipeline. ([Elastic][19])
3. **Search**: `RRF` combining BM25 and `sparse_vector` query. ([Elastic][1])

### C) Using `semantic_text` (if on 9.x)

* Map `semantic_text` for claims; Elasticsearch **auto-embeds** at index + query time via an inference endpoint (defaults to **ELSER v2** if not specified). You can still run **BM25**, **kNN**, or **sparse** on the same field and **hybridize** with RRF. ([Elastic][4])

---

## 13) Example Queries

**Hybrid with BM25 + kNN (dense)** (RRF retriever) — see Section 7A. ([Elastic][1])

**Hybrid with BM25 + ELSER (sparse)** — see Section 7B. ([Elastic][1])

**kNN plus lexical in one `_search`** (non-retriever style; useful if you’re on older minor versions): Elastic shows a `knn` block combined with a lexical `query`, then a `rank/rrf` section to fuse. ([Elastic][6])

**kNN-only quick check** (for debugging):

```json
GET claims_chunks/_search
{
  "knn": {
    "field": "claim_vec",
    "query_vector": [/* vector */],
    "k": 10,
    "num_candidates": 100
  }
}
```

— doc parameters & behavior per kNN docs. ([Elastic][2])

---

## 14) Operational Considerations

* **Sizing**: Vector data should **fit page cache**; shard for parallelism; use **byte vectors** where feasible; watch **`num_candidates`**. ([Elastic][2])
* **Filters**: Prefer **post‑retrieval filters** on BM25 results; use **filtered kNN** sparingly or with careful benchmarking (it’s supported but can be slower due to graph exploration). ([Elastic][2])
* **Explainability**: Provide **highlights** from the **BM25 leg**; vector legs don’t produce highlight snippets directly. ([Elastic][1])
* **Version compatibility**: Retrievers require **≥ 8.14**, GA in **8.16**. On older stacks, use the tutorial’s `rank/rrf` pattern. ([Elastic][15])
* **Pipelines**: Keep **ELSER** or **embedding** inference in a dedicated **ingest pipeline** so you can reindex/upgrade models later (ELSER v2 upgrade path is documented). ([Elastic][3])
* **Dev acceleration**: Elastic’s **Elasticsearch Labs** repo has ready notebooks for **02-hybrid-search**, **03-ELSER**, and **semantic_text** to bootstrap code quickly. ([GitHub][20])

---

## 15) Tooling, Frameworks & References

* **Elastic Labs GitHub** (`elasticsearch-labs`): notebooks for hybrid, ELSER, LangChain. ([GitHub][21])
* **Hybrid tutorial (RRF)** (Search Labs): example of **full-text + kNN** plus `rrf` fusion. ([Elastic][6])
* **ESRE intro** (Elastic blog): vector DB + ML models for semantic search & LLM apps. ([Elastic][5])
* **ELSER docs**: semantic search with sparse vectors; `text_expansion` / `sparse_vector`. ([Elastic][3])
* **Dense vectors & kNN**: mapping, HNSW, num_candidates, quantization. ([Elastic][2])
* **RRF API docs** (retrievers): formula, request shape, parameters. ([Elastic][1])
* **`semantic_text`** docs & blog: simplify semantic search (auto-embed). ([Elastic][4])
* **Aman.ai hybrid overview**: motivation, fusion methods, evaluation. ([Aman AI][7])

---

## 16) Suggested Phased Build

**Phase 1 (fastest & impressive)**

* Choose **ELSER v2 + BM25 + RRF** **or** **Dense (E5/BGE) + BM25 + RRF** depending on infra:

  * If on **Elastic Cloud** and want zero outside dependencies → **ELSER**.
  * If you already have embedding infra or want model control → **dense**.
* Index **claims first** (`claims_chunks`) for best UX; add `patents_core` and `desc_chunks` next.
* Ship a minimal UI with **explanatory highlights** (from BM25 leg) + “why we matched” signals.

**Phase 2 (impactful next step)**

* Add **cross‑encoder reranker** via Elastic **Inference API** (`rerank` providers: Cohere/Voyage/HF/Jina/Vertex). This significantly sharpens the **Top‑K** after hybrid retrieval. ([Elastic][8])

**Phase 3 (multi‑modal enrichment)**

* Add **image vectors** per figure + **caption BM25**; enable **visual + text hybrid** and lay groundwork for chemistry OCSR.

---

## 17) Appendix — Example: `semantic_text` (9.x)

```json
PUT claims_chunks
{
  "mappings": {
    "properties": {
      "claim_sem": {
        "type": "semantic_text",
        "inference_id": ".elser-2-elasticsearch",   // default if omitted
        "chunking_settings": { "strategy": "none" } // toggle if you pre-chunk
      },
      "claim_text": { "type": "text" }
    }
  }
}
```

* Then query with **`semantic`** **and** a lexical **`match`** in one request (and/or via an RRF retriever), fully supported by the docs. ([Elastic][22])

---

## 18) Risks & Mitigations

* **Vector memory pressure** → **byte quantization**, fewer dims, shard sizing. ([Elastic][2])
* **Filtered kNN latency** → re-check filter selectivity; consider pre‑filter via metadata or two‑stage (BM25 filter then kNN on smaller candidate set). ([Elastic][2])
* **Highlights on semantic** → originate snippets from lexical leg and show a “semantic match” badge on fused results. ([Elastic][1])
* **Version mismatches** → pin to ≥ **8.16** for **retrievers**; else fallback to `rank/rrf` tutorial shape. ([Elastic][15])

---

## 19) What you can say in the interview (short talking points)

* “We use **RRF** to combine BM25 with semantic candidates so we don’t miss exact term matches **or** concept matches, and it’s **first-class** in Elasticsearch.” ([Elastic][1])
* “We started with **claim-level indexing** for better unit-level relevance and highlighting.”
* “We can switch semantic legs between **ELSER v2 (sparse)** and **dense kNN** with the **same hybrid/RRF skeleton**.” ([Elastic][3])
* “Phase‑2 adds a **cross‑encoder reranker** via Elastic’s **Inference API** for Top‑K precision.” ([Elastic][8])
* “For images (e.g., molecule sketches), we’ll add **image embeddings** and **hybridize** captions + image vectors, as per Elastic’s image-similarity guidance.” ([Elastic][17])

---

### Footnotes / Primary References

* Elastic Labs **Hybrid Search** tutorial and notebooks. ([Elastic][6])
* **RRF retriever** API & formula. ([Elastic][1])
* **kNN** vector search mapping, tuning, quantization, filtering. ([Elastic][2])
* **ELSER v2** (sparse) semantics and `sparse_vector`/`text_expansion`. ([Elastic][3])
* **`semantic_text`** GA simplifications. ([Elastic][16])
* **Inference API** **rerank** providers for cross‑encoder stage. ([Elastic][8])
* Hybrid search background rationale. ([Aman AI][7])

---

If you want, I can also turn this into a **starter repo** with:

* `docker-compose.yml` for Elasticsearch,
* `scripts/create-indexes.sh` with the mappings above,
* a **Python client** notebook for **ELSER v2**, and
* a **FastAPI endpoint** that runs the **RRF retriever** (BM25 + semantic) and returns highlighted hits.

[1]: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion "Reciprocal rank fusion | Reference"
[2]: https://elastic.aiops.work/guide/en/elasticsearch/reference/current/knn-search.html "kNN search in Elasticsearch | Elastic Docs"
[3]: https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-elser?utm_source=chatgpt.com "ELSER - Elastic Docs"
[4]: https://www.elastic.co/guide/en/elasticsearch/reference/8.19/semantic-text.html?utm_source=chatgpt.com "Semantic text field type | Elasticsearch Guide [8.19] | Elastic"
[5]: https://www.elastic.co/search-labs/blog/introducing-elasticsearch-relevance-engine-esre?utm_source=chatgpt.com "Introducing Elasticsearch Relevance Engine (ESRE) — Advanced search for ..."
[6]: https://www.elastic.co/search-labs/tutorials/search-tutorial/vector-search/hybrid-search "Hybrid Search: Combined Full-Text and kNN Results - Elasticsearch Labs"
[7]: https://aman.ai/recsys/search/?utm_source=chatgpt.com "Aman's AI Journal • Recommendation Systems • Search"
[8]: https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-inference-put?utm_source=chatgpt.com "Create an inference endpoint | Elasticsearch API documentation"
[9]: https://python.langchain.com/docs/integrations/vectorstores/elasticsearch/?utm_source=chatgpt.com "Elasticsearch | ️ LangChain"
[10]: https://www.elastic.co/docs/api/doc/elasticsearch/v8/operation/operation-inference-text-embedding?utm_source=chatgpt.com "Perform text embedding inference on the service | Elasticsearch API ..."
[11]: https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-inference?utm_source=chatgpt.com "Add NLP inference to ingest pipelines - Elastic Docs"
[12]: https://elastic.aiops.work/guide/en/elasticsearch/reference/current/semantic-search-elser.html?utm_source=chatgpt.com "Semantic search with ELSER | Elastic Docs"
[13]: https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-text-expansion-query?utm_source=chatgpt.com "Text expansion query | Reference - Elastic"
[14]: https://www.elastic.co/search-labs/blog/introducing-elser-v2-part-1?utm_source=chatgpt.com "ElSER v2: Improved information retrieval & inference ... - Elastic"
[15]: https://elastic.aiops.work/guide/en/elasticsearch/reference/8.19/retrievers-overview.html?utm_source=chatgpt.com "Retrievers | Elasticsearch Guide [8.19] | Elastic"
[16]: https://www.elastic.co/search-labs/blog/elasticsearch-semantic-text-ga?utm_source=chatgpt.com "Semantic text in Elasticsearch: Simpler, better, leaner, stronger"
[17]: https://www.elastic.co/search-labs/blog/implement-image-similarity-search-elastic?utm_source=chatgpt.com "How to implement image similarity search in Elasticsearch"
[18]: https://www.elastic.co/guide/en/elasticsearch/reference/8.19/sparse-vector.html?utm_source=chatgpt.com "Sparse vector field type | Elasticsearch Guide [8.19] | Elastic"
[19]: https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-text-emb-vector-search-example?utm_source=chatgpt.com "Text embedding and semantic search - Elastic Docs"
[20]: https://github.com/elastic/elasticsearch-labs/blob/main/notebooks/search/02-hybrid-search.ipynb?utm_source=chatgpt.com "elasticsearch-labs/notebooks/search/02-hybrid-search.ipynb at main ..."
[21]: https://github.com/elastic/elasticsearch-labs?utm_source=chatgpt.com "GitHub - elastic/elasticsearch-labs: Notebooks & Example Apps for ..."
[22]: https://elastic.aiops.work/guide/en/elasticsearch/reference/current/semantic-search-semantic-text.html?utm_source=chatgpt.com "Semantic search with semantic_text | Elastic Docs"
