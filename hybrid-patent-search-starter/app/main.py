"""
FastAPI application for hybrid patent search
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.es_client import get_es
from app.search_builders import build_multi_index_search, build_filter_query
from config import config

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Patent Search API",
    description="Search patents using BM25 + Dense Vectors + ELSER v2 with RRF fusion",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize sentence transformer model (lazy loading)
_embedder = None

def get_embedder():
    """Get or initialize the sentence transformer model"""
    global _embedder
    if _embedder is None:
        model_info = config.models.embedding_model_info
        _embedder = SentenceTransformer(model_info["model_name"])
    return _embedder


class SearchRequest(BaseModel):
    """Search request model"""
    q: str = Field(..., description="Search query text")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of results to return")
    search_scope: str = Field(
        default="all",
        description="Scope of search: all, patents, claims, descriptions"
    )
    use_embeddings: bool = Field(
        default=True,
        description="Whether to use dense embeddings for semantic search"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters (e.g., {'cpc_codes': ['A01B'], 'filing_date': {'gte': '2020-01-01'}})"
    )
    aggregate_by_patent: bool = Field(
        default=True,
        description="Whether to aggregate results by patent ID"
    )
    rerank: bool = Field(
        default=False,
        description="Whether to apply reranking (Phase 2 feature)"
    )


class PatentHit(BaseModel):
    """Individual search hit"""
    id: str
    index: str
    score: float
    patent_id: str
    doc_type: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    claim_text: Optional[str] = None
    claim_num: Optional[int] = None
    desc_text: Optional[str] = None
    highlights: Dict[str, List[str]]
    metadata: Dict[str, Any]


class AggregatedPatent(BaseModel):
    """Patent with aggregated hits from multiple indices"""
    patent_id: str
    total_score: float
    best_score: float
    hit_count: int
    title: Optional[str] = None
    abstract: Optional[str] = None
    hits: List[PatentHit]


class SearchResponse(BaseModel):
    """Search response model"""
    total_hits: int
    aggregated_results: Optional[List[AggregatedPatent]] = None
    raw_hits: Optional[List[PatentHit]] = None
    search_time_ms: int
    query_info: Dict[str, Any]


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Hybrid Patent Search API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    es = get_es()
    
    try:
        # Check Elasticsearch
        es_info = es.info()
        es_health = es.cluster.health()
        
        # Check indices
        indices = config.get_index_list()
        index_info = {}
        
        for index in indices:
            try:
                stats = es.indices.stats(index=index)
                doc_count = stats["indices"][index]["primaries"]["docs"]["count"]
                index_info[index] = {"status": "green", "docs": doc_count}
            except:
                index_info[index] = {"status": "missing", "docs": 0}
        
        # Check ML models
        ml_status = {}
        try:
            elser_stats = es.ml.get_trained_models_stats(model_id=config.models.elser_model_id)
            if elser_stats["trained_model_stats"]:
                deployment = elser_stats["trained_model_stats"][0].get("deployment_stats", {})
                ml_status["elser"] = deployment.get("state", "unknown")
            else:
                ml_status["elser"] = "not_deployed"
        except:
            ml_status["elser"] = "error"
        
        # Check embedder
        try:
            embedder = get_embedder()
            ml_status["dense_embedder"] = "ready"
        except:
            ml_status["dense_embedder"] = "error"
        
        return {
            "status": "healthy" if es_health["status"] in ["green", "yellow"] else "unhealthy",
            "elasticsearch": {
                "version": es_info["version"]["number"],
                "cluster_status": es_health["status"]
            },
            "indices": index_info,
            "ml_models": ml_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Main search endpoint - performs hybrid search across patent indices
    """
    import time
    start_time = time.time()
    
    es = get_es()
    
    # Generate query embedding if requested
    query_vector = None
    if request.use_embeddings:
        try:
            embedder = get_embedder()
            query_vector = embedder.encode(request.q, show_progress_bar=False).tolist()
        except Exception as e:
            print(f"Warning: Failed to generate embeddings: {e}")
            # Continue without embeddings
    
    # Build filter query if provided
    es_filters = None
    if request.filters:
        filter_clauses = build_filter_query(request.filters)
        if filter_clauses:
            es_filters = {"bool": {"filter": filter_clauses}}
    
    # Build multi-index search request
    search_bodies = build_multi_index_search(
        query_text=request.q,
        query_vector=query_vector,
        search_scope=request.search_scope,
        top_k=request.top_k if not request.aggregate_by_patent else request.top_k * 3,
        filters=es_filters
    )
    
    # Execute multi-index search
    try:
        responses = es.msearch(body=search_bodies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    # Process responses
    all_hits = []
    total_hits = 0
    
    for i in range(0, len(search_bodies), 2):
        index = search_bodies[i]["index"]
        response = responses["responses"][i // 2]
        
        if "error" in response:
            print(f"Error searching {index}: {response['error']}")
            continue
        
        hits = response.get("hits", {}).get("hits", [])
        total_hits += response.get("hits", {}).get("total", {}).get("value", 0)
        
        # Process each hit
        for hit in hits:
            source = hit.get("_source", {})
            
            # Determine document type
            if index == config.indices.patents_core:
                doc_type = "patent"
            elif index == config.indices.claims_chunks:
                doc_type = "claim"
            else:
                doc_type = "description"
            
            # Extract patent ID
            patent_id = source.get("patent_id") or source.get("doc_number", "")
            
            # Build hit object
            patent_hit = PatentHit(
                id=hit["_id"],
                index=index,
                score=hit.get("_score", 0.0),
                patent_id=patent_id,
                doc_type=doc_type,
                title=source.get("title"),
                abstract=source.get("abstract"),
                claim_text=source.get("claim_text"),
                claim_num=source.get("claim_num"),
                desc_text=source.get("desc_text"),
                highlights=hit.get("highlight", {}),
                metadata={
                    k: v for k, v in source.items()
                    if k not in ["title", "abstract", "claim_text", "desc_text"]
                    and not k.endswith("_vec") and not k.endswith("_elser")
                }
            )
            
            all_hits.append(patent_hit)
    
    # Apply reranking if requested (placeholder for Phase 2)
    if request.rerank:
        # TODO: Implement cross-encoder reranking
        pass
    
    # Sort hits by score
    all_hits.sort(key=lambda x: x.score, reverse=True)
    
    # Aggregate results by patent if requested
    if request.aggregate_by_patent:
        patent_groups = defaultdict(list)
        
        for hit in all_hits:
            patent_groups[hit.patent_id].append(hit)
        
        # Create aggregated patents
        aggregated_patents = []
        
        for patent_id, hits in patent_groups.items():
            # Get best scoring hit's metadata
            best_hit = max(hits, key=lambda x: x.score)
            
            aggregated = AggregatedPatent(
                patent_id=patent_id,
                total_score=sum(h.score for h in hits),
                best_score=best_hit.score,
                hit_count=len(hits),
                title=next((h.title for h in hits if h.title), None),
                abstract=next((h.abstract for h in hits if h.abstract), None),
                hits=sorted(hits, key=lambda x: x.score, reverse=True)
            )
            
            aggregated_patents.append(aggregated)
        
        # Sort by total score
        aggregated_patents.sort(key=lambda x: x.total_score, reverse=True)
        
        # Limit to top_k
        aggregated_patents = aggregated_patents[:request.top_k]
        
        response_data = {
            "total_hits": len(patent_groups),
            "aggregated_results": aggregated_patents,
            "raw_hits": None
        }
    else:
        # Return raw hits
        response_data = {
            "total_hits": total_hits,
            "aggregated_results": None,
            "raw_hits": all_hits[:request.top_k]
        }
    
    # Add query info and timing
    response_data.update({
        "search_time_ms": int((time.time() - start_time) * 1000),
        "query_info": {
            "query": request.q,
            "search_scope": request.search_scope,
            "used_embeddings": query_vector is not None,
            "used_elser": True,  # Always true when available
            "indices_searched": list(set(b["index"] for b in search_bodies if "index" in b))
        }
    })
    
    return SearchResponse(**response_data)


@app.get("/stats")
def get_stats():
    """Get index statistics"""
    es = get_es()
    indices = config.get_index_list()
    
    stats = {}
    
    for index in indices:
        try:
            index_stats = es.indices.stats(index=index)
            count = es.count(index=index)
            
            stats[index] = {
                "doc_count": count["count"],
                "size_mb": index_stats["indices"][index]["primaries"]["store"]["size_in_bytes"] / (1024 * 1024),
                "status": "available"
            }
        except:
            stats[index] = {"status": "not_available"}
    
    return stats


@app.post("/embed")
async def generate_embedding(text: str):
    """Generate embedding for text (useful for testing)"""
    try:
        embedder = get_embedder()
        embedding = embedder.encode(text, show_progress_bar=False)
        
        return {
            "text": text,
            "embedding": embedding.tolist(),
            "dims": len(embedding),
            "model": config.models.dense_embedding_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload
    )