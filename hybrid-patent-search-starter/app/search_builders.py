"""
Search query builders for hybrid patent search with RRF fusion
"""

import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

from config import config, get_bm25_fields, get_dense_vector_field, get_sparse_vector_fields


def build_bm25_query(query: str, index: str) -> Dict[str, Any]:
    """
    Build BM25 query for specific index
    
    Args:
        query: Search query text
        index: Index name to search
    
    Returns:
        Elasticsearch query dict
    """
    fields = get_bm25_fields(index)
    
    if not fields:
        # Fallback if no fields configured
        fields = ["_all"]
    
    return {
        "multi_match": {
            "query": query,
            "fields": fields,
            "type": "best_fields",
            "tie_breaker": 0.3,
            "minimum_should_match": "30%"
        }
    }


def build_dense_knn_query(
    query_vector: List[float],
    index: str,
    k: int = 50,
    num_candidates: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Build kNN query for dense vector search
    
    Args:
        query_vector: Query embedding vector
        index: Index name to search
        k: Number of nearest neighbors
        num_candidates: Number of candidates for HNSW
    
    Returns:
        kNN query dict or None if no vector field
    """
    vector_field = get_dense_vector_field(index)
    
    if not vector_field:
        return None
    
    if num_candidates is None:
        num_candidates = config.search.num_candidates_knn
    
    return {
        "field": vector_field,
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates
    }


def build_sparse_vector_query(
    query_text: str,
    index: str,
    model_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Build ELSER sparse vector query
    
    Args:
        query_text: Query text for ELSER
        index: Index name to search
        model_id: ELSER model ID (uses config default if not provided)
    
    Returns:
        Sparse vector query dict or None if no sparse fields
    """
    sparse_fields = get_sparse_vector_fields(index)
    
    if not sparse_fields:
        return None
    
    if model_id is None:
        model_id = config.models.elser_model_id
    
    # For multiple sparse fields, we use a dis_max query
    if len(sparse_fields) > 1:
        return {
            "dis_max": {
                "queries": [
                    {
                        "sparse_vector": {
                            "field": field,
                            "inference_id": model_id,
                            "query": query_text
                        }
                    }
                    for field in sparse_fields
                ]
            }
        }
    else:
        # Single sparse field
        return {
            "sparse_vector": {
                "field": sparse_fields[0],
                "inference_id": model_id,
                "query": query_text
            }
        }


def build_rrf_retriever(
    query_text: str,
    query_vector: Optional[List[float]] = None,
    indices: Optional[List[str]] = None,
    top_k: int = 20,
    rank_window_size: Optional[int] = None,
    rank_constant: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build RRF retriever combining BM25, dense kNN, and ELSER
    
    Args:
        query_text: Text query
        query_vector: Optional dense embedding vector
        indices: Indices to search (uses default if not provided)
        top_k: Number of results to return
        rank_window_size: RRF window size
        rank_constant: RRF rank constant
    
    Returns:
        RRF retriever configuration
    """
    if indices is None:
        indices = config.get_search_indices("all")
    
    if rank_window_size is None:
        rank_window_size = config.search.rrf_window_size
    
    if rank_constant is None:
        rank_constant = config.search.rrf_rank_constant
    
    # Ensure window size is at least as large as top_k
    rank_window_size = max(rank_window_size, top_k)
    
    retrievers = []
    
    # Build retrievers for each index
    for index in indices:
        # BM25 retriever
        bm25_query = build_bm25_query(query_text, index)
        retrievers.append({
            "standard": {
                "query": bm25_query
            }
        })
        
        # Dense kNN retriever (if vector provided)
        if query_vector:
            knn_config = build_dense_knn_query(
                query_vector,
                index,
                k=rank_window_size
            )
            if knn_config:
                retrievers.append({
                    "knn": knn_config
                })
        
        # ELSER sparse retriever
        sparse_query = build_sparse_vector_query(query_text, index)
        if sparse_query:
            retrievers.append({
                "standard": {
                    "query": sparse_query
                }
            })
    
    return {
        "retriever": {
            "rrf": {
                "retrievers": retrievers,
                "rank_window_size": rank_window_size,
                "rank_constant": rank_constant
            }
        }
    }


def build_multi_index_search(
    query_text: str,
    query_vector: Optional[List[float]] = None,
    search_scope: str = "all",
    top_k: int = 20,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Build multi-index search request for _msearch API
    
    Args:
        query_text: Search query
        query_vector: Optional query embedding
        search_scope: Scope of search ("all", "patents", "claims", "descriptions")
        top_k: Number of results
        filters: Optional filters to apply
    
    Returns:
        List of index/body pairs for _msearch
    """
    indices = config.get_search_indices(search_scope)
    search_bodies = []
    
    for index in indices:
        # Build RRF retriever for this index
        retriever = build_rrf_retriever(
            query_text=query_text,
            query_vector=query_vector,
            indices=[index],
            top_k=top_k
        )
        
        # Add highlighting
        body = {
            **retriever,
            "size": top_k,
            "highlight": config.search.highlight_config,
            "_source": {
                "excludes": ["*_vec", "*_elser"]  # Exclude vector fields from response
            }
        }
        
        # Add filters if provided
        if filters:
            # Wrap existing query with bool filter
            if "retriever" in body:
                # For retrievers, we need to add filter at the top level
                body["filter"] = filters
            else:
                # For regular queries
                body["query"] = {
                    "bool": {
                        "must": body.get("query", {"match_all": {}}),
                        "filter": filters
                    }
                }
        
        search_bodies.extend([
            {"index": index},
            body
        ])
    
    return search_bodies


def build_aggregation_query(
    index: str,
    agg_field: str,
    size: int = 10
) -> Dict[str, Any]:
    """
    Build aggregation query for faceting
    
    Args:
        index: Index to aggregate on
        agg_field: Field to aggregate
        size: Number of buckets
    
    Returns:
        Aggregation query
    """
    return {
        "size": 0,
        "aggs": {
            f"{agg_field}_facet": {
                "terms": {
                    "field": agg_field,
                    "size": size,
                    "order": {"_count": "desc"}
                }
            }
        }
    }


def build_filter_query(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build filter clauses from filter dictionary
    
    Args:
        filters: Dictionary of field:value filters
    
    Returns:
        List of filter clauses
    """
    filter_clauses = []
    
    for field, value in filters.items():
        if isinstance(value, list):
            # Multiple values - use terms query
            filter_clauses.append({
                "terms": {field: value}
            })
        elif isinstance(value, dict):
            # Range query
            filter_clauses.append({
                "range": {field: value}
            })
        else:
            # Single value
            filter_clauses.append({
                "term": {field: value}
            })
    
    return filter_clauses