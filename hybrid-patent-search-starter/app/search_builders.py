import os

def build_bm25_query(q: str):
    return {
        "multi_match": {
            "query": q,
            "fields": ["claims_text^3", "abstract^2", "title"]
        }
    }

def build_sparse_vector_query(field: str, model_id: str, q: str):
    # ELSER text_expansion style
    return {
        "sparse_vector": {
            "field": field,
            "inference_id": model_id,
            "query": q
        }
    }

def build_rrf_hybrid_body(q: str, top_k: int, use_semantic: str, elser_model_id: str):
    """
    Builds a retriever.rrf request body that fuses BM25 + semantic in one _search.
    For dense path, you must add a query_vector for kNN (not implemented in this starter).
    """
    bm25_retriever = {"standard": {"query": build_bm25_query(q)}}

    if use_semantic == "elser":
        semantic_retriever = {
            "standard": {
                "query": build_sparse_vector_query("ml.tokens", elser_model_id, q)
            }
        }
    else:
        # Placeholder for dense: add query embeddings and use "knn" retriever
        semantic_retriever = bm25_retriever  # fallback to BM25 only

    body = {
        "retriever": {
            "rrf": {
                "retrievers": [bm25_retriever, semantic_retriever],
                "rank_window_size": max(50, top_k),
                "rank_constant": 60
            }
        },
        "size": top_k,
        "highlight": {
            "fields": {
                "claims_text": {},
                "abstract": {}
            }
        }
    }
    return body
