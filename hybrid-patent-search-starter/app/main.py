from fastapi import FastAPI
from pydantic import BaseModel
import os
from .es_client import get_es
from .search_builders import build_rrf_hybrid_body

app = FastAPI(title="Hybrid Patent Search")

class SearchRequest(BaseModel):
    q: str
    top_k: int = 10
    rerank: bool = False

@app.post("/search")
def search(req: SearchRequest):
    es = get_es()
    index = os.getenv("ES_INDEX_NAME", "patents")
    use_semantic = os.getenv("ES_USE_SEMANTIC", "elser").lower()
    elser_model_id = os.getenv("ELSER_MODEL_ID", ".elser_model_2")

    body = build_rrf_hybrid_body(req.q, req.top_k, use_semantic, elser_model_id)
    res = es.search(index=index, body=body)

    hits = []
    for h in res.get("hits", {}).get("hits", []):
        source = h.get("_source", {})
        hit = {
            "id": h.get("_id"),
            "score": h.get("_score"),
            "doc_number": source.get("doc_number"),
            "section_type": source.get("section_type"),
            "title": source.get("title"),
            "abstract": source.get("abstract"),
            "claims_text": source.get("claims_text"),
            "highlights": h.get("highlight", {})
        }
        hits.append(hit)

    return {"total": res.get("hits", {}).get("total", {}), "hits": hits}
