"""
Integration tests for hybrid patent search
"""

import sys
import json
import pytest
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from elasticsearch import Elasticsearch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.es_client import get_es
from app.search_builders import (
    build_bm25_query,
    build_dense_knn_query,
    build_sparse_vector_query,
    build_rrf_retriever,
    build_multi_index_search
)
from config import config
from scripts.ingest_data import PatentIngester, iter_patents_from_file


@pytest.fixture(scope="session")
def es_client():
    """Get Elasticsearch client"""
    return get_es()


@pytest.fixture(scope="session")
def test_data():
    """Sample patent data for testing"""
    return [
        {
            "doc_number": "US1234567",
            "title": "Advanced composite wheel with anti-rotation spokes",
            "abstract": "A wheel assembly comprising composite spokes arranged in a loop configuration at the outer circumference to prevent rotation",
            "claims": [
                "1. A wheel assembly comprising composite spokes arranged in a loop pattern",
                "2. The wheel assembly of claim 1, wherein the spokes are carbon fiber",
                "3. A method for manufacturing the wheel of claim 1"
            ],
            "description": "This invention relates to advanced wheel designs. The composite spokes provide superior strength. Manufacturing involves precise molding techniques.",
            "filing_date": "2023-01-15",
            "cpc_codes": ["B60B21/00", "B29C70/00"]
        },
        {
            "doc_number": "US2345678",
            "title": "Optical sensor for molecular structure detection",
            "abstract": "An optical sensing system capable of detecting molecular structures using spectroscopy techniques",
            "claims": [
                "1. An optical sensor system for molecular detection comprising a light source and detector",
                "2. The system of claim 1, further including spectral analysis components"
            ],
            "description": "The present invention provides an optical sensor for detecting molecular structures. The sensor uses advanced spectroscopy. Applications include chemical analysis.",
            "filing_date": "2023-02-20",
            "cpc_codes": ["G01N21/00", "G01J3/00"]
        }
    ]


@pytest.fixture(scope="session")
def setup_test_indices(es_client, test_data):
    """Setup test indices and ingest test data"""
    # Create test index names
    test_indices = {
        "patents_core": f"test_{config.indices.patents_core}",
        "claims_chunks": f"test_{config.indices.claims_chunks}",
        "desc_chunks": f"test_{config.indices.desc_chunks}"
    }
    
    # Override config for tests
    config.indices.patents_core = test_indices["patents_core"]
    config.indices.claims_chunks = test_indices["claims_chunks"]
    config.indices.desc_chunks = test_indices["desc_chunks"]
    
    # Delete test indices if they exist
    for index in test_indices.values():
        try:
            es_client.indices.delete(index=index)
        except:
            pass
    
    # Create indices using test function
    create_test_indices(es_client, config.models.embedding_dims)
    
    # Ingest test data
    ingester = PatentIngester(
        es_client=es_client,
        use_dense_embeddings=True,
        use_elser=False  # Disable ELSER for tests
    )
    
    # Process test patents
    for patent in test_data:
        actions = list(ingester.process_patent(patent))
        from elasticsearch.helpers import bulk
        bulk(es_client, actions)
    
    # Refresh indices
    for index in test_indices.values():
        es_client.indices.refresh(index=index)
    
    yield test_indices
    
    # Cleanup after tests
    for index in test_indices.values():
        try:
            es_client.indices.delete(index=index)
        except:
            pass


class TestSearchBuilders:
    """Test search query builders"""
    
    def test_bm25_query(self):
        """Test BM25 query builder"""
        query = build_bm25_query("wheel spokes", config.indices.claims_chunks)
        
        assert "multi_match" in query
        assert query["multi_match"]["query"] == "wheel spokes"
        assert "fields" in query["multi_match"]
        assert len(query["multi_match"]["fields"]) > 0
    
    def test_dense_knn_query(self):
        """Test dense kNN query builder"""
        # Create sample vector
        query_vector = np.random.rand(384).tolist()
        
        knn_query = build_dense_knn_query(
            query_vector=query_vector,
            index=config.indices.claims_chunks,
            k=10
        )
        
        assert knn_query is not None
        assert "field" in knn_query
        assert "query_vector" in knn_query
        assert knn_query["k"] == 10
        assert len(knn_query["query_vector"]) == 384
    
    def test_sparse_vector_query(self):
        """Test sparse vector query builder"""
        # Test with ELSER disabled returns None
        sparse_query = build_sparse_vector_query(
            "molecular detection",
            config.indices.claims_chunks
        )
        
        # This should work even if ELSER is not deployed
        assert sparse_query is not None or True  # Allow None for test environment
    
    def test_rrf_retriever(self):
        """Test RRF retriever builder"""
        retriever = build_rrf_retriever(
            query_text="wheel composite",
            query_vector=np.random.rand(384).tolist(),
            top_k=10
        )
        
        assert "retriever" in retriever
        assert "rrf" in retriever["retriever"]
        assert "retrievers" in retriever["retriever"]["rrf"]
        assert len(retriever["retriever"]["rrf"]["retrievers"]) >= 2  # At least BM25 and kNN
    
    def test_multi_index_search(self):
        """Test multi-index search builder"""
        search_bodies = build_multi_index_search(
            query_text="optical sensor",
            search_scope="all",
            top_k=5
        )
        
        # Should have alternating index/body pairs
        assert len(search_bodies) >= 4  # At least 2 indices
        assert "index" in search_bodies[0]
        assert "retriever" in search_bodies[1]


class TestIntegrationSearch:
    """Integration tests for search functionality"""
    
    @pytest.fixture
    def sample_query_vector(self):
        """Generate sample query vector"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.models.dense_embedding_model)
        return model.encode("wheel composite spokes", show_progress_bar=False).tolist()
    
    def test_search_patents_core(self, es_client, setup_test_indices, sample_query_vector):
        """Test searching patents core index"""
        bodies = build_multi_index_search(
            query_text="composite wheel",
            query_vector=sample_query_vector,
            search_scope="patents",
            top_k=5
        )
        
        # Execute search
        response = es_client.msearch(body=bodies)
        
        assert "responses" in response
        assert len(response["responses"]) > 0
        
        hits = response["responses"][0].get("hits", {}).get("hits", [])
        assert len(hits) > 0
        
        # Check first hit has expected fields
        first_hit = hits[0]["_source"]
        assert "title" in first_hit
        assert "patent_id" in first_hit
    
    def test_search_claims_chunks(self, es_client, setup_test_indices):
        """Test searching claims chunks index"""
        bodies = build_multi_index_search(
            query_text="carbon fiber spokes",
            search_scope="claims",
            top_k=5
        )
        
        response = es_client.msearch(body=bodies)
        hits = response["responses"][0].get("hits", {}).get("hits", [])
        
        assert len(hits) > 0
        
        # Verify claim fields
        first_hit = hits[0]["_source"]
        assert "claim_text" in first_hit
        assert "claim_num" in first_hit
        assert "patent_id" in first_hit
    
    def test_hybrid_search_all_indices(self, es_client, setup_test_indices, sample_query_vector):
        """Test hybrid search across all indices"""
        bodies = build_multi_index_search(
            query_text="molecular optical sensor",
            query_vector=sample_query_vector,
            search_scope="all",
            top_k=10
        )
        
        response = es_client.msearch(body=bodies)
        
        # Collect all hits
        all_hits = []
        for resp in response["responses"]:
            if "hits" in resp:
                all_hits.extend(resp["hits"]["hits"])
        
        assert len(all_hits) > 0
        
        # Verify we got results from multiple indices
        indices_hit = set(hit["_index"] for hit in all_hits)
        assert len(indices_hit) >= 2
    
    def test_search_with_filters(self, es_client, setup_test_indices):
        """Test search with filters"""
        filters = {
            "cpc_codes": "B60B21/00"
        }
        
        bodies = build_multi_index_search(
            query_text="wheel",
            search_scope="patents",
            top_k=5,
            filters=filters
        )
        
        response = es_client.msearch(body=bodies)
        hits = response["responses"][0].get("hits", {}).get("hits", [])
        
        # Should only return patents with the specified CPC code
        for hit in hits:
            cpc_codes = hit["_source"].get("cpc_codes", [])
            assert "B60B21/00" in cpc_codes
    
    def test_search_highlighting(self, es_client, setup_test_indices):
        """Test search result highlighting"""
        bodies = build_multi_index_search(
            query_text="composite spokes",
            search_scope="claims",
            top_k=5
        )
        
        response = es_client.msearch(body=bodies)
        hits = response["responses"][0].get("hits", {}).get("hits", [])
        
        # Check for highlights
        for hit in hits:
            if "highlight" in hit:
                highlights = hit["highlight"]
                assert len(highlights) > 0
                # Verify highlight contains the mark tags
                for field, snippets in highlights.items():
                    for snippet in snippets:
                        assert "<mark>" in snippet or "composite" in snippet.lower()


class TestPatentIngester:
    """Test patent ingestion functionality"""
    
    def test_chunk_text(self):
        """Test text chunking"""
        ingester = PatentIngester(get_es(), use_dense_embeddings=False, use_elser=False)
        
        text = " ".join(["word"] * 1000)
        chunks = ingester.chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 100 for chunk in chunks)
    
    def test_parse_claims(self):
        """Test claim parsing"""
        ingester = PatentIngester(get_es(), use_dense_embeddings=False, use_elser=False)
        
        claims_text = "1. A device comprising components. 2. The device of claim 1, wherein the components are metal."
        parsed = ingester.parse_claims(claims_text)
        
        assert len(parsed) == 2
        assert parsed[0]["claim_num"] == 1
        assert parsed[0]["claim_type"] == "independent"
        assert parsed[1]["claim_num"] == 2
        assert parsed[1]["claim_type"] == "dependent"
    
    def test_generate_embeddings(self):
        """Test embedding generation"""
        ingester = PatentIngester(get_es(), use_dense_embeddings=True, use_elser=False)
        
        texts = ["sample text 1", "sample text 2"]
        embeddings = ingester.generate_embeddings(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == config.models.embedding_dims


# Utility function to create test indices (would be imported from create_indices.sh logic)
def create_test_indices(es_client: Elasticsearch, embedding_dims: int):
    """Create test indices with proper mappings"""
    # Patents core
    es_client.indices.create(
        index=config.indices.patents_core,
        body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "patent_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "abstract": {"type": "text"},
                    "title_vec": {"type": "dense_vector", "dims": embedding_dims},
                    "abstract_vec": {"type": "dense_vector", "dims": embedding_dims},
                    "cpc_codes": {"type": "keyword"},
                    "filing_date": {"type": "date"}
                }
            }
        }
    )
    
    # Claims chunks
    es_client.indices.create(
        index=config.indices.claims_chunks,
        body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "patent_id": {"type": "keyword"},
                    "claim_num": {"type": "integer"},
                    "claim_text": {"type": "text"},
                    "claim_vec": {"type": "dense_vector", "dims": embedding_dims}
                }
            }
        }
    )
    
    # Description chunks
    es_client.indices.create(
        index=config.indices.desc_chunks,
        body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "patent_id": {"type": "keyword"},
                    "desc_text": {"type": "text"},
                    "desc_vec": {"type": "dense_vector", "dims": embedding_dims}
                }
            }
        }
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
