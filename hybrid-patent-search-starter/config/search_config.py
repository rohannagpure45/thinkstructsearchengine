"""
Centralized configuration for hybrid patent search
"""

import os
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import json


class IndexConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    """Configuration for Elasticsearch indices"""
    
    patents_core: str = Field(
        default="patents_core",
        env="ES_INDEX_PATENTS_CORE",
        description="Index for patent-level metadata"
    )
    claims_chunks: str = Field(
        default="claims_chunks",
        env="ES_INDEX_CLAIMS_CHUNKS", 
        description="Index for individual claims"
    )
    desc_chunks: str = Field(
        default="desc_chunks",
        env="ES_INDEX_DESC_CHUNKS",
        description="Index for description paragraphs"
    )
    
class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    """Configuration for ML models"""
    
    elser_model_id: str = Field(
        default=".elser_model_2",
        env="ELSER_MODEL_ID",
        description="ELSER v2 model identifier"
    )
    dense_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="DENSE_EMBEDDING_MODEL",
        description="Sentence transformer model for dense embeddings"
    )
    embedding_dims: int = Field(
        default=384,
        env="EMBEDDING_DIMS",
        description="Dimension of dense embeddings"
    )
    
    @property
    def embedding_model_info(self) -> Dict:
        """Load embedding model info if available"""
        config_file = Path(__file__).parent / "embedding_model.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                # Fall back to defaults if file is malformed or unreadable
                pass
        return {
            "model_name": self.dense_embedding_model,
            "embedding_dims": self.embedding_dims,
            "max_seq_length": 256
        }
    
class SearchConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    """Configuration for search parameters"""
    
    default_top_k: int = Field(
        default=20,
        env="DEFAULT_TOP_K",
        description="Default number of results to return"
    )
    rrf_rank_constant: int = Field(
        default=60,
        env="RRF_RANK_CONSTANT",
        description="RRF rank constant parameter"
    )
    rrf_window_size: int = Field(
        default=50,
        env="RRF_WINDOW_SIZE",
        description="RRF window size for fusion"
    )
    num_candidates_knn: int = Field(
        default=200,
        env="NUM_CANDIDATES_KNN",
        description="Number of candidates for kNN search"
    )
    
    # Field configurations
    bm25_fields: Dict[str, List[str]] = {
        "patents_core": ["title^3", "abstract^2"],
        "claims_chunks": ["claim_text^3"],
        "desc_chunks": ["desc_text^2"]
    }
    
    dense_vector_fields: Dict[str, str] = {
        "patents_core": "abstract_vec",  # Primary vector field
        "claims_chunks": "claim_vec",
        "desc_chunks": "desc_vec"
    }
    
    sparse_vector_fields: Dict[str, List[str]] = {
        "patents_core": ["title_elser", "abstract_elser"],
        "claims_chunks": ["claim_elser"],
        "desc_chunks": ["desc_elser"]
    }
    
    # Highlighting configuration
    highlight_config: Dict = {
        "fields": {
            "title": {"number_of_fragments": 1, "fragment_size": 200},
            "abstract": {"number_of_fragments": 3, "fragment_size": 150},
            "claim_text": {"number_of_fragments": 2, "fragment_size": 200},
            "desc_text": {"number_of_fragments": 2, "fragment_size": 150}
        },
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"],
        "encoder": "html"
    }
    
class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    """Configuration for ingest pipelines"""
    
    elser_pipeline_base: str = Field(
        default="elser_ingest_pipeline",
        env="ELSER_PIPELINE_NAME",
        description="Base name for ELSER pipelines"
    )
    
    @property
    def pipeline_names(self) -> Dict[str, str]:
        """Get pipeline names for each index"""
        return {
            "patents_core": f"{self.elser_pipeline_base}_patents",
            "claims_chunks": f"{self.elser_pipeline_base}_claims",
            "desc_chunks": f"{self.elser_pipeline_base}_desc"
        }
    
class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    """Main configuration class"""
    
    # Sub-configurations
    indices: IndexConfig = IndexConfig()
    models: ModelConfig = ModelConfig()
    search: SearchConfig = SearchConfig()
    pipelines: PipelineConfig = PipelineConfig()
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    
    def get_index_list(self) -> List[str]:
        """Get list of all index names"""
        return [
            self.indices.patents_core,
            self.indices.claims_chunks,
            self.indices.desc_chunks
        ]
    
    def get_index_for_doc_type(self, doc_type: str) -> Optional[str]:
        """Get index name for a document type"""
        mapping = {
            "patent": self.indices.patents_core,
            "claim": self.indices.claims_chunks,
            "description": self.indices.desc_chunks
        }
        return mapping.get(doc_type)
    
    def get_search_indices(self, search_scope: str = "all") -> List[str]:
        """Get indices to search based on scope"""
        if search_scope == "patents":
            return [self.indices.patents_core]
        elif search_scope == "claims":
            return [self.indices.claims_chunks]
        elif search_scope == "descriptions":
            return [self.indices.desc_chunks]
        else:  # all
            return [
                self.indices.claims_chunks,
                self.indices.desc_chunks,
                self.indices.patents_core
            ]
    
# Singleton instance
config = Config()


# Helper functions for common access patterns
def get_bm25_fields(index: str) -> List[str]:
    """Get BM25 fields for an index"""
    return config.search.bm25_fields.get(index, [])


def get_dense_vector_field(index: str) -> Optional[str]:
    """Get dense vector field for an index"""
    return config.search.dense_vector_fields.get(index)


def get_sparse_vector_fields(index: str) -> List[str]:
    """Get sparse vector fields for an index"""
    return config.search.sparse_vector_fields.get(index, [])


def get_pipeline_name(index: str) -> Optional[str]:
    """Get pipeline name for an index"""
    return config.pipelines.pipeline_names.get(index)
