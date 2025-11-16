#!/usr/bin/env python3
"""
Ingest patent data into Elasticsearch with chunking and dual embeddings
"""

import argparse
import os
import re
import sys
import glob
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.es_client import get_es
from config import config, get_pipeline_name

load_dotenv()


class PatentIngester:
    """Handles patent data ingestion with chunking and embeddings"""
    
    def __init__(
        self,
        es_client: Elasticsearch,
        use_dense_embeddings: bool = True,
        use_elser: bool = True,
        batch_size: int = 100,
        chunk_size: int = 400
    ):
        self.es = es_client
        self.use_dense_embeddings = use_dense_embeddings
        self.use_elser = use_elser
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # Initialize sentence transformer if needed
        self.embedder = None
        if use_dense_embeddings:
            model_info = config.models.embedding_model_info
            print(f"Loading embedding model: {model_info['model_name']}")
            self.embedder = SentenceTransformer(model_info['model_name'])
            self.embedding_dims = model_info['embedding_dims']
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Chunk text into overlapping segments
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk (in words)
            overlap: Number of words to overlap between chunks
        
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            
            if end >= len(words):
                break
                
            start = end - overlap
        
        return chunks
    
    def parse_claims(self, claims_data: Any) -> List[Dict[str, Any]]:
        """
        Parse claims into structured format
        
        Args:
            claims_data: Claims data (can be string, list, or dict)
        
        Returns:
            List of parsed claims with metadata
        """
        claims = []
        
        if isinstance(claims_data, str):
            # Simple string format - split by claim numbers
            # Split by patterns like "1.", "2.", etc.
            claim_pattern = r'(\d+)\.\s*(.*?)(?=\d+\.|$)'
            matches = re.findall(claim_pattern, claims_data, re.DOTALL)
            
            for num, text in matches:
                text = text.strip()
                if text:
                    # Default to independent claim
                    claim_type = "independent"
                    dependency = None
                    
                    # Check for explicit claim reference at the start of the claim text
                    # Pattern matches: "The/An/A ... of/according to claim(s) X" at the beginning
                    # Case-insensitive match after trimming whitespace
                    dep_pattern = r'^(the|an|a)\s+.*?(?:of|according\s+to)\s+claim[s]?\s+(\d+)'
                    dep_match = re.search(dep_pattern, text, re.IGNORECASE)
                    if dep_match:
                        claim_type = "dependent"
                        dependency = int(dep_match.group(2))
                    
                    claims.append({
                        "claim_num": int(num),
                        "claim_text": text,
                        "claim_type": claim_type,
                        "claim_dependency": dependency
                    })
        
        elif isinstance(claims_data, list):
            # List format
            for i, claim_text in enumerate(claims_data, 1):
                if isinstance(claim_text, str) and claim_text.strip():
                    claims.append({
                        "claim_num": i,
                        "claim_text": claim_text.strip(),
                        "claim_type": "unknown",
                        "claim_dependency": None
                    })
        
        return claims
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate dense embeddings for texts"""
        if not self.embedder or not texts:
            return np.array([])
        
        return self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    def create_patent_core_doc(self, patent: Dict[str, Any]) -> Dict[str, Any]:
        """Create patent core document"""
        # Extract basic fields
        patent_id = patent.get("patent_id") or patent.get("doc_number") or str(uuid.uuid4())
        title = self.normalize_text(patent.get("title") or patent.get("Title"))
        abstract = self.normalize_text(patent.get("abstract") or patent.get("Abstract"))
        
        doc = {
            "_op_type": "index",
            "_index": config.indices.patents_core,
            "_id": patent_id,
            "_source": {
                "patent_id": patent_id,
                "doc_number": patent.get("doc_number") or patent.get("Document Number"),
                "title": title,
                "abstract": abstract,
                "filing_date": patent.get("filing_date") or patent.get("Filing Date"),
                "publication_date": patent.get("publication_date") or patent.get("Publication Date"),
                "inventor": self.normalize_list(patent.get("inventor") or patent.get("Inventor")),
                "assignee": self.normalize_list(patent.get("assignee") or patent.get("Assignee")),
                "cpc_codes": self.normalize_list(patent.get("cpc_codes") or patent.get("CPC")),
                "ipc_codes": self.normalize_list(patent.get("ipc_codes") or patent.get("IPC"))
            }
        }
        
        # Add dense embeddings
        if self.use_dense_embeddings and (title or abstract):
            embeddings = self.generate_embeddings([title or "", abstract or ""])
            if len(embeddings) > 0 and title:
                doc["_source"]["title_vec"] = embeddings[0].tolist()
            if len(embeddings) > 1 and abstract:
                doc["_source"]["abstract_vec"] = embeddings[1].tolist()
        
        # Add ELSER pipeline
        if self.use_elser:
            pipeline = get_pipeline_name(config.indices.patents_core)
            if pipeline:
                doc["pipeline"] = pipeline
        
        return doc
    
    def create_claim_docs(self, patent_id: str, claims_data: Any) -> List[Dict[str, Any]]:
        """Create claim chunk documents"""
        claims = self.parse_claims(claims_data)
        docs = []
        
        # Extract claim texts for batch embedding
        claim_texts = [c["claim_text"] for c in claims if c["claim_text"]]
        
        # Generate embeddings in batch
        if self.use_dense_embeddings and claim_texts:
            embeddings = self.generate_embeddings(claim_texts)
        else:
            embeddings = []
        
        # Create documents
        for i, claim in enumerate(claims):
            doc = {
                "_op_type": "index",
                "_index": config.indices.claims_chunks,
                "_id": f"{patent_id}#CLAIM#{claim['claim_num']}",
                "_source": {
                    "patent_id": patent_id,
                    "claim_num": claim["claim_num"],
                    "claim_text": claim["claim_text"],
                    "claim_type": claim["claim_type"],
                    "claim_dependency": claim["claim_dependency"]
                }
            }
            
            # Add dense embedding
            if self.use_dense_embeddings and i < len(embeddings):
                doc["_source"]["claim_vec"] = embeddings[i].tolist()
            
            # Add ELSER pipeline
            if self.use_elser:
                pipeline = get_pipeline_name(config.indices.claims_chunks)
                if pipeline:
                    doc["pipeline"] = pipeline
            
            docs.append(doc)
        
        return docs
    
    def create_description_docs(
        self,
        patent_id: str,
        description: str
    ) -> List[Dict[str, Any]]:
        """Create description chunk documents"""
        if not description:
            return []
        
        # Chunk the description
        chunks = self.chunk_text(description, chunk_size=self.chunk_size, overlap=50)
        
        # Generate embeddings in batch
        if self.use_dense_embeddings and chunks:
            embeddings = self.generate_embeddings(chunks)
        else:
            embeddings = []
        
        docs = []
        
        for i, chunk_text in enumerate(chunks):
            doc = {
                "_op_type": "index",
                "_index": config.indices.desc_chunks,
                "_id": f"{patent_id}#DESC#{i+1}",
                "_source": {
                    "patent_id": patent_id,
                    "chunk_id": f"chunk_{i+1}",
                    "paragraph_num": i + 1,
                    "desc_text": chunk_text
                }
            }
            
            # Add dense embedding
            if self.use_dense_embeddings and i < len(embeddings):
                doc["_source"]["desc_vec"] = embeddings[i].tolist()
            
            # Add ELSER pipeline
            if self.use_elser:
                pipeline = get_pipeline_name(config.indices.desc_chunks)
                if pipeline:
                    doc["pipeline"] = pipeline
            
            docs.append(doc)
        
        return docs
    
    def normalize_text(self, text: Any) -> Optional[str]:
        """Normalize text field"""
        if not text:
            return None
        
        if isinstance(text, list):
            text = " ".join([str(seg) for seg in text if seg])
        
        return str(text).strip()
    
    def normalize_list(self, data: Any) -> List[str]:
        """Normalize list field"""
        if not data:
            return []
        
        if isinstance(data, str):
            # Split by common delimiters
            items = [item.strip() for item in data.split(";")]
            return [item for item in items if item]
        
        if isinstance(data, list):
            return [str(item).strip() for item in data if item]
        
        return [str(data).strip()]
    
    def process_patent(self, patent: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process a single patent and yield all documents"""
        # Extract patent ID
        patent_id = patent.get("patent_id") or patent.get("doc_number") or str(uuid.uuid4())
        
        # Create patent core document
        yield self.create_patent_core_doc(patent)
        
        # Create claim documents
        claims_data = patent.get("claims") or patent.get("Claims")
        if claims_data:
            for doc in self.create_claim_docs(patent_id, claims_data):
                yield doc
        
        # Create description documents
        description = self.normalize_text(
            patent.get("description") or patent.get("Description") or patent.get("Specification")
        )
        if description:
            for doc in self.create_description_docs(patent_id, description):
                yield doc
    
    def ingest_patents(self, patents: Iterator[Dict[str, Any]], show_progress: bool = True):
        """Ingest patents into Elasticsearch"""
        total_docs = 0
        failed_docs = 0
        
        # Create progress bar
        if show_progress:
            pbar = tqdm(desc="Indexing documents", unit="docs")
        
        # Process in batches
        batch_actions = []
        
        for patent in patents:
            try:
                # Process patent and add to batch
                for doc in self.process_patent(patent):
                    batch_actions.append(doc)
                    
                    # Index batch if full
                    if len(batch_actions) >= self.batch_size:
                        success, failed = self._index_batch(batch_actions)
                        total_docs += success
                        failed_docs += failed
                        
                        if show_progress:
                            pbar.update(success)
                        
                        batch_actions = []
                        
            except Exception as e:
                print(f"Error processing patent: {e}")
                failed_docs += 1
        
        # Index remaining documents
        if batch_actions:
            success, failed = self._index_batch(batch_actions)
            total_docs += success
            failed_docs += failed
            
            if show_progress:
                pbar.update(success)
        
        if show_progress:
            pbar.close()
        
        # Refresh indices
        for index in config.get_index_list():
            try:
                self.es.indices.refresh(index=index)
            except Exception as e:
                logging.exception(f"Warning: Failed to refresh index {index}: {e}")
        
        return total_docs, failed_docs
    
    def _index_batch(self, actions: List[Dict[str, Any]]) -> tuple:
        """Index a batch of documents"""
        success_count = 0
        failed_count = 0
        
        for ok, result in streaming_bulk(
            self.es,
            actions,
            chunk_size=50,
            max_retries=3,
            request_timeout=60
        ):
            if ok:
                success_count += 1
            else:
                failed_count += 1
                print(f"Failed to index: {result}")
        
        return success_count, failed_count


def iter_patents_from_file(filepath: str) -> Iterator[Dict[str, Any]]:
    """Iterate patents from a JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict) and "patents" in data:
                data = data["patents"]
            
            if not isinstance(data, list):
                data = [data]
            
            for patent in data:
                if isinstance(patent, dict):
                    yield patent
                    
        except Exception as e:
            print(f"Error reading {filepath}: {e}")


def iter_patents_from_path(path: str) -> Iterator[Dict[str, Any]]:
    """Iterate patents from file or directory"""
    path_obj = Path(path)
    
    if path_obj.is_file():
        # Single file
        yield from iter_patents_from_file(str(path_obj))
    
    elif path_obj.is_dir():
        # Directory of files
        json_files = sorted(path_obj.glob("*.json"))
        
        for json_file in json_files:
            print(f"Processing: {json_file}")
            yield from iter_patents_from_file(str(json_file))
    
    else:
        raise ValueError(f"Path does not exist: {path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest patent data into Elasticsearch")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSON file or directory containing patent data"
    )
    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="Disable dense embeddings (only use BM25 + ELSER)"
    )
    parser.add_argument(
        "--no-elser",
        action="store_true",
        help="Disable ELSER (only use BM25 + dense)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing (default: 100)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Chunk size for description text in words (default: 400)"
    )
    
    args = parser.parse_args()
    
    # Initialize Elasticsearch client
    es = get_es()
    
    # Test connection
    try:
        info = es.info()
        print(f"Connected to Elasticsearch {info['version']['number']}")
    except Exception as e:
        print(f"Failed to connect to Elasticsearch: {e}")
        sys.exit(1)
    
    # Check indices exist
    for index in config.get_index_list():
        if not es.indices.exists(index=index):
            print(f"Warning: Index '{index}' does not exist. Run create_indices.sh first.")
    
    # Initialize ingester
    ingester = PatentIngester(
        es_client=es,
        use_dense_embeddings=not args.no_dense,
        use_elser=not args.no_elser,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    
    print(f"\nIngestion settings:")
    print(f"  Dense embeddings: {'Enabled' if not args.no_dense else 'Disabled'}")
    print(f"  ELSER v2: {'Enabled' if not args.no_elser else 'Disabled'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Description chunk size: {args.chunk_size} words")
    print("")
    
    # Get patent iterator
    patents = iter_patents_from_path(args.input)
    
    # Ingest patents
    print("Starting ingestion...")
    total, failed = ingester.ingest_patents(patents, show_progress=True)
    
    print(f"\nIngestion complete!")
    print(f"  Total documents indexed: {total}")
    print(f"  Failed documents: {failed}")
    
    # Show index counts
    print("\nIndex statistics:")
    for index in config.get_index_list():
        try:
            count = es.count(index=index)
            print(f"  {index}: {count['count']} documents")
        except Exception as e:
            print(f"  {index}: Error getting count - {e}")


if __name__ == "__main__":
    main()