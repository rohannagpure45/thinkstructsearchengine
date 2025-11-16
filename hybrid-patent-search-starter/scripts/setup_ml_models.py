#!/usr/bin/env python3
"""
Setup ML models for hybrid patent search:
1. Deploy ELSER v2 model in Elasticsearch
2. Initialize sentence-transformers for dense embeddings
"""

import os
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
from app.es_client import get_es

load_dotenv()

def wait_for_model_deployment(es: Elasticsearch, model_id: str, timeout: int = 300):
    """Wait for ML model to be deployed and started"""
    print(f"Waiting for model '{model_id}' to be deployed...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check model stats
            stats = es.ml.get_trained_models_stats(model_id=model_id)
            if stats.get("trained_model_stats"):
                model_stats = stats["trained_model_stats"][0]
                deployment_stats = model_stats.get("deployment_stats", {})
                state = deployment_stats.get("state", "")
                
                if state == "started":
                    print(f"✓ Model '{model_id}' is deployed and started!")
                    return True
                elif state == "failed":
                    error = deployment_stats.get("reason", "Unknown error")
                    print(f"✗ Model deployment failed: {error}")
                    return False
                else:
                    print(f"  Model state: {state}")
                    
        except Exception as e:
            print(f"  Checking model status... ({e})")
            
        time.sleep(5)
    
    print(f"✗ Timeout waiting for model deployment after {timeout}s")
    return False

def deploy_elser_model(es: Elasticsearch):
    """Deploy ELSER v2 model in Elasticsearch"""
    elser_model_id = os.getenv("ELSER_MODEL_ID", ".elser_model_2")
    
    print(f"\n=== Deploying ELSER v2 Model: {elser_model_id} ===")
    
    # Check if model exists
    try:
        model_info = es.ml.get_trained_models(model_id=elser_model_id)
        print(f"✓ Model '{elser_model_id}' already exists")
    except:
        print(f"✗ Model '{elser_model_id}' not found. Please ensure you have the correct Elasticsearch license.")
        print("  ELSER requires at least a Platinum license or Elastic Cloud.")
        print("  You can start a free trial: https://www.elastic.co/subscriptions")
        return False
    
    # Check deployment status
    try:
        stats = es.ml.get_trained_models_stats(model_id=elser_model_id)
        model_stats = stats.get("trained_model_stats", [{}])[0]
        deployment_stats = model_stats.get("deployment_stats", {})
        
        if deployment_stats.get("state") == "started":
            print(f"✓ Model '{elser_model_id}' is already deployed and running")
            return True
            
    except Exception as e:
        print(f"  Model not deployed yet: {e}")
    
    # Deploy the model
    print(f"Deploying model '{elser_model_id}'...")
    try:
        # Start deployment
        es.ml.start_trained_model_deployment(
            model_id=elser_model_id,
            number_of_allocations=1,
            threads_per_allocation=2,
            queue_capacity=1000,
            wait_for="started",
            timeout="5m"
        )
        
        # Wait for deployment
        if wait_for_model_deployment(es, elser_model_id):
            print(f"✓ Successfully deployed ELSER model: {elser_model_id}")
            return True
        else:
            print(f"✗ Failed to deploy ELSER model")
            return False
            
    except Exception as e:
        print(f"✗ Error deploying model: {e}")
        return False

def test_elser_model(es: Elasticsearch):
    """Test ELSER model with sample text"""
    elser_model_id = os.getenv("ELSER_MODEL_ID", ".elser_model_2")
    
    print(f"\n=== Testing ELSER Model ===")
    
    test_text = "A method for detecting molecular structures using optical sensors"
    
    try:
        # Test inference
        response = es.ml.infer_trained_model(
            model_id=elser_model_id,
            docs=[{"text_field": test_text}]
        )
        
        # Check response
        if response.get("inference_results"):
            tokens = response["inference_results"][0].get("predicted_value", {})
            print(f"✓ ELSER inference successful! Generated {len(tokens)} tokens")
            print(f"  Sample tokens: {list(tokens.keys())[:5]}")
            return True
        else:
            print("✗ ELSER inference returned no results")
            return False
            
    except Exception as e:
        print(f"✗ Error testing ELSER: {e}")
        return False

def initialize_dense_embedder():
    """Initialize and test sentence-transformer model"""
    model_name = os.getenv("DENSE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"\n=== Initializing Dense Embedder: {model_name} ===")
    
    try:
        # Load model
        print(f"Loading model '{model_name}'...")
        model = SentenceTransformer(model_name)
        
        # Test embedding
        test_texts = [
            "A method for detecting molecular structures",
            "Optical sensor for chemical analysis"
        ]
        
        embeddings = model.encode(test_texts, show_progress_bar=False)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Embedding dimensions: {embeddings.shape[1]}")
        print(f"  Test embeddings shape: {embeddings.shape}")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "embedding_dims": int(embeddings.shape[1]),
            "max_seq_length": model.max_seq_length
        }
        
        # Ensure config directory exists
        config_dir = Path(__file__).parent.parent / "config"
        config_dir.mkdir(exist_ok=True)
        
        with open(config_dir / "embedding_model.json", "w") as f:
            json.dump(model_info, f, indent=2)
            
        print(f"✓ Model info saved to config/embedding_model.json")
        
        return True
        
    except Exception as e:
        print(f"✗ Error initializing dense embedder: {e}")
        return False

def verify_ml_node(es: Elasticsearch):
    """Verify ML node is available"""
    print("\n=== Verifying ML Node Configuration ===")
    
    try:
        # Get ML info
        ml_info = es.ml.info()
        
        if ml_info.get("defaults", {}).get("ml", {}).get("enabled", False):
            print("✓ ML features are enabled")
        else:
            print("✗ ML features are not enabled. Please set xpack.ml.enabled=true")
            return False
            
        # Check nodes
        nodes = es.nodes.info()
        ml_nodes = 0
        
        for node_id, node_info in nodes.get("nodes", {}).items():
            roles = node_info.get("roles", [])
            if "ml" in roles:
                ml_nodes += 1
                print(f"✓ Found ML node: {node_info.get('name')}")
                
        if ml_nodes == 0:
            print("✗ No ML nodes found. Please ensure at least one node has the 'ml' role")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error checking ML configuration: {e}")
        return False

def main():
    """Main setup function"""
    print("Starting ML Model Setup for Patent Search")
    print("=" * 50)
    
    # Get Elasticsearch client
    es = get_es()
    
    # Check Elasticsearch connection
    try:
        info = es.info()
        print(f"✓ Connected to Elasticsearch {info['version']['number']}")
    except Exception as e:
        print(f"✗ Failed to connect to Elasticsearch: {e}")
        sys.exit(1)
    
    # Verify ML node
    if not verify_ml_node(es):
        print("\n⚠️  ML node verification failed. ELSER deployment may not work.")
        print("   Continuing with dense embeddings setup...")
    else:
        # Deploy ELSER
        elser_success = deploy_elser_model(es)
        
        if elser_success:
            # Test ELSER
            test_elser_model(es)
        else:
            print("\n⚠️  ELSER deployment failed. You can still use dense embeddings.")
    
    # Initialize dense embedder
    dense_success = initialize_dense_embedder()
    
    # Summary
    print("\n" + "=" * 50)
    print("Setup Summary:")
    print(f"  ELSER v2 Model: {'✓ Ready' if 'elser_success' in locals() and elser_success else '✗ Not available'}")
    print(f"  Dense Embedder: {'✓ Ready' if dense_success else '✗ Failed'}")
    
    if not (('elser_success' in locals() and elser_success) or dense_success):
        print("\n✗ No embedding models available. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✓ ML models are ready for hybrid search!")

if __name__ == "__main__":
    main()
