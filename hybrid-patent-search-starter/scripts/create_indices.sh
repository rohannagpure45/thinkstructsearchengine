#!/usr/bin/env bash
set -euo pipefail

# Load environment variables
ES_HOST=${ES_HOST:-"http://localhost:9200"}
ES_USER=${ES_USERNAME:-"elastic"}
ES_PASS=${ES_PASSWORD:-"changeme"}

# Index names from environment or defaults
PATENTS_CORE=${ES_INDEX_PATENTS_CORE:-"patents_core"}
CLAIMS_CHUNKS=${ES_INDEX_CLAIMS_CHUNKS:-"claims_chunks"}
DESC_CHUNKS=${ES_INDEX_DESC_CHUNKS:-"desc_chunks"}

# Embedding dimensions
EMBEDDING_DIMS=${EMBEDDING_DIMS:-384}

echo "Creating patent search indices at $ES_HOST"
echo "Using embedding dimensions: $EMBEDDING_DIMS"

# Function to create an index
create_index() {
  local index_name=$1
  local mapping=$2
  
  echo "Creating index: $index_name"
  
  # Delete if exists (for development)
  curl -s -u "$ES_USER:$ES_PASS" -X DELETE "$ES_HOST/$index_name" 2>/dev/null || true
  
  # Create index with mapping
  curl -s -u "$ES_USER:$ES_PASS" -X PUT "$ES_HOST/$index_name" \
    -H 'Content-Type: application/json' \
    -d "$mapping" | jq '.' || echo "Failed to create $index_name"
}

# Patents Core Index - Patent-level metadata
PATENTS_CORE_MAPPING=$(cat <<JSON
{
  "settings": {
    "number_of_shards": 2,
    "number_of_replicas": 0,
    "analysis": {
      "analyzer": {
        "patent_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "char_filter": ["html_strip"],
          "filter": [
            "lowercase",
            "english_stop",
            "english_keywords",
            "english_stemmer",
            "trim"
          ]
        }
      },
      "filter": {
        "english_stop": {
          "type": "stop",
          "stopwords": "_english_"
        },
        "english_keywords": {
          "type": "keyword_marker",
          "keywords": ["patent", "invention", "claim", "embodiment", "molecule", "compound"]
        },
        "english_stemmer": {
          "type": "stemmer",
          "language": "english"
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "patent_id": { "type": "keyword" },
      "doc_number": { "type": "keyword" },
      "title": { 
        "type": "text", 
        "analyzer": "patent_analyzer",
        "fields": {
          "keyword": { "type": "keyword" }
        }
      },
      "abstract": { 
        "type": "text", 
        "analyzer": "patent_analyzer" 
      },
      "filing_date": { "type": "date" },
      "publication_date": { "type": "date" },
      "inventor": { "type": "keyword" },
      "assignee": { "type": "keyword" },
      "cpc_codes": { "type": "keyword" },
      "ipc_codes": { "type": "keyword" },
      "title_vec": {
        "type": "dense_vector",
        "dims": $EMBEDDING_DIMS,
        "index": true,
        "similarity": "cosine",
        "index_options": {
          "type": "hnsw",
          "m": 32,
          "ef_construction": 100
        }
      },
      "abstract_vec": {
        "type": "dense_vector",
        "dims": $EMBEDDING_DIMS,
        "index": true,
        "similarity": "cosine",
        "index_options": {
          "type": "hnsw",
          "m": 32,
          "ef_construction": 100
        }
      },
      "title_elser": { "type": "sparse_vector" },
      "abstract_elser": { "type": "sparse_vector" }
    }
  }
}
JSON
)

# Claims Chunks Index - Individual claims
CLAIMS_CHUNKS_MAPPING=$(cat <<JSON
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 0,
    "analysis": {
      "analyzer": {
        "patent_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "char_filter": ["html_strip"],
          "filter": [
            "lowercase",
            "english_stop",
            "english_keywords",
            "english_stemmer",
            "trim"
          ]
        }
      },
      "filter": {
        "english_stop": {
          "type": "stop",
          "stopwords": "_english_"
        },
        "english_keywords": {
          "type": "keyword_marker",
          "keywords": ["patent", "invention", "claim", "embodiment", "molecule", "compound", "comprising", "wherein", "thereof"]
        },
        "english_stemmer": {
          "type": "stemmer",
          "language": "english"
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "patent_id": { "type": "keyword" },
      "claim_num": { "type": "integer" },
      "claim_type": { 
        "type": "keyword",
        "ignore_above": 20
      },
      "claim_text": { 
        "type": "text", 
        "analyzer": "patent_analyzer",
        "term_vector": "with_positions_offsets"
      },
      "claim_dependency": { "type": "integer" },
      "claim_vec": {
        "type": "dense_vector",
        "dims": $EMBEDDING_DIMS,
        "index": true,
        "similarity": "cosine",
        "index_options": {
          "type": "hnsw",
          "m": 32,
          "ef_construction": 100
        }
      },
      "claim_elser": { "type": "sparse_vector" }
    }
  }
}
JSON
)

# Description Chunks Index - Paragraphs from description
DESC_CHUNKS_MAPPING=$(cat <<JSON
{
  "settings": {
    "number_of_shards": 2,
    "number_of_replicas": 0,
    "analysis": {
      "analyzer": {
        "patent_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "char_filter": ["html_strip"],
          "filter": [
            "lowercase",
            "english_stop",
            "english_keywords",
            "english_stemmer",
            "trim"
          ]
        }
      },
      "filter": {
        "english_stop": {
          "type": "stop",
          "stopwords": "_english_"
        },
        "english_keywords": {
          "type": "keyword_marker",
          "keywords": ["patent", "invention", "figure", "embodiment", "molecule", "compound", "example"]
        },
        "english_stemmer": {
          "type": "stemmer",
          "language": "english"
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "patent_id": { "type": "keyword" },
      "chunk_id": { "type": "keyword" },
      "paragraph_num": { "type": "integer" },
      "desc_text": { 
        "type": "text", 
        "analyzer": "patent_analyzer",
        "term_vector": "with_positions_offsets"
      },
      "section_heading": { "type": "keyword" },
      "desc_vec": {
        "type": "dense_vector",
        "dims": $EMBEDDING_DIMS,
        "index": true,
        "similarity": "cosine",
        "index_options": {
          "type": "hnsw",
          "m": 32,
          "ef_construction": 100
        }
      },
      "desc_elser": { "type": "sparse_vector" }
    }
  }
}
JSON
)

# Create all indices
create_index "$PATENTS_CORE" "$PATENTS_CORE_MAPPING"
create_index "$CLAIMS_CHUNKS" "$CLAIMS_CHUNKS_MAPPING"
create_index "$DESC_CHUNKS" "$DESC_CHUNKS_MAPPING"

echo ""
echo "All indices created successfully!"
echo "Patents core index: $PATENTS_CORE"
echo "Claims chunks index: $CLAIMS_CHUNKS"
echo "Description chunks index: $DESC_CHUNKS"