#!/usr/bin/env bash
set -euo pipefail

ES_HOST=${ES_HOST:-"http://localhost:9200"}
ES_USER=${ES_USERNAME:-"elastic"}
ES_PASS=${ES_PASSWORD:-"changeme"}
INDEX_NAME=${ES_INDEX_NAME:-"patents"}

echo "Creating index [$INDEX_NAME] at $ES_HOST"

curl -s -u "$ES_USER:$ES_PASS" -X PUT "$ES_HOST/$INDEX_NAME" -H 'Content-Type: application/json' -d @- <<'JSON'
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "doc_number":     { "type": "keyword" },
      "classification": { "type": "keyword" },
      "section_type":   { "type": "keyword" },
      "title":          { "type": "text", "analyzer": "english" },
      "abstract":       { "type": "text", "analyzer": "english" },
      "claims_text":    { "type": "text", "analyzer": "english" },
      "desc_text":      { "type": "text", "analyzer": "english" },
      "embedding": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine",
        "index_options": { "type": "hnsw", "m": 32, "ef_construction": 100 }
      },
      "ml.tokens": { "type": "sparse_vector" }
    }
  }
}
JSON

echo "Done."
