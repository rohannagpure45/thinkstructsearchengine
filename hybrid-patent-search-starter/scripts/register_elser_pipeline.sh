#!/usr/bin/env bash
set -euo pipefail

# Load environment variables
ES_HOST=${ES_HOST:-"http://localhost:9200"}
ES_USER=${ES_USERNAME:-"elastic"}
ES_PASS=${ES_PASSWORD:-"changeme"}
ES_API_KEY=${ES_API_KEY:-""}

if [ -n "$ES_API_KEY" ]; then
  CURL_AUTH=(-H "Authorization: ApiKey $ES_API_KEY")
else
  CURL_AUTH=(-u "$ES_USER:$ES_PASS")
fi
ELSER_MODEL_ID=${ELSER_MODEL_ID:-".elser_model_2"}

# Pipeline names
BASE_PIPELINE=${ELSER_PIPELINE_NAME:-"elser_ingest_pipeline"}
PATENTS_PIPELINE="${BASE_PIPELINE}_patents"
CLAIMS_PIPELINE="${BASE_PIPELINE}_claims"
DESC_PIPELINE="${BASE_PIPELINE}_desc"

echo "Registering ELSER ingest pipelines using model [$ELSER_MODEL_ID]"
echo "=================================================="

# Function to create pipeline
create_pipeline() {
  local pipeline_name=$1
  local pipeline_config=$2
  
  echo "Creating pipeline: $pipeline_name"
  
  # Delete if exists
  curl -s "${CURL_AUTH[@]}" -X DELETE "$ES_HOST/_ingest/pipeline/$pipeline_name" 2>/dev/null || true
  
  # Create pipeline
  response=$(curl -s "${CURL_AUTH[@]}" -X PUT "$ES_HOST/_ingest/pipeline/$pipeline_name" \
    -H 'Content-Type: application/json' \
    -d "$pipeline_config")
  
  if echo "$response" | grep -q '"acknowledged":true'; then
    echo "✓ Pipeline '$pipeline_name' created successfully"
  else
    echo "✗ Failed to create pipeline '$pipeline_name'"
    echo "  Response: $response"
  fi
}

# Patents Core Pipeline - processes title and abstract
PATENTS_PIPELINE_CONFIG=$(cat <<JSON
{
  "description": "ELSER v2 inference for patent titles and abstracts",
  "processors": [
    {
      "inference": {
        "model_id": "$ELSER_MODEL_ID",
        "if": "ctx.title != null && ctx.title.length() > 0",
        "input_output": [
          {
            "input_field": "title",
            "output_field": "title_elser"
          }
        ],
        "on_failure": [
          {
            "set": {
              "field": "_ingest_error",
              "value": "ELSER inference failed for title: {{_ingest.on_failure_message}}"
            }
          }
        ]
      }
    },
    {
      "inference": {
        "model_id": "$ELSER_MODEL_ID",
        "if": "ctx.abstract != null && ctx.abstract.length() > 0",
        "input_output": [
          {
            "input_field": "abstract",
            "output_field": "abstract_elser"
          }
        ],
        "on_failure": [
          {
            "set": {
              "field": "_ingest_error",
              "value": "ELSER inference failed for abstract: {{_ingest.on_failure_message}}"
            }
          }
        ]
      }
    }
  ],
  "on_failure": [
    {
      "set": {
        "field": "_index_error",
        "value": "Pipeline failed: {{_ingest.on_failure_message}}"
      }
    }
  ]
}
JSON
)

# Claims Pipeline - processes claim text
CLAIMS_PIPELINE_CONFIG=$(cat <<JSON
{
  "description": "ELSER v2 inference for patent claims",
  "processors": [
    {
      "inference": {
        "model_id": "$ELSER_MODEL_ID",
        "if": "ctx.claim_text != null && ctx.claim_text.length() > 0",
        "input_output": [
          {
            "input_field": "claim_text",
            "output_field": "claim_elser"
          }
        ],
        "on_failure": [
          {
            "set": {
              "field": "_ingest_error",
              "value": "ELSER inference failed for claim_text: {{_ingest.on_failure_message}}"
            }
          }
        ]
      }
    }
  ],
  "on_failure": [
    {
      "set": {
        "field": "_index_error",
        "value": "Pipeline failed: {{_ingest.on_failure_message}}"
      }
    }
  ]
}
JSON
)

# Description Pipeline - processes description text chunks
DESC_PIPELINE_CONFIG=$(cat <<JSON
{
  "description": "ELSER v2 inference for patent descriptions",
  "processors": [
    {
      "inference": {
        "model_id": "$ELSER_MODEL_ID",
        "if": "ctx.desc_text != null && ctx.desc_text.length() > 0",
        "input_output": [
          {
            "input_field": "desc_text",
            "output_field": "desc_elser"
          }
        ],
        "on_failure": [
          {
            "set": {
              "field": "_ingest_error",
              "value": "ELSER inference failed for desc_text: {{_ingest.on_failure_message}}"
            }
          }
        ]
      }
    }
  ],
  "on_failure": [
    {
      "set": {
        "field": "_index_error",
        "value": "Pipeline failed: {{_ingest.on_failure_message}}"
      }
    }
  ]
}
JSON
)

# Create all pipelines
# index through the appropriate pipeline (?pipeline=...) so those fields are ready for semantic queries
create_pipeline "$PATENTS_PIPELINE" "$PATENTS_PIPELINE_CONFIG"
create_pipeline "$CLAIMS_PIPELINE" "$CLAIMS_PIPELINE_CONFIG"
create_pipeline "$DESC_PIPELINE" "$DESC_PIPELINE_CONFIG"

echo ""
echo "All ELSER pipelines registered successfully!"
echo "Patents pipeline: $PATENTS_PIPELINE"
echo "Claims pipeline: $CLAIMS_PIPELINE"
echo "Description pipeline: $DESC_PIPELINE"
echo ""
echo "To use these pipelines during indexing:"
echo "  - Patents: ?pipeline=$PATENTS_PIPELINE"
echo "  - Claims: ?pipeline=$CLAIMS_PIPELINE"
echo "  - Descriptions: ?pipeline=$DESC_PIPELINE"