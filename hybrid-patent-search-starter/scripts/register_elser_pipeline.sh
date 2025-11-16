#!/usr/bin/env bash
set -euo pipefail

# Load environment variables
ES_HOST=${ES_HOST:-"http://localhost:9200"}
ES_USER=${ES_USERNAME:-"elastic"}
ES_PASS=${ES_PASSWORD:-"changeme"}
ELSER_MODEL_ID=${ELSER_MODEL_ID:-".elser_model_2"}

# Pipeline names
BASE_PIPELINE=${ELSER_PIPELINE_NAME:-"elser_patents_pipeline"}
PATENTS_PIPELINE="$BASE_PIPELINE"
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
  curl -s -u "$ES_USER:$ES_PASS" -X DELETE "$ES_HOST/_ingest/pipeline/$pipeline_name" 2>/dev/null || true
  
  # Create pipeline
  response=$(curl -s -u "$ES_USER:$ES_PASS" -X PUT "$ES_HOST/_ingest/pipeline/$pipeline_name" \
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
        "field_map": {
          "title": "text_field",
          "abstract": "text_field"
        },
        "target_field": "_ml_inference",
        "on_failure": [
          {
            "set": {
              "field": "_ingest_error",
              "value": "ELSER inference failed: {{_ingest.on_failure_message}}"
            }
          }
        ]
      }
    },
    {
      "script": {
        "description": "Extract ELSER tokens to separate sparse vector fields",
        "lang": "painless",
        "source": """
          if (ctx._ml_inference?.title != null) {
            ctx.title_elser = ctx._ml_inference.title.predicted_value;
          }
          if (ctx._ml_inference?.abstract != null) {
            ctx.abstract_elser = ctx._ml_inference.abstract.predicted_value;
          }
          ctx.remove('_ml_inference');
        """
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
        "field_map": {
          "claim_text": "text_field"
        },
        "target_field": "_ml_inference",
        "if": "ctx.claim_text != null && ctx.claim_text.length() > 0",
        "on_failure": [
          {
            "set": {
              "field": "_ingest_error",
              "value": "ELSER inference failed: {{_ingest.on_failure_message}}"
            }
          }
        ]
      }
    },
    {
      "script": {
        "description": "Extract ELSER tokens to sparse vector field",
        "lang": "painless",
        "if": "ctx._ml_inference?.claim_text != null",
        "source": """
          ctx.claim_elser = ctx._ml_inference.claim_text.predicted_value;
          ctx.remove('_ml_inference');
        """
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
        "field_map": {
          "desc_text": "text_field"
        },
        "target_field": "_ml_inference",
        "if": "ctx.desc_text != null && ctx.desc_text.length() > 0",
        "on_failure": [
          {
            "set": {
              "field": "_ingest_error",
              "value": "ELSER inference failed: {{_ingest.on_failure_message}}"
            }
          }
        ]
      }
    },
    {
      "script": {
        "description": "Extract ELSER tokens to sparse vector field",
        "lang": "painless",
        "if": "ctx._ml_inference?.desc_text != null",
        "source": """
          ctx.desc_elser = ctx._ml_inference.desc_text.predicted_value;
          ctx.remove('_ml_inference');
        """
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