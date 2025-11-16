#!/usr/bin/env bash
set -euo pipefail

ES_HOST=${ES_HOST:-"http://localhost:9200"}
ES_USER=${ES_USERNAME:-"elastic"}
ES_PASS=${ES_PASSWORD:-"changeme"}
PIPELINE_NAME=${PIPELINE_NAME:-"elser_ingest"}
ELSER_MODEL_ID=${ELSER_MODEL_ID:-".elser_model_2"}

echo "Registering ELSER ingest pipeline [$PIPELINE_NAME] using model [$ELSER_MODEL_ID]"

curl -s -u "$ES_USER:$ES_PASS" -X PUT "$ES_HOST/_ingest/pipeline/$PIPELINE_NAME" \
  -H 'Content-Type: application/json' -d @- <<JSON
{
  "description": "ELSER v2 inference — write tokens to ml.tokens",
  "processors": [
    {
      "inference": {
        "model_id": "$ELSER_MODEL_ID",
        "input_output": [
          {
            "input_field": "claims_text",
            "output_field": "ml.tokens"
          }
        ]
      }
    }
  ]
}
JSON

echo "Done."
