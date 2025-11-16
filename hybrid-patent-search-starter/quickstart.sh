#!/bin/bash

set -euo pipefail
trap 'echo "Error: Setup failed at line $LINENO"; exit 1' ERR

# Guard helper functions
fail() {
    echo "Error: $1" >&2
    exit 1
}

ensure_file() {
    if [ ! -f "$1" ]; then
        fail "Required file not found: $1"
    fi
}

ensure_executable() {
    if [ ! -f "$1" ]; then
        fail "Required script not found: $1"
    fi
    if [ ! -x "$1" ]; then
        echo "Making $1 executable..."
        if ! chmod +x "$1"; then
            fail "Failed to make $1 executable"
        fi
    fi
}

run_or_exit() {
    if ! "$@"; then
        fail "Command failed: $*"
    fi
}

echo "==================================="
echo "Hybrid Patent Search - Quick Start"
echo "==================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    if [ ! -f env.example ]; then
        echo "Error: env.example template not found." >&2
        exit 1
    fi
    cp env.example .env
    echo "✓ Created .env file"
    echo ""
fi

# Start Elasticsearch
echo "Starting Elasticsearch with Docker Compose..."
if ! docker-compose up -d; then
    echo "Error: Failed to start Docker Compose." >&2
    exit 1
fi

# Wait for Elasticsearch to be healthy
echo "Waiting for Elasticsearch to be ready..."
ES_READY=0
health_response=""
for i in {1..30}; do
    health_response=$(curl -s --max-time 2 http://localhost:9200/_cluster/health || true)
    if echo "$health_response" | grep -qE '"status":"(green|yellow)"'; then
        echo "✓ Elasticsearch is ready!"
        ES_READY=1
        break
    fi
    echo -n "."
    sleep 2
done
if [ $ES_READY -eq 0 ]; then
    echo ""
    echo "Error: Elasticsearch did not become ready after 60 seconds" >&2
    echo "Last health response:"
    echo "$health_response"
    echo ""
    echo "Docker Compose status:"
    docker-compose ps
    exit 1
fi
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ ! -f "venv/bin/activate" ]; then
    echo "Error: venv/bin/activate not found. Please ensure the virtual environment was created successfully." >&2
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Failed to activate the virtual environment." >&2
    exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Please ensure it exists in the repository root." >&2
    exit 1
fi
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Preflight checks: verify all required scripts and data files exist
echo "Running preflight checks..."
ensure_file "scripts/setup_ml_models.py"
ensure_file "scripts/ingest_data.py"
ensure_executable "scripts/create_indices.sh"
ensure_executable "scripts/register_elser_pipeline.sh"
ensure_file "data/sample_patents.json"
echo "✓ All preflight checks passed"
echo ""

# Setup ML models
echo "Setting up ML models..."
run_or_exit python scripts/setup_ml_models.py
echo ""

# Create indices
echo "Creating Elasticsearch indices..."
run_or_exit ./scripts/create_indices.sh
echo ""

# Register ELSER pipeline
echo "Registering ELSER ingest pipelines..."
run_or_exit ./scripts/register_elser_pipeline.sh
echo ""

# Ingest sample data
echo "Ingesting sample patent data..."
run_or_exit python scripts/ingest_data.py --input data/sample_patents.json
echo ""

echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To start the API server:"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --reload"
echo ""
echo "API will be available at:"
echo "  http://localhost:8000"
echo "  http://localhost:8000/docs (Swagger UI)"
echo ""
echo "To access Kibana:"
echo "  http://localhost:5601"
echo ""
echo "To stop Elasticsearch:"
echo "  docker-compose down"
echo ""
