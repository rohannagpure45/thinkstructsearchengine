#!/bin/bash

echo "==================================="
echo "Hybrid Patent Search - Quick Start"
echo "==================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp env.example .env
    echo "✓ Created .env file"
    echo ""
fi

# Start Elasticsearch
echo "Starting Elasticsearch with Docker Compose..."
docker-compose up -d

# Wait for Elasticsearch to be healthy
echo "Waiting for Elasticsearch to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:9200/_cluster/health | grep -q '"status":"green"\|"status":"yellow"'; then
        echo "✓ Elasticsearch is ready!"
        break
    fi
    echo -n "."
    sleep 2
done
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

# Setup ML models
echo "Setting up ML models..."
python scripts/setup_ml_models.py
echo ""

# Create indices
echo "Creating Elasticsearch indices..."
./scripts/create_indices.sh
echo ""

# Register ELSER pipeline
echo "Registering ELSER ingest pipelines..."
./scripts/register_elser_pipeline.sh
echo ""

# Ingest sample data
echo "Ingesting sample patent data..."
python scripts/ingest_data.py --input data/sample_patents.json
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
