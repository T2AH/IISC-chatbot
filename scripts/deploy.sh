#!/bin/bash
# Deploy IISc Chatbot to AWS EC2
# Run this from the application directory

set -e

APP_DIR="/opt/iisc-chatbot"
COMPOSE_FILE="docker-compose.prod.yml"

echo "=== Deploying IISc Chatbot ==="

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create .env from .env.example and add your GEMINI_API_KEY"
    exit 1
fi

# Pull latest changes (if using git)
if [ -d .git ]; then
    echo "Pulling latest changes..."
    git pull origin main
fi

# Stop existing containers
echo "Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down

# Build and start containers
echo "Building and starting containers..."
docker-compose -f $COMPOSE_FILE build --no-cache
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo "Waiting for services to start..."
sleep 10

# Check service health
echo "Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ API is healthy"
else
    echo "✗ API health check failed"
    docker-compose -f $COMPOSE_FILE logs api
    exit 1
fi

if curl -f http://localhost:8501 > /dev/null 2>&1; then
    echo "✓ UI is healthy"
else
    echo "✗ UI health check failed"
    docker-compose -f $COMPOSE_FILE logs chatbot
    exit 1
fi

echo "=== Deployment successful! ==="
echo "API: http://localhost:8000"
echo "UI: http://localhost:8501"
echo ""
echo "View logs with: docker-compose -f $COMPOSE_FILE logs -f"
