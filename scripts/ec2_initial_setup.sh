#!/bin/bash
# EC2 Initial Setup Script
# Run this ONCE after cloning the repo on EC2

set -e

echo "=== IISc Chatbot - EC2 Initial Setup ==="
echo ""

# Configuration
S3_BUCKET="iisc-chatbot-data-mumbai"
S3_REGION="ap-south-1"

# Check if running on EC2
echo "Step 1: Checking environment..."
if [ ! -d /opt/iisc-chatbot ]; then
    echo "ERROR: Please run this from /opt/iisc-chatbot"
    exit 1
fi

cd /opt/iisc-chatbot

# Check if AWS CLI is installed
echo "Step 2: Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    sudo apt-get update
    sudo apt-get install -y awscli
fi

# Configure AWS (if not already configured)
echo "Step 3: Configuring AWS..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "AWS credentials not configured!"
    echo "Please run: aws configure"
    echo "Then run this script again."
    exit 1
fi

# Download data from S3
echo "Step 4: Downloading data from S3..."
echo "Bucket: s3://$S3_BUCKET/"
echo "Size: ~12 GB (this will take 5-10 minutes)"
echo ""

if [ -d "data/index" ] && [ "$(ls -A data/index)" ]; then
    read -p "Data directory exists. Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping data download."
    else
        echo "Syncing data from S3..."
        aws s3 sync s3://$S3_BUCKET/ data/ --region $S3_REGION
        echo "✓ Data synced from S3"
    fi
else
    echo "Downloading data from S3..."
    mkdir -p data
    aws s3 sync s3://$S3_BUCKET/ data/ --region $S3_REGION
    echo "✓ Data downloaded from S3"
fi

# Verify data
echo ""
echo "Step 5: Verifying data..."
REQUIRED_INDEX="data/index/fastembed_bge_small_iisc_2025nov07"
if [ -d "$REQUIRED_INDEX" ]; then
    echo "✓ Index found: $REQUIRED_INDEX"
    INDEX_SIZE=$(du -sh "$REQUIRED_INDEX" | cut -f1)
    echo "  Size: $INDEX_SIZE"
else
    echo "✗ ERROR: Required index not found!"
    echo "  Expected: $REQUIRED_INDEX"
    exit 1
fi

# Check for .env file
echo ""
echo "Step 6: Checking environment configuration..."
if [ -f .env ]; then
    echo "✓ .env file exists"
    
    # Check if GEMINI_API_KEY is set
    if grep -q "GEMINI_API_KEY=your_gemini_api_key_here" .env; then
        echo "⚠️  WARNING: GEMINI_API_KEY not configured!"
        echo "  Please edit .env and add your API key"
    else
        echo "✓ GEMINI_API_KEY appears to be configured"
    fi
else
    echo "⚠️  .env file not found!"
    echo "Creating .env from ENV_PLACEHOLDER..."
    
    if [ -f ENV_PLACEHOLDER ]; then
        cp ENV_PLACEHOLDER .env
        echo "✓ .env file created"
        echo ""
        echo "IMPORTANT: Edit .env and add your GEMINI_API_KEY:"
        echo "  nano .env"
    else
        echo "✗ ERROR: ENV_PLACEHOLDER not found!"
        exit 1
    fi
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Configure .env file (if not done):"
echo "   nano .env"
echo ""
echo "2. Deploy the application:"
echo "   ./scripts/deploy.sh"
echo ""
echo "3. Check deployment status:"
echo "   docker ps"
echo "   docker-compose -f docker-compose.prod.yml logs -f"
