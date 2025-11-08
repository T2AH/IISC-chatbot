# IISc Chatbot – Cloud Native Deployment Guide

## Final Goal
Deploy a cloud-native chatbot that:
- Runs locally or on any machine (no EC2 required for testing)
- Loads all required data (vector index, metadata) directly from an S3 bucket
- Can be shared with other teams for easy testing and validation
- Uses environment variables for S3 and AWS credentials

## Steps Already Completed

### 1. Prepare S3 Bucket
- Uploaded all required index files (e.g., `vectors.npz`, `metadata.jsonl`) to S3 bucket: `iisc-chatbot-data-mumbai`

### 2. Update Python Requirements
- Added `boto3` to `requirements.txt` for S3 access

### 3. Refactor Code for S3 Data Loading
- Updated `src/vector_store/numpy_store.py`:
  - Now supports loading index files from S3 if `S3_BUCKET` environment variable is set
  - Automatically downloads `vectors.npz` and `metadata.jsonl` from S3 before loading

### 4. Install Dependencies
- Created a fresh Python virtual environment
- Installed all requirements with `pip install -r requirements.txt`

### 5. Verify boto3 Installation
- Confirmed with: `python -c "import boto3; print(boto3.__version__)"`

## Next Steps

### 1. Set AWS Credentials and S3 Environment Variables
Add these to your shell or `.env` file:
```
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=ap-south-1
S3_BUCKET=iisc-chatbot-data-mumbai
S3_INDEX_PREFIX=index/fastembed_bge_small_iisc_2025nov07
```

### 2. Run the Chatbot
- Run your application as usual (API, UI, CLI)
- The code will fetch index files from S3 automatically

### 3. Share with Other Teams
- Give them this repo and guide
- Provide IAM credentials for S3 read-only access
- They can run the chatbot locally and fetch data from S3

## Notes
- No EC2 instance is required for local testing
- EC2 is only needed for public demo/production hosting
- For other loaders (FAISS, HNSW, etc.), similar S3 support can be added if needed

---

**This guide documents the cloud-native deployment and data sharing workflow for the IISc Chatbot project.**
