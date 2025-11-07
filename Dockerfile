# Multi-purpose Dockerfile for IISc CDS Chatbot
# Build once, run API or UI via docker-compose or overriding CMD.

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (build tools because some deps compile native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

# Copy application code
COPY . /app

# Default environment (override via compose or env)
ENV INDEX_DIR=/app/data/index/hash_numpy \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    UI_PORT=8501

EXPOSE 8000 8501

# Default command runs API; docker-compose overrides per service
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
