# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    langgraph \
    langchain \
    langchain-openai \
    langchain-core \
    python-dotenv \
    fastapi \
    uvicorn[standard] \
    pydantic

# Copy application code
COPY src/ ./src/
COPY main.py chat.py api.py ./
COPY config.yaml ./

# Copy static files
COPY static/ ./static/

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
