# Production Dockerfile for StudentRAG / NVIDIA Nemotron RAG Studio
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data directory exists
RUN mkdir -p data/faiss_index data/sample_documents

# Expose default port
EXPOSE 8000

# Run uvicorn server
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
