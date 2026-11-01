# Production Dockerfile for StudentRAG / NVIDIA Nemotron RAG Studio
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and grant permissions
COPY --chown=user:user . /app

# Ensure data directories exist and are writable by non-root user
RUN mkdir -p /app/data/faiss_index /app/data/sample_documents && chown -R user:user /app

# Switch to non-root user
USER user

# Expose default port (7860 for Hugging Face Spaces, overridable by $PORT)
EXPOSE 7860

# Run uvicorn server
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
