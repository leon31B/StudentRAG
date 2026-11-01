"""
Configuration module for the NVIDIA RAG Environment.
Loads configuration from environment variables or .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Locate base directory and load .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class RAGConfig(BaseSettings):
    """RAG Environment Settings"""

    # NVIDIA API Key (obtain from https://build.nvidia.com)
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")

    # Embedding model specifically requested
    embedding_model: str = os.getenv(
        "NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b"
    )

    # LLM for generation (active in NVIDIA NIM)
    llm_model: str = os.getenv(
        "NVIDIA_LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
    )

    # Vector store path
    index_dir: Path = BASE_DIR / os.getenv("INDEX_DIR", "data/faiss_index")
    sample_docs_dir: Path = BASE_DIR / "data" / "sample_documents"

    # Ingestion & Retrieval parameters
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Dimension for nemotron-3.5-embed-1b (supports 2048, 1024, 512)
    embedding_dim: int = 2048

    def is_api_key_valid(self) -> bool:
        """Check if NVIDIA API key is configured."""
        key = self.nvidia_api_key.strip()
        return bool(key) and not key.startswith("nvapi-your")


config = RAGConfig()
