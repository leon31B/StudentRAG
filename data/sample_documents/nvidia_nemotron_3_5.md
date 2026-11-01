# NVIDIA Nemotron-3.5-Embed-1B Overview

## Introduction
`nvidia/nemotron-3.5-embed-1b` (Nemotron 3.5 Embed 1B) is a state-of-the-art multimodal and multilingual embedding model developed by NVIDIA as part of the Nemotron Retriever family. With approximately 1.14 billion parameters, it delivers an optimal balance of high throughput, low latency, and leading retrieval accuracy for modern Retrieval-Augmented Generation (RAG) systems.

## Key Capabilities
- **Multimodal Retrieval**: Embeds text, tabular data, infographics, charts, and document screenshots into a unified vector space.
- **Asymmetric Retrieval Modes**:
  - `passage`: Used for embedding documents, passages, tables, and document screenshots into the vector index.
  - `query`: Used for embedding user questions or search queries.
- **Context Window**: Supports long contexts up to 32,768 tokens, allowing large continuous sections of documents or codebases to be represented without aggressive fragmentation.
- **Matryoshka Representations (MRL)**: Outputs dense 2048-dimensional embeddings by default, which can be truncated to 1024 or 512 dimensions while retaining high semantic fidelity.
- **Multilingual Support**: Trained across dozens of human languages, enabling cross-lingual semantic search and question answering.

## Deployment Options
Nemotron-3.5-Embed-1B can be deployed via:
1. **NVIDIA Cloud AI Endpoints (NIM)**: Serverless hosted API at `integrate.api.nvidia.com`, integrated with `langchain-nvidia-ai-endpoints`.
2. **NVIDIA NeMo & NIM Self-Hosted Microservices**: Containerized deployment on Kubernetes or local NVIDIA RTX / DGX GPUs.
3. **Hugging Face / Open Source**: Weights available as `nvidia/Nemotron-3.5-Embed-1B-BF16` and quantized `NVFP4`.
