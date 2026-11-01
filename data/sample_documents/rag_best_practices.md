# Retrieval-Augmented Generation (RAG) Best Practices

## Pipeline Architecture
An enterprise RAG pipeline comprises five core phases:
1. **Document Loading**: Ingesting unstructured sources (PDF, Markdown, HTML, TXT, JSON) and extracting clean semantic text and metadata.
2. **Chunking & Splitting**: Segmenting documents into coherent passages. Recommended configurations for Nemotron-3.5-Embed-1B:
   - Chunk Size: 600 - 1200 characters / tokens depending on content density.
   - Chunk Overlap: 100 - 200 characters to prevent loss of boundary context.
3. **Embedding Generation**: Converting text into dense semantic vectors. For asymmetric models like Nemotron, index vectors must use `input_type="passage"` and user queries must use `input_type="query"`.
4. **Vector Indexing & Retrieval**: Storing vectors in FAISS, Milvus, Qdrant, or pgvector with L2 distance or cosine similarity. Top-k retrieval extracts the most relevant passages.
5. **Contextual Generation**: Feeding retrieved passages alongside the user question into an LLM (such as LLaMA 3.3 70B Instruct or Nemotron 70B) with strict system instructions to base its answer on verified context and cite sources.

## Performance Optimization
- **Index Persistence**: Cache FAISS indices on disk to avoid re-embedding unchanged documents.
- **Batched Ingestion**: Process documents in batches of 16-64 passages to maximize GPU / API throughput.
- **Deduplication**: Filter redundant chunks before vector store ingestion.
