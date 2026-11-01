"""
NVIDIA RAG Pipeline with Nemotron-3.5-Embed-1B.

Handles:
- Document loading & chunking
- Dense vector embedding using nvidia/nemotron-3.5-embed-1b
- Persistent FAISS vector store
- Contextual retrieval and LLM generation via ChatNVIDIA
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import config


class MissingAPIKeyError(Exception):
    """Raised when NVIDIA_API_KEY is not configured."""
    pass


class NVIDIARagPipeline:
    """
    Complete RAG Pipeline powered by NVIDIA Nemotron-3.5-Embed-1B
    and NVIDIA NIM Chat Models.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or config.nvidia_api_key).strip()
        self.embedding_model_name = config.embedding_model
        self.llm_model_name = config.llm_model
        self.index_dir = config.index_dir
        self.vector_store: Optional[FAISS] = None

        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        # Initialize Embedder & LLM lazily
        self._embedder = None
        self._llm = None

    def _ensure_api_key(self):
        """Validates that a usable API key is present."""
        if not self.api_key or self.api_key.startswith("nvapi-your"):
            raise MissingAPIKeyError(
                "NVIDIA_API_KEY is not configured or is still using the placeholder!\n"
                "Please get your free API key at https://build.nvidia.com\n"
                "and set it in .env (NVIDIA_API_KEY=nvapi-...) or pass it to NVIDIARagPipeline."
            )

    @property
    def embedder(self):
        """NVIDIAEmbeddings instance for nemotron-3.5-embed-1b."""
        if self._embedder is None:
            self._ensure_api_key()
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            self._embedder = NVIDIAEmbeddings(
                model=self.embedding_model_name,
                nvidia_api_key=self.api_key,
                # truncate="END",
            )
        return self._embedder

    @property
    def llm(self):
        """ChatNVIDIA instance for response generation."""
        if self._llm is None:
            self._ensure_api_key()
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            self._llm = ChatNVIDIA(
                model=self.llm_model_name,
                nvidia_api_key=self.api_key,
                temperature=0.2,
                max_tokens=3072,
            )
        return self._llm

    def load_file(self, file_path: Path) -> List[Document]:
        """Loads a single file (PDF, MD, TXT, JSON, CSV)."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        elif ext in [".txt", ".md", ".json", ".csv", ".py", ".yaml", ".yml"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
        else:
            # Fallback to UTF-8 text loader
            loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()

        # Enhance metadata
        for i, doc in enumerate(docs):
            doc.metadata["source_file"] = file_path.name
            doc.metadata["source_path"] = str(file_path)
            doc.metadata["page_or_section"] = doc.metadata.get("page", i + 1)
        return docs

    def load_directory(self, dir_path: Path) -> List[Document]:
        """Loads all supported documents from a directory."""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        supported_extensions = [".pdf", ".md", ".txt", ".json", ".csv"]
        all_docs = []
        for ext in supported_extensions:
            for file_path in dir_path.glob(f"*{ext}"):
                try:
                    all_docs.extend(self.load_file(file_path))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path.name}: {e}")
        return all_docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits raw documents into semantic chunks with metadata preservation."""
        chunks = self.text_splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
        return chunks

    def ingest_documents(
        self,
        docs_or_chunks: List[Document],
        persist: bool = True
    ) -> Tuple[int, int]:
        """
        Embeds document chunks using Nemotron-3.5-Embed-1B (passage mode)
        and indexes them into the FAISS vector database.
        Returns: (num_documents, num_chunks)
        """
        if not docs_or_chunks:
            return 0, 0

        # If documents haven't been split yet, split them
        if any(len(d.page_content) > config.chunk_size * 1.5 for d in docs_or_chunks):
            chunks = self.chunk_documents(docs_or_chunks)
        else:
            chunks = docs_or_chunks

        # Initialize or add to FAISS
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embedder)
        else:
            self.vector_store.add_documents(chunks)

        if persist:
            self.save_index()

        return len(docs_or_chunks), len(chunks)

    def save_index(self, path: Optional[Path] = None):
        """Saves current FAISS index to disk."""
        target_path = Path(path or self.index_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        if self.vector_store is not None:
            self.vector_store.save_local(str(target_path))

    def load_index(self, path: Optional[Path] = None) -> bool:
        """Loads FAISS index from disk if present."""
        target_path = Path(path or self.index_dir)
        index_file = target_path / "index.faiss"
        if not index_file.exists():
            return False

        self.vector_store = FAISS.load_local(
            str(target_path),
            self.embedder,
            allow_dangerous_deserialization=True,
        )
        return True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Performs dense semantic retrieval using Nemotron-3.5-Embed-1B (query mode).
        Returns list of (Document, score) pairs.
        """
        if self.vector_store is None:
            # Try loading saved index
            loaded = self.load_index()
            if not loaded or self.vector_store is None:
                raise ValueError("Vector store is empty! Please ingest documents first.")

        k = top_k or config.top_k
        # FAISS similarity_search_with_score returns L2 distance (lower is closer)
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG inference:
        1. Retrieve relevant passages using Nemotron-3.5-Embed-1B
        2. Format context with source citations
        3. Generate grounded response using NVIDIA Chat Model
        """
        retrieved = self.retrieve(query, top_k=top_k)

        # Build formatted context
        context_parts = []
        sources = []
        for idx, (doc, score) in enumerate(retrieved, start=1):
            source_file = doc.metadata.get("source_file", "unknown")
            page_sec = doc.metadata.get("page_or_section", "1")
            chunk_id = doc.metadata.get("chunk_id", idx)

            context_parts.append(
                f"[Source {idx} | File: {source_file} | Chunk: {chunk_id}]\n{doc.page_content.strip()}"
            )
            sources.append({
                "source_file": source_file,
                "chunk_id": chunk_id,
                "page": page_sec,
                "score": float(score),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })

        formatted_context = "\n\n---\n\n".join(context_parts)

        # Prompt template: ChatGPT-style clean paragraph & structured bullet points
        system_prompt = (
            "You are a helpful, expert AI assistant. Answer the user's question clearly, "
            "authoritatively, and elegantly, strictly grounded in the provided context.\n\n"
            "Formatting & Tone Guidelines (ChatGPT-style):\n"
            "1. Lead with a clear, concise introductory paragraph directly answering the question. "
            "Never start with robotic filler like 'Based on the provided context...' or 'According to Source 1...'.\n"
            "2. Break down the key details into clean, structured bullet points with bold key headings "
            "(e.g., • **Key Achievement**: Detailed explanation).\n"
            "3. If applicable, conclude with a brief, helpful summary sentence.\n"
            "4. Do NOT output robotic negative meta-statements (e.g., avoid 'No other facts were mentioned in the remaining sources').\n"
            "5. Cite references unobtrusively using [Source X] at the end of relevant points.\n"
            "6. If the context does not contain enough information, explain what is known and state what is not covered naturally.\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": formatted_context, "question": query})

        # Ensure answer is string and not empty
        if not answer or not str(answer).strip():
            raw_msg = self.llm.invoke(prompt.format_messages(context=formatted_context, question=query))
            if hasattr(raw_msg, "content") and raw_msg.content and str(raw_msg.content).strip():
                answer = str(raw_msg.content).strip()
            elif hasattr(raw_msg, "additional_kwargs") and raw_msg.additional_kwargs.get("reasoning_content"):
                answer = str(raw_msg.additional_kwargs.get("reasoning_content")).strip()

        return {
            "query": query,
            "answer": str(answer).strip(),
            "sources": sources,
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
        }

    def get_index_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current vector index."""
        if self.vector_store is None:
            self.load_index()

        if self.vector_store is None:
            return {
                "indexed": False,
                "total_vectors": 0,
                "embedding_model": self.embedding_model_name,
                "index_dir": str(self.index_dir)
            }

        faiss_index = self.vector_store.index
        docstore = self.vector_store.docstore._dict

        # Collect source files
        source_files = set()
        for doc in docstore.values():
            if hasattr(doc, "metadata") and "source_file" in doc.metadata:
                source_files.add(doc.metadata["source_file"])

        return {
            "indexed": True,
            "total_vectors": faiss_index.ntotal,
            "dimensions": faiss_index.d,
            "source_files": list(source_files),
            "embedding_model": self.embedding_model_name,
            "index_dir": str(self.index_dir),
        }
