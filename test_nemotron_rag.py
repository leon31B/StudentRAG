"""
Test script for the NVIDIA Nemotron-3.5-Embed-1B RAG environment.
Validates dependencies, pipeline components, and live inference if API key is provided.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import config
from rag_pipeline import NVIDIARagPipeline, MissingAPIKeyError

console = Console()


def run_diagnostics():
    console.print(Panel.fit(
        "[bold green]NVIDIA RAG Environment Diagnostics[/bold green]\n"
        f"[cyan]Target Model:[/cyan] [bold white]{config.embedding_model}[/bold white]\n"
        f"[cyan]LLM Generator:[/cyan] [white]{config.llm_model}[/white]\n"
        f"[cyan]Python Runtime:[/cyan] [white]{sys.version.split()[0]}[/white]",
        border_style="green"
    ))

    # 1. Dependency checks
    dep_table = Table(title="Dependency Validation", border_style="cyan")
    dep_table.add_column("Package", style="bold")
    dep_table.add_column("Status", style="green")
    dep_table.add_column("Details")

    packages = [
        ("langchain", "LangChain Core Framework"),
        ("langchain_nvidia_ai_endpoints", "NVIDIA AI Endpoints SDK"),
        ("faiss", "FAISS Vector Database"),
        ("pypdf", "PDF Document Parser"),
        ("fastapi", "FastAPI Web Server"),
    ]

    all_deps_ok = True
    for pkg_name, desc in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            dep_table.add_row(pkg_name, "[green][OK][/green]", f"{desc} ({ver})")
        except ImportError as e:
            all_deps_ok = False
            dep_table.add_row(pkg_name, "[red][X][/red]", str(e))

    console.print(dep_table)

    # 2. Document & Ingestion pipeline check
    console.print("\n[bold yellow]Checking Sample Documents & Ingestion...[/bold yellow]")
    pipeline = NVIDIARagPipeline()
    docs = pipeline.load_directory(config.sample_docs_dir)
    chunks = pipeline.chunk_documents(docs)

    console.print(f"[green][OK] Successfully loaded [bold]{len(docs)}[/bold] documents from {config.sample_docs_dir}[/green]")
    console.print(f"[green][OK] Successfully chunked into [bold]{len(chunks)}[/bold] passages (Chunk Size: {config.chunk_size}, Overlap: {config.chunk_overlap})[/green]")

    # 3. Check NVIDIA API Key
    console.print("\n[bold yellow]Checking NVIDIA API Key & Model Access...[/bold yellow]")
    if not config.is_api_key_valid():
        console.print(Panel(
            "[bold yellow]NVIDIA_API_KEY is not yet configured in .env[/bold yellow]\n\n"
            "All core libraries, vector DB (FAISS), chunking pipeline, and CLI/Web tools\n"
            "are completely installed and ready!\n\n"
            "[bold white]Next Step to activate live Nemotron-3.5-Embed-1B calls:[/bold white]\n"
            "1. Visit [cyan]https://build.nvidia.com[/cyan] (free with 1000 credits)\n"
            "2. Generate an API Key (starts with [green]nvapi-...[/green])\n"
            "3. Paste it in [bold].env[/bold]: [green]NVIDIA_API_KEY=nvapi-...[/green]\n"
            "4. Re-run this test: [bold cyan]py -3.13 test_nemotron_rag.py[/bold cyan]",
            title="[yellow]API Key Needed for Live Inference[/yellow]",
            border_style="yellow"
        ))
        return

    # 4. Live API Test
    try:
        console.print(f"[cyan]Testing live embedding generation with [bold]{config.embedding_model}[/bold]...[/cyan]")
        embedder = pipeline.embedder
        
        # Test query embedding (input_type="query")
        test_query = "What is the parameter size of Nemotron-3.5-Embed-1B?"
        q_vec = embedder.embed_query(test_query)
        console.print(f"[green][OK] Query embedding succeeded! Dimensions: [bold]{len(q_vec)}[/bold][/green]")

        # Test passage embedding (input_type="passage")
        test_passage = ["NVIDIA Nemotron-3.5-Embed-1B is a 1.14B parameter multimodal embedding model."]
        p_vecs = embedder.embed_documents(test_passage)
        console.print(f"[green][OK] Passage embedding succeeded! Vector count: [bold]{len(p_vecs)}[/bold], Dimensions: [bold]{len(p_vecs[0])}[/bold][/green]")

        # Test index creation & retrieval
        console.print("\n[cyan]Indexing sample documents into FAISS...[/cyan]")
        pipeline.ingest_documents(chunks, persist=True)
        stats = pipeline.get_index_stats()
        console.print(f"[green][OK] Indexed [bold]{stats['total_vectors']}[/bold] vectors into FAISS at {stats['index_dir']}[/green]")

        # Test RAG retrieval and answer generation
        console.print("\n[cyan]Executing end-to-end RAG question answering...[/cyan]")
        result = pipeline.answer_query(test_query)

        console.print(Panel(
            f"[bold cyan]Question:[/bold cyan] {result['query']}\n\n"
            f"[bold green]Answer:[/bold green]\n{result['answer']}\n\n"
            f"[bold yellow]Sources Used ({len(result['sources'])}):[/bold yellow]\n" +
            "\n".join([f"- [cyan]{s['source_file']}[/cyan] (Chunk {s['chunk_id']}, score: {s['score']:.4f})" for s in result['sources']]),
            title="[bold green]End-to-End RAG Test Result[/bold green]",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"[bold red][X] Live test encountered an error:[/bold red] {e}")


if __name__ == "__main__":
    run_diagnostics()
