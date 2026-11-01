"""
Interactive Command-Line Interface for NVIDIA Nemotron-3.5-Embed-1B RAG.
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
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from config import config
from rag_pipeline import NVIDIARagPipeline, MissingAPIKeyError

console = Console()


def print_banner():
    console.print(Panel(
        "[bold green]NVIDIA Nemotron-3.5-Embed-1B[/bold green] [bold white]RAG Terminal[/bold white]\n"
        f"[cyan]Embedding Model:[/cyan] [bold white]{config.embedding_model}[/bold white] (2048-dim MRL)\n"
        f"[cyan]LLM Generator:[/cyan]   [white]{config.llm_model}[/white]\n"
        f"[cyan]Vector Store:[/cyan]    [white]FAISS (Dense Semantic Index)[/white]",
        border_style="green",
        expand=False
    ))


def check_api_key(pipeline: NVIDIARagPipeline) -> bool:
    if not config.is_api_key_valid():
        console.print(Panel(
            "[bold yellow]NVIDIA_API_KEY is not set in .env[/bold yellow]\n\n"
            "To use NVIDIA cloud models (Nemotron-3.5-Embed-1B & LLaMA 3.3 70B):\n"
            "1. Visit [cyan]https://build.nvidia.com[/cyan] (free with 1000 credits)\n"
            "2. Generate an API Key (nvapi-...)\n"
            "3. Paste into [bold].env[/bold] file in this directory.",
            border_style="yellow"
        ))
        return False
    return True


def cmd_ingest_sample(pipeline: NVIDIARagPipeline):
    console.print(f"\n[cyan]Ingesting sample documents from:[/cyan] {config.sample_docs_dir}")
    try:
        docs = pipeline.load_directory(config.sample_docs_dir)
        num_docs, num_chunks = pipeline.ingest_documents(docs, persist=True)
        console.print(f"[bold green][OK] Successfully ingested {num_docs} documents into {num_chunks} vector chunks![/bold green]")
    except Exception as e:
        console.print(f"[bold red][X] Ingestion failed:[/bold red] {e}")


def cmd_ingest_custom(pipeline: NVIDIARagPipeline):
    path_str = Prompt.ask("\n[cyan]Enter file or directory path to ingest[/cyan]").strip('"')
    p = Path(path_str)
    if not p.exists():
        console.print(f"[red]Error: Path {p} does not exist.[/red]")
        return

    try:
        if p.is_dir():
            docs = pipeline.load_directory(p)
        else:
            docs = pipeline.load_file(p)

        num_docs, num_chunks = pipeline.ingest_documents(docs, persist=True)
        console.print(f"[bold green][OK] Successfully ingested {num_docs} files into {num_chunks} vector chunks![/bold green]")
    except Exception as e:
        console.print(f"[bold red][X] Ingestion failed:[/bold red] {e}")


def cmd_query_rag(pipeline: NVIDIARagPipeline):
    stats = pipeline.get_index_stats()
    if not stats.get("indexed") or stats.get("total_vectors", 0) == 0:
        console.print("[yellow]Index is empty. Ingesting sample documents first...[/yellow]")
        cmd_ingest_sample(pipeline)

    console.print("\n[bold cyan]=== NVIDIA RAG Interactive Q&A Mode ===[/bold cyan]")
    console.print("[dim]Type your question, or 'exit' / 'back' to return to menu.[/dim]\n")

    while True:
        question = Prompt.ask("[bold green]Ask RAG[/bold green]").strip()
        if question.lower() in ["exit", "back", "quit", "q"]:
            break
        if not question:
            continue

        console.print(f"\n[dim]Searching index with {config.embedding_model}...[/dim]")
        try:
            result = pipeline.answer_query(question)

            # Display Answer
            console.print(Panel(
                Markdown(result["answer"]),
                title="[bold green]Answer[/bold green]",
                border_style="green"
            ))

            # Display Sources
            sources_table = Table(title="Retrieved Citations", border_style="cyan")
            sources_table.add_column("Source #", style="bold cyan", width=10)
            sources_table.add_column("File", style="yellow")
            sources_table.add_column("L2 Distance", style="green", width=12)
            sources_table.add_column("Snippet Preview")

            for idx, s in enumerate(result["sources"], 1):
                sources_table.add_row(
                    f"Source {idx}",
                    s["source_file"],
                    f"{s['score']:.4f}",
                    s["snippet"].replace("\n", " ")[:120] + "..."
                )
            console.print(sources_table)
            console.print("")

        except Exception as e:
            console.print(f"[bold red][X] Query failed:[/bold red] {e}\n")


def cmd_test_embedding(pipeline: NVIDIARagPipeline):
    text = Prompt.ask("\n[cyan]Enter sample text to embed with Nemotron-3.5-Embed-1B[/cyan]", default="NVIDIA AI Enterprise")
    console.print(f"[dim]Generating embedding using {config.embedding_model}...[/dim]")
    try:
        vec = pipeline.embedder.embed_query(text)
        console.print(f"[bold green][OK] Embedding successfully generated![/bold green]")
        console.print(f"[cyan]Dimensions:[/cyan] [bold white]{len(vec)}[/bold white]")
        console.print(f"[cyan]First 8 vector values:[/cyan] {vec[:8]}")
    except Exception as e:
        console.print(f"[bold red][X] Embedding test failed:[/bold red] {e}")


def cmd_show_stats(pipeline: NVIDIARagPipeline):
    stats = pipeline.get_index_stats()
    table = Table(title="FAISS Index Statistics", border_style="green")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Status", "[green]Ready[/green]" if stats.get("indexed") else "[yellow]Empty[/yellow]")
    table.add_row("Total Vectors", str(stats.get("total_vectors", 0)))
    table.add_row("Embedding Model", str(stats.get("embedding_model", "N/A")))
    table.add_row("Vector Dimensions", str(stats.get("dimensions", "N/A")))
    table.add_row("Index Directory", str(stats.get("index_dir", "N/A")))
    table.add_row("Indexed Files", ", ".join(stats.get("source_files", [])) or "None")

    console.print(table)


def main():
    print_banner()
    pipeline = NVIDIARagPipeline()

    while True:
        console.print("\n[bold]Select an action:[/bold]")
        console.print("  [bold green]1.[/bold green] Ingest Sample Knowledge Base (2 documents)")
        console.print("  [bold green]2.[/bold green] Ingest Custom File or Folder (PDF, MD, TXT, etc.)")
        console.print("  [bold green]3.[/bold green] Query RAG (Interactive Q&A)")
        console.print("  [bold green]4.[/bold green] Test Nemotron-3.5-Embed-1B Direct Embedding")
        console.print("  [bold green]5.[/bold green] View Vector Index Statistics")
        console.print("  [bold green]6.[/bold green] Exit")

        choice = Prompt.ask("\n[bold yellow]Option[/bold yellow]", choices=["1", "2", "3", "4", "5", "6"], default="3")

        if choice == "1":
            if check_api_key(pipeline):
                cmd_ingest_sample(pipeline)
        elif choice == "2":
            if check_api_key(pipeline):
                cmd_ingest_custom(pipeline)
        elif choice == "3":
            if check_api_key(pipeline):
                cmd_query_rag(pipeline)
        elif choice == "4":
            if check_api_key(pipeline):
                cmd_test_embedding(pipeline)
        elif choice == "5":
            cmd_show_stats(pipeline)
        elif choice == "6":
            console.print("[green]Goodbye![/green]")
            break


if __name__ == "__main__":
    main()
