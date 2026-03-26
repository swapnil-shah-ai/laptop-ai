"""
Laptop AI - Phase 1: Scan & Index
Crawls your laptop, chunks every document, embeds locally via Ollama,
stores in ChromaDB. Run once to index, re-run to update.

Usage:
    python scan.py              # Full scan
    python scan.py --reset      # Wipe DB and re-scan from scratch
"""

import os
import sys
import time
import yaml
import argparse
import shutil
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

import chromadb
from chromadb.config import Settings

from crawler import discover_files
from chunker import extract_and_chunk

console = Console()

# ---- Ollama embedding wrapper ----

def get_embedding_function(model_name: str):
    """Create Ollama embedding function for ChromaDB."""
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    return OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name=model_name,
    )


def check_ollama_ready(model_name: str) -> bool:
    """Check if Ollama is running and the model is available."""
    import urllib.request
    import json

    try:
        # Check if Ollama is running
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            
            # Check for model (with or without :latest tag)
            model_found = any(
                m == model_name or m.startswith(f"{model_name}:")
                for m in models
            )
            
            if not model_found:
                console.print(f"\n[yellow]⚠ Model '{model_name}' not found in Ollama.[/yellow]")
                console.print(f"  Run: [bold]ollama pull {model_name}[/bold]")
                return False
            
            return True

    except Exception:
        console.print("\n[red]✗ Ollama is not running.[/red]")
        console.print("  Start it: [bold]ollama serve[/bold] (in a separate terminal)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Laptop AI - Scan & Index")
    parser.add_argument("--reset", action="store_true", help="Wipe DB and re-scan from scratch")
    args = parser.parse_args()

    # ---- Banner ----
    console.print(Panel(
        "[bold cyan]LAPTOP AI[/bold cyan] — Phase 1: RAG\n"
        "[dim]Scan. Chunk. Embed. Query. All local.[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # ---- Load config ----
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        console.print("[red]✗ config.yaml not found. Copy config.yaml.example to config.yaml and edit.[/red]")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ---- Check Ollama ----
    embed_model = config.get("embedding_model", "nomic-embed-text")
    console.print(f"\n[bold]Checking Ollama...[/bold]")
    if not check_ollama_ready(embed_model):
        sys.exit(1)
    console.print(f"  [green]✓ Ollama running, {embed_model} ready[/green]")

    # ---- Reset if requested ----
    db_path = Path(config.get("db_path", "./chroma_db"))
    if args.reset and db_path.exists():
        console.print(f"\n[yellow]Wiping existing database at {db_path}...[/yellow]")
        shutil.rmtree(db_path)
        console.print("  [green]✓ Database cleared[/green]")

    # ---- Step 1: Crawl ----
    console.print("\n" + "=" * 60)
    console.print("[bold]STEP 1: CRAWL[/bold] — Finding files on your laptop")
    console.print("=" * 60)

    files = discover_files(config)
    if not files:
        console.print("[red]No files found. Check your scan_paths in config.yaml[/red]")
        sys.exit(1)

    # ---- Step 2: Chunk ----
    console.print("\n" + "=" * 60)
    console.print("[bold]STEP 2: CHUNK[/bold] — Extracting and splitting text")
    console.print("=" * 60 + "\n")

    chunk_size = config.get("chunk_size", 800)
    chunk_overlap = config.get("chunk_overlap", 100)
    
    all_chunks = []
    files_processed = 0
    files_failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking files...", total=len(files))
        
        for file_info in files:
            try:
                chunks = extract_and_chunk(file_info, chunk_size, chunk_overlap)
                if chunks:
                    all_chunks.extend(chunks)
                    files_processed += 1
                else:
                    files_failed += 1
            except Exception:
                files_failed += 1
            
            progress.update(task, advance=1)

    console.print(f"\n  [green]✓ {len(all_chunks):,} chunks from {files_processed:,} files[/green]")
    if files_failed:
        console.print(f"  [dim]  {files_failed} files couldn't be read (binary/corrupt/locked)[/dim]")

    if not all_chunks:
        console.print("[red]No text extracted from any file. Check your files.[/red]")
        sys.exit(1)

    # ---- Step 3: Embed & Store ----
    console.print("\n" + "=" * 60)
    console.print("[bold]STEP 3: EMBED & STORE[/bold] — Creating vectors in ChromaDB")
    console.print("=" * 60 + "\n")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(db_path))
    collection_name = config.get("collection_name", "laptop_files")
    
    embed_fn = get_embedding_function(embed_model)
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch insert (ChromaDB handles dedup by ID)
    batch_size = 50  # keep small to avoid overwhelming Ollama on 16GB
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(all_chunks))
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            
            ids = [f"chunk_{i + j}" for j in range(len(batch))]
            documents = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
            except Exception as e:
                # Batch failed — try chunks individually so one bad chunk doesn't kill the batch
                rescued = 0
                for j in range(len(ids)):
                    try:
                        # Truncate as safety net
                        doc = documents[j][:6000] if len(documents[j]) > 6000 else documents[j]
                        collection.upsert(
                            ids=[ids[j]],
                            documents=[doc],
                            metadatas=[metadatas[j]],
                        )
                        rescued += 1
                    except Exception:
                        pass  # skip this chunk entirely
                if rescued < len(ids):
                    console.print(f"  [yellow]⚠ Batch {i // batch_size + 1}: {rescued}/{len(ids)} chunks saved[/yellow]")
            
            progress.update(task, advance=len(batch))

    # ---- Summary ----
    final_count = collection.count()
    
    console.print("\n" + "=" * 60)
    console.print("[bold green]SCAN COMPLETE[/bold green]")
    console.print("=" * 60)
    
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("Files scanned", f"{len(files):,}")
    summary.add_row("Files indexed", f"{files_processed:,}")
    summary.add_row("Chunks stored", f"{final_count:,}")
    summary.add_row("Database", str(db_path.resolve()))
    summary.add_row("Embedding model", embed_model)
    console.print(summary)

    console.print(f"\n[bold cyan]Next: run [white]python query.py[/white] to ask questions about your files.[/bold cyan]\n")


if __name__ == "__main__":
    main()
