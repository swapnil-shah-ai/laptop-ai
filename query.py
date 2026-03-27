"""
Laptop AI - Phase 1: Query Interface
Ask questions about YOUR files. Get answers with source citations.
All local — Ollama for LLM, ChromaDB for retrieval.

Usage:
    python query.py                     # Interactive chat
    python query.py --ask "question"    # Single question mode
"""

import sys
import yaml
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

import chromadb
from phase4_predict.logger import log_query

console = Console()


def get_embedding_function(model_name: str):
    """Create Ollama embedding function for ChromaDB."""
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    return OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name=model_name,
    )


def query_ollama(prompt: str, model: str) -> str:
    """Send a prompt to Ollama and get a response."""
    import urllib.request
    import json

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,      # low temp = factual, not creative
            "num_predict": 1024,      # max tokens in response
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "No response from model.")
    except Exception as e:
        return f"Error talking to Ollama: {e}"


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build a RAG prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        # Shorten path for readability
        source_short = Path(source).name
        context_parts.append(f"[Source {i}: {source_short}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant that answers questions based ONLY on the provided context from the user's local files. 

RULES:
- Answer based on the context below. If the context doesn't contain the answer, say "I couldn't find this in your files."
- Cite which source(s) you used by mentioning the filename.
- Be concise and direct.
- If the question is about data in a spreadsheet, present it clearly.

CONTEXT FROM USER'S FILES:
{context}

QUESTION: {question}

ANSWER:"""


def retrieve_and_answer(question: str, collection, config: dict) -> None:
    """Retrieve relevant chunks and generate an answer."""
    top_k = config.get("top_k", 5)
    llm_model = config.get("llm_model", "mistral")

    # Retrieve
    console.print(f"\n  [dim]Searching {collection.count():,} chunks...[/dim]")
    
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    # Build context chunks
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
        })

    # Show sources found
    console.print(f"  [dim]Found {len(chunks)} relevant chunks. Generating answer...[/dim]\n")

    # Generate answer
    prompt = build_prompt(question, chunks)
    answer = query_ollama(prompt, llm_model)

    # Display answer
    console.print(Panel(
        Markdown(answer),
        title="[bold cyan]Answer[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # Display sources
    console.print("\n[bold]Sources:[/bold]")
    seen_sources = set()
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        if source in seen_sources:
            continue
        seen_sources.add(source)
        
        distance = chunk["distance"]
        relevance = max(0, round((1 - distance) * 100, 1))
        
        # Extra context
        extra = []
        if "page" in chunk["metadata"]:
            extra.append(f"page {chunk['metadata']['page']}")
        if "sheet" in chunk["metadata"]:
            extra.append(f"sheet: {chunk['metadata']['sheet']}")
        if "slide" in chunk["metadata"]:
            extra.append(f"slide {chunk['metadata']['slide']}")
        
        extra_str = f" ({', '.join(extra)})" if extra else ""
        
        console.print(f"  [{i}] {source}{extra_str}  [dim]relevance: {relevance}%[/dim]")

# Phase 4: silently log this query for ML prediction
    log_query(question, chunks)

def interactive_mode(collection, config: dict):
    """Interactive chat loop."""
    console.print(Panel(
        "[bold cyan]LAPTOP AI[/bold cyan] — Ask anything about your files\n"
        "[dim]Type your question and press Enter. Type 'quit' or 'exit' to stop.\n"
        "Type 'stats' to see database info. Type 'sources' to list indexed files.[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    console.print(f"  [dim]{collection.count():,} chunks indexed | "
                  f"Model: {config.get('llm_model', 'mistral')} | "
                  f"Top-K: {config.get('top_k', 5)}[/dim]\n")

    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if question.lower() == "stats":
            _show_stats(collection)
            continue

        if question.lower() == "sources":
            _show_sources(collection)
            continue

        retrieve_and_answer(question, collection, config)
        console.print()  # spacing


def _show_stats(collection):
    """Show database statistics."""
    count = collection.count()
    
    # Sample metadata to count file types
    if count > 0:
        sample = collection.get(limit=min(count, 1000), include=["metadatas"])
        file_types = {}
        unique_files = set()
        for meta in sample["metadatas"]:
            ft = meta.get("file_type", "unknown")
            file_types[ft] = file_types.get(ft, 0) + 1
            unique_files.add(meta.get("source", ""))
        
        console.print(f"\n  [bold]Database stats:[/bold]")
        console.print(f"    Total chunks:  {count:,}")
        console.print(f"    Unique files:  {len(unique_files):,}")
        console.print(f"    File types:    {', '.join(f'{k}({v})' for k, v in sorted(file_types.items()))}")
    console.print()


def _show_sources(collection):
    """List all indexed source files."""
    count = collection.count()
    if count == 0:
        console.print("  [yellow]No files indexed yet.[/yellow]")
        return
    
    # Get all metadata (up to 10000)
    sample = collection.get(limit=min(count, 10000), include=["metadatas"])
    sources = {}
    for meta in sample["metadatas"]:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    
    console.print(f"\n  [bold]Indexed files ({len(sources)}):[/bold]")
    for src, chunk_count in sorted(sources.items()):
        console.print(f"    {Path(src).name:40s}  [{chunk_count} chunks]  [dim]{src}[/dim]")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Laptop AI - Query your files")
    parser.add_argument("--ask", type=str, help="Ask a single question (non-interactive)")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        console.print("[red]✗ config.yaml not found.[/red]")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check DB exists
    db_path = Path(config.get("db_path", "./chroma_db"))
    if not db_path.exists():
        console.print("[red]✗ No database found. Run [bold]python scan.py[/bold] first.[/red]")
        sys.exit(1)

    # Connect to ChromaDB
    embed_model = config.get("embedding_model", "nomic-embed-text")
    embed_fn = get_embedding_function(embed_model)
    
    client = chromadb.PersistentClient(path=str(db_path))
    collection_name = config.get("collection_name", "laptop_files")
    
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )
    except Exception:
        console.print("[red]✗ Collection not found. Run [bold]python scan.py[/bold] first.[/red]")
        sys.exit(1)

    if collection.count() == 0:
        console.print("[red]✗ Database is empty. Run [bold]python scan.py[/bold] first.[/red]")
        sys.exit(1)

    # Single question or interactive
    if args.ask:
        retrieve_and_answer(args.ask, collection, config)
    else:
        interactive_mode(collection, config)


if __name__ == "__main__":
    main()
