"""
Laptop AI - Phase 3: Generate Questions
Reads chunks from ChromaDB and generates questions that can be answered
from those chunks. These questions will be answered by the teacher model.

Usage:
    python generate_questions.py                    # Generate 100 questions
    python generate_questions.py --count 200        # Generate 200 questions
"""

import json
import random
import yaml
import sys
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

import chromadb

console = Console()


def get_embedding_function(model_name: str):
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    return OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name=model_name,
    )


def ask_ollama(prompt: str, model: str = "mistral") -> str:
    """Ask Ollama to generate a question from a chunk."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,    # some creativity for diverse questions
            "num_predict": 100,    # questions are short
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "").strip()
    except Exception as e:
        return ""


def generate_question_from_chunk(chunk_text: str, source: str) -> str:
    """Use Mistral to generate a question that can be answered from the chunk."""
    prompt = f"""Based on the following text from a document, generate ONE specific question that can be answered using this text. The question should be natural — like something a person would actually ask.

Rules:
- Return ONLY the question, nothing else
- No numbering, no prefix like "Question:"
- Make it specific to the content, not generic
- Keep it under 20 words

Text from "{Path(source).name}":
{chunk_text[:1500]}

Question:"""

    return ask_ollama(prompt)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate questions from indexed chunks")
    parser.add_argument("--count", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--output", type=str, default="distill_questions.jsonl", help="Output file")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]LAPTOP AI[/bold cyan] — Phase 3: Generate Questions\n"
        "[dim]Creating questions from your indexed files for distillation.[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Connect to ChromaDB
    db_path = Path(config.get("db_path", "./chroma_db"))
    if not db_path.exists():
        console.print("[red]✗ No database found. Run python scan.py first.[/red]")
        sys.exit(1)

    embed_model = config.get("embedding_model", "nomic-embed-text")
    embed_fn = get_embedding_function(embed_model)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(
        name=config.get("collection_name", "laptop_files"),
        embedding_function=embed_fn,
    )

    total_chunks = collection.count()
    console.print(f"  Chunks in database: {total_chunks:,}")
    console.print(f"  Questions to generate: {args.count}")

    # Sample random chunks
    sample_size = min(args.count * 2, total_chunks)  # sample more, filter later
    sample = collection.get(
        limit=sample_size,
        include=["documents", "metadatas"],
    )

    # Shuffle and pick diverse chunks
    indices = list(range(len(sample["documents"])))
    random.shuffle(indices)

    questions = []
    seen_sources = {}

    console.print(f"\n[bold]Generating questions using Mistral...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=args.count)

        for idx in indices:
            if len(questions) >= args.count:
                break

            chunk_text = sample["documents"][idx]
            metadata = sample["metadatas"][idx]
            source = metadata.get("source", "unknown")

            # Skip if we already have too many questions from same source
            source_name = Path(source).name
            if seen_sources.get(source_name, 0) >= 3:
                continue

            # Skip very short chunks
            if len(chunk_text.split()) < 30:
                continue

            # Generate question
            question = generate_question_from_chunk(chunk_text, source)

            # Validate — must be a question, not empty, not too long
            if not question:
                continue
            if len(question) < 10 or len(question) > 200:
                continue
            if not question.endswith("?"):
                question = question.rstrip(".") + "?"

            questions.append({
                "question": question,
                "source_chunk": chunk_text[:500],  # keep first 500 chars for reference
                "source_file": source,
            })

            seen_sources[source_name] = seen_sources.get(source_name, 0) + 1
            progress.update(task, advance=1)

    # Save questions
    with open(args.output, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    console.print(f"\n[green]✓ {len(questions)} questions saved to {args.output}[/green]")

    # Show a few examples
    console.print(f"\n[bold]Sample questions:[/bold]")
    for q in questions[:5]:
        console.print(f"  • {q['question']}")
        console.print(f"    [dim]from: {Path(q['source_file']).name}[/dim]")

    console.print(f"\n[dim]Next: run [bold]python distill.py[/bold] to get teacher answers and train the student.[/dim]\n")


if __name__ == "__main__":
    main()
