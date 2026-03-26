"""
Laptop AI - Phase 3: Distillation
Teacher model (Mistral 7B) answers questions via RAG.
Student model (TinyLlama) learns from those answers.

Usage:
    python distill.py                                           # Full pipeline
    python distill.py --step teacher                            # Only generate teacher answers
    python distill.py --step student                            # Only train student (if teacher answers exist)
    python distill.py --student-model TinyLlama/TinyLlama-1.1B-Chat-v1.0   # Specify student model
"""

import json
import yaml
import sys
import time
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.table import Table

import chromadb

console = Console()


def get_embedding_function(model_name: str):
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    return OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name=model_name,
    )


def query_ollama(prompt: str, model: str, temperature: float = 0.1) -> str:
    """Query Ollama."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 512,
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
            return data.get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


def build_rag_prompt(question: str, context_chunks: list) -> str:
    """Build RAG prompt with retrieved context."""
    context_parts = []
    for i, (doc, meta) in enumerate(context_chunks, 1):
        source_name = Path(meta.get("source", "unknown")).name
        context_parts.append(f"[Source {i}: {source_name}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant. Answer the question based ONLY on the provided context. Be concise, specific and accurate. If the context doesn't contain the answer, say "I couldn't find this in the provided documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def run_teacher(questions_file: str, output_file: str, config: dict):
    """Run teacher model (Mistral) on questions via RAG."""
    console.print(f"\n[bold]STEP 1: TEACHER ANSWERS[/bold]")
    console.print(f"  Teacher model: mistral (via Ollama)")
    console.print(f"  Method: RAG — retrieve chunks, then answer\n")

    # Load questions
    questions = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    console.print(f"  Questions loaded: {len(questions)}")

    # Connect to ChromaDB
    db_path = Path(config.get("db_path", "./chroma_db"))
    embed_model = config.get("embedding_model", "nomic-embed-text")
    embed_fn = get_embedding_function(embed_model)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(
        name=config.get("collection_name", "laptop_files"),
        embedding_function=embed_fn,
    )

    top_k = config.get("top_k", 5)
    teacher_model = config.get("llm_model", "mistral")
    pairs = []
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Teacher answering...", total=len(questions))

        for q in questions:
            question = q["question"]

            # Retrieve relevant chunks
            try:
                results = collection.query(
                    query_texts=[question],
                    n_results=min(top_k, collection.count()),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                skipped += 1
                progress.update(task, advance=1)
                continue

            if not results["documents"] or not results["documents"][0]:
                skipped += 1
                progress.update(task, advance=1)
                continue

            # Build context
            context_chunks = list(zip(
                results["documents"][0],
                results["metadatas"][0],
            ))

            # Build RAG prompt and get teacher answer
            prompt = build_rag_prompt(question, context_chunks)
            answer = query_ollama(prompt, teacher_model)

            # Skip bad answers
            if not answer or "Error:" in answer or len(answer) < 20:
                skipped += 1
                progress.update(task, advance=1)
                continue

            if "couldn't find" in answer.lower() or "not found" in answer.lower():
                skipped += 1
                progress.update(task, advance=1)
                continue

            pairs.append({
                "question": question,
                "answer": answer,
                "sources": [m.get("source", "") for m in results["metadatas"][0]],
            })

            progress.update(task, advance=1)

    # Save teacher answers
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    console.print(f"\n  [green]✓ {len(pairs)} teacher answers saved to {output_file}[/green]")
    if skipped:
        console.print(f"  [dim]  {skipped} questions skipped (no answer or irrelevant)[/dim]")

    # Show samples
    console.print(f"\n[bold]Sample teacher answers:[/bold]")
    for pair in pairs[:3]:
        console.print(f"\n  [cyan]Q: {pair['question']}[/cyan]")
        console.print(f"  A: {pair['answer'][:200]}...")

    return pairs


def run_student(teacher_file: str, student_model_name: str, output_dir: str, epochs: int = 3, lora_rank: int = 4):
    """Train student model on teacher's answers."""
    console.print(f"\n{'=' * 60}")
    console.print(f"[bold]STEP 2: STUDENT TRAINING[/bold]")
    console.print(f"{'=' * 60}")
    console.print(f"  Student model: {student_model_name}")
    console.print(f"  Learning from: {teacher_file}")

    # Load teacher answers
    pairs = []
    with open(teacher_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    console.print(f"  Training examples: {len(pairs)}")

    if len(pairs) < 5:
        console.print(f"  [red]✗ Too few teacher answers. Need at least 5. Run more questions.[/red]")
        sys.exit(1)

    # Convert to training format
    training_data = []
    for pair in pairs:
        training_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the user's local files. Be concise, specific and accurate. Always cite which document your answer comes from."
                },
                {
                    "role": "user",
                    "content": pair["question"]
                },
                {
                    "role": "assistant",
                    "content": pair["answer"]
                }
            ]
        })

    # Save as JSONL for the fine-tuning script
    distill_training_file = "distill_training_data.jsonl"
    with open(distill_training_file, "w", encoding="utf-8") as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.print(f"  Training data saved: {distill_training_file}")

    # Import and run fine-tuning
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    # Load tokenizer
    console.print(f"\n[bold]Loading student model...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    console.print(f"  [green]✓ Student model loaded[/green]")

    # Apply LoRA
    model_name_lower = student_model_name.lower()
    if "tinyllama" in model_name_lower or "llama" in model_name_lower:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"  LoRA applied — training {100 * trainable / total:.2f}% of parameters")

    # Prepare dataset
    def format_for_training(examples):
        formatted = []
        for ex in examples:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted.append({"text": text})
        return formatted

    formatted = format_for_training(training_data)

    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_list(formatted)
    tokenized = dataset.map(tokenize_function, remove_columns=["text"])
    console.print(f"  [green]✓ {len(tokenized)} examples tokenized[/green]")

    # Train
    console.print(f"\n[bold]Training student on teacher's knowledge...[/bold]")
    console.print(f"  [dim]This teaches TinyLlama to answer like Mistral about YOUR files[/dim]\n")

    training_args = TrainingArguments(
        output_dir="./distill_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=5,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        report_to="none",
        use_cpu=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    start_time = time.time()

    try:
        trainer.train()
    except Exception as e:
        if "memory" in str(e).lower():
            console.print(f"\n[red]✗ Out of memory. Try --lora-rank 2[/red]")
        else:
            console.print(f"\n[red]✗ Training failed: {e}[/red]")
        sys.exit(1)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    console.print(f"\n{'=' * 60}")
    console.print(f"[bold green]DISTILLATION COMPLETE[/bold green]")
    console.print(f"{'=' * 60}")

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("Teacher model", "mistral (via RAG)")
    summary.add_row("Student model", student_model_name)
    summary.add_row("Training examples", str(len(training_data)))
    summary.add_row("Epochs", str(epochs))
    summary.add_row("Training time", f"{hours}h {minutes}m")
    summary.add_row("Adapter saved to", output_dir)
    console.print(summary)

    console.print(f"\n[bold cyan]Next: export to Ollama with:[/bold cyan]")
    console.print(f"  python export_ollama.py --adapter {output_dir} --base-model {student_model_name} --name swapnil-distilled")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Laptop AI - Phase 3: Distillation")
    parser.add_argument("--step", type=str, choices=["teacher", "student", "both"], default="both",
                        help="Which step to run")
    parser.add_argument("--questions", type=str, default="distill_questions.jsonl",
                        help="Questions file from generate_questions.py")
    parser.add_argument("--teacher-output", type=str, default="teacher_answers.jsonl",
                        help="File to save teacher answers")
    parser.add_argument("--student-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Student model to train")
    parser.add_argument("--output", type=str, default="./swapnil-distilled-lora",
                        help="Output directory for distilled adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]LAPTOP AI[/bold cyan] — Phase 3: Distillation\n"
        "[dim]Teacher (Mistral 7B) teaches Student (TinyLlama) via RAG.[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check questions file exists
    if args.step in ("teacher", "both"):
        if not Path(args.questions).exists():
            console.print(f"[red]✗ {args.questions} not found. Run python generate_questions.py first.[/red]")
            sys.exit(1)

    # Run steps
    if args.step in ("teacher", "both"):
        run_teacher(args.questions, args.teacher_output, config)

    if args.step in ("student", "both"):
        if not Path(args.teacher_output).exists():
            console.print(f"[red]✗ {args.teacher_output} not found. Run --step teacher first.[/red]")
            sys.exit(1)
        run_student(args.teacher_output, args.student_model, args.output, args.epochs, args.lora_rank)


if __name__ == "__main__":
    main()
