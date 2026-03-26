"""
Laptop AI - Phase 2: LoRA Fine-Tuning
Trains a LoRA adapter on your writing style using Phi-3 Mini.
Runs on CPU with 16GB RAM. Slow but real.

Usage:
    python finetune.py                          # Full training
    python finetune.py --epochs 1               # Quick test (1 epoch)
    python finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0   # Use smaller model
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Default model - Phi-3 Mini is 3.8B params, fits in 16GB with LoRA
DEFAULT_MODEL = "microsoft/phi-3-mini-4k-instruct"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def check_memory():
    """Check available RAM and warn if tight."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        console.print(f"  RAM: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
        if available_gb < 10:
            console.print(f"  [yellow]⚠ Low memory. Close other apps. If training crashes, use --model {FALLBACK_MODEL}[/yellow]")
        return available_gb
    except ImportError:
        return None


def load_training_data(path: str) -> list[dict]:
    """Load JSONL training data."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_for_training(examples: list[dict], tokenizer) -> list[dict]:
    """Format training examples into tokenized inputs."""
    formatted = []
    for ex in examples:
        messages = ex["messages"]
        # Build chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted.append({"text": text})
    return formatted


def main():
    parser = argparse.ArgumentParser(description="Laptop AI - LoRA Fine-Tuning")
    parser.add_argument("--data", type=str, default="training_data.jsonl", help="Training data JSONL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--output", type=str, default="./swapnil-lora", help="Output directory for LoRA adapter")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (lower = less memory)")
    args = parser.parse_args()

    # ---- Banner ----
    console.print(Panel(
        "[bold cyan]LAPTOP AI[/bold cyan] — Phase 2: Fine-Tuning\n"
        "[dim]Teaching the model YOUR writing style via LoRA.[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # ---- Check prerequisites ----
    console.print("\n[bold]Pre-flight checks:[/bold]")

    # Memory
    check_memory()

    # Training data
    data_path = Path(args.data)
    if not data_path.exists():
        console.print(f"  [red]✗ {data_path} not found. Run python prepare_data.py first.[/red]")
        sys.exit(1)

    training_data = load_training_data(str(data_path))
    console.print(f"  Training examples: {len(training_data)}")
    console.print(f"  Model: {args.model}")
    console.print(f"  Epochs: {args.epochs}")
    console.print(f"  LoRA rank: {args.lora_rank}")
    console.print(f"  Output: {args.output}")

    # ---- Import heavy libraries ----
    console.print(f"\n[bold]Loading libraries...[/bold]")
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

    device = "cpu"
    dtype = torch.bfloat16 if torch.cpu.is_available() else torch.float32
    console.print(f"  Device: {device}")
    console.print(f"  Dtype: {dtype}")

    # ---- Load tokenizer ----
    console.print(f"\n[bold]Loading tokenizer...[/bold]")
    console.print(f"  [dim]Downloading {args.model} (first time takes a few minutes)...[/dim]")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"  [green]✓ Tokenizer loaded[/green]")

    # ---- Load model ----
    console.print(f"\n[bold]Loading model into RAM...[/bold]")
    console.print(f"  [dim]This will use ~7-8GB of RAM for Phi-3 Mini...[/dim]")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        console.print(f"  [green]✓ Model loaded[/green]")
    except Exception as e:
        console.print(f"  [red]✗ Failed to load {args.model}: {e}[/red]")
        if args.model != FALLBACK_MODEL:
            console.print(f"  [yellow]Try: python finetune.py --model {FALLBACK_MODEL}[/yellow]")
        sys.exit(1)

    # ---- Apply LoRA ----
    console.print(f"\n[bold]Applying LoRA adapter...[/bold]")

    # Find target modules (attention layers)
    # Phi-3 uses qkv_proj and o_proj
    # TinyLlama uses q_proj, k_proj, v_proj, o_proj
    target_modules = None
    model_name_lower = args.model.lower()
    if "phi" in model_name_lower:
        target_modules = ["qkv_proj", "o_proj"]
    elif "tinyllama" in model_name_lower or "llama" in model_name_lower:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]  # safe default

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Show trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total

    console.print(f"  Total parameters:     {total:>12,}")
    console.print(f"  Trainable (LoRA):     {trainable:>12,}")
    console.print(f"  Trainable %:          {pct:.2f}%")
    console.print(f"  [green]✓ LoRA applied — training only {pct:.2f}% of the model[/green]")

    # ---- Prepare dataset ----
    console.print(f"\n[bold]Preparing dataset...[/bold]")

    formatted = format_for_training(training_data, tokenizer)

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

    # ---- Training ----
    console.print(f"\n{'=' * 60}")
    console.print(f"[bold]TRAINING — this will take several hours on CPU[/bold]")
    console.print(f"{'=' * 60}")
    console.print(f"  [dim]~{len(tokenized)} examples × {args.epochs} epochs = {len(tokenized) * args.epochs} training steps[/dim]")
    console.print(f"  [dim]Estimated time: 2-6 hours depending on model size[/dim]")
    console.print(f"  [dim]Safe to close laptop lid (we set 'do nothing' earlier)[/dim]")
    console.print()

    training_args = TrainingArguments(
        output_dir="./training_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,        # tiny batch for CPU
        gradient_accumulation_steps=4,         # effective batch size = 4
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=5,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,                            # no GPU
        bf16=False,                            # handled at model load
        optim="adamw_torch",
        gradient_checkpointing=True,           # saves memory
        dataloader_pin_memory=False,           # CPU mode
        report_to="none",                      # no wandb etc
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
    except torch.cuda.OutOfMemoryError:
        console.print(f"\n[red]✗ Out of memory. Try: python finetune.py --model {FALLBACK_MODEL} --lora-rank 4[/red]")
        sys.exit(1)
    except Exception as e:
        if "memory" in str(e).lower() or "alloc" in str(e).lower():
            console.print(f"\n[red]✗ Out of memory. Try: python finetune.py --model {FALLBACK_MODEL} --lora-rank 4[/red]")
        else:
            console.print(f"\n[red]✗ Training failed: {e}[/red]")
        sys.exit(1)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    # ---- Save LoRA adapter ----
    console.print(f"\n[bold]Saving LoRA adapter...[/bold]")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    console.print(f"  [green]✓ Saved to {args.output}[/green]")

    # ---- Summary ----
    console.print(f"\n{'=' * 60}")
    console.print(f"[bold green]FINE-TUNING COMPLETE[/bold green]")
    console.print(f"{'=' * 60}")

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("Base model", args.model)
    summary.add_row("Training examples", str(len(training_data)))
    summary.add_row("Epochs", str(args.epochs))
    summary.add_row("LoRA rank", str(args.lora_rank))
    summary.add_row("Training time", f"{hours}h {minutes}m")
    summary.add_row("Adapter saved to", args.output)
    console.print(summary)

    console.print(f"\n[bold cyan]Next: run [white]python export_ollama.py[/white] to create your custom Ollama model.[/bold cyan]\n")


if __name__ == "__main__":
    main()
