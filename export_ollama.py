"""
Laptop AI - Phase 2: Export to Ollama
Merges LoRA adapter with base model and creates a custom Ollama model.

Usage:
    python export_ollama.py
    python export_ollama.py --adapter ./swapnil-lora --name swapnil-phi3
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()

DEFAULT_MODEL = "microsoft/phi-3-mini-4k-instruct"


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to Ollama")
    parser.add_argument("--adapter", type=str, default="./swapnil-lora", help="LoRA adapter directory")
    parser.add_argument("--name", type=str, default="swapnil-phi3", help="Ollama model name")
    parser.add_argument("--base-model", type=str, default=DEFAULT_MODEL, help="Base model used for fine-tuning")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]LAPTOP AI[/bold cyan] — Phase 2: Export to Ollama\n"
        "[dim]Merging LoRA adapter and creating your custom model.[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # ---- Check adapter exists ----
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        console.print(f"[red]✗ Adapter not found at {adapter_path}. Run finetune.py first.[/red]")
        sys.exit(1)

    # ---- Step 1: Merge LoRA into base model ----
    console.print(f"\n[bold]Step 1: Merging LoRA adapter with base model...[/bold]")
    console.print(f"  [dim]This loads both models into RAM — may take a few minutes...[/dim]")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dtype = torch.bfloat16

    # Load base model
    console.print(f"  Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load and merge LoRA
    console.print(f"  Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    console.print(f"  Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    merged_path = "./swapnil-merged"
    console.print(f"  Saving merged model to {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    console.print(f"  [green]✓ Merged model saved[/green]")

    # Free memory
    del model
    del base_model
    if hasattr(torch, 'cuda'):
        torch.cuda.empty_cache()

    # ---- Step 2: Convert to GGUF ----
    console.print(f"\n[bold]Step 2: Converting to GGUF format...[/bold]")
    console.print(f"  [dim]Downloading llama.cpp converter...[/dim]")

    # Download convert script from llama.cpp
    convert_script = Path("./convert_hf_to_gguf.py")
    if not convert_script.exists():
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
            urllib.request.urlretrieve(url, str(convert_script))
            console.print(f"  [green]✓ Converter downloaded[/green]")
        except Exception as e:
            console.print(f"  [yellow]⚠ Could not download converter: {e}[/yellow]")
            console.print(f"  [yellow]  Falling back to Ollama Modelfile method...[/yellow]")
            _create_modelfile_fallback(merged_path, args.name)
            return

    # Run conversion
    gguf_path = f"./swapnil-phi3.gguf"
    try:
        result = subprocess.run(
            [sys.executable, str(convert_script), merged_path, "--outfile", gguf_path, "--outtype", "f16"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            console.print(f"  [green]✓ GGUF file created: {gguf_path}[/green]")
        else:
            console.print(f"  [yellow]⚠ GGUF conversion failed: {result.stderr[:200]}[/yellow]")
            console.print(f"  [yellow]  Falling back to Ollama Modelfile method...[/yellow]")
            _create_modelfile_fallback(merged_path, args.name)
            return
    except Exception as e:
        console.print(f"  [yellow]⚠ GGUF conversion failed: {e}[/yellow]")
        _create_modelfile_fallback(merged_path, args.name)
        return

    # ---- Step 3: Create Ollama model ----
    console.print(f"\n[bold]Step 3: Creating Ollama model...[/bold]")

    modelfile_content = f"""FROM {gguf_path}

SYSTEM "You are Swapnil Shah. Write exactly in his style: direct, no fluff, short punchy lines, rhetorical questions, occasional Hindi/Gujarati, lowercase casual tone, sign off with 'Warmth, Swapnil' in professional contexts. Never use corporate jargon. Never be generic. Always be specific and honest."

PARAMETER temperature 0.3
PARAMETER num_predict 1024
"""

    modelfile_path = Path("./Modelfile")
    modelfile_path.write_text(modelfile_content)

    try:
        result = subprocess.run(
            ["ollama", "create", args.name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            console.print(f"  [green]✓ Ollama model '{args.name}' created![/green]")
        else:
            console.print(f"  [yellow]⚠ Ollama create failed: {result.stderr[:200]}[/yellow]")
            console.print(f"  [dim]  You can manually run: ollama create {args.name} -f Modelfile[/dim]")
    except FileNotFoundError:
        console.print(f"  [yellow]⚠ Ollama not found in PATH[/yellow]")
        console.print(f"  [dim]  Run manually: ollama create {args.name} -f Modelfile[/dim]")

    # ---- Done ----
    console.print(f"\n{'=' * 60}")
    console.print(f"[bold green]EXPORT COMPLETE[/bold green]")
    console.print(f"{'=' * 60}")
    console.print(f"\n  To use in RAG pipeline, update config.yaml:")
    console.print(f"    llm_model: {args.name}")
    console.print(f"\n  To test directly:")
    console.print(f"    ollama run {args.name}")
    console.print(f"\n  To compare base vs fine-tuned:")
    console.print(f"    python test_finetune.py")
    console.print()


def _create_modelfile_fallback(merged_path: str, model_name: str):
    """Fallback: create Ollama model from safetensors using Modelfile."""
    console.print(f"\n[bold]Alternative: Creating Ollama model from safetensors...[/bold]")

    modelfile_content = f"""FROM {merged_path}

SYSTEM "You are Swapnil Shah. Write exactly in his style: direct, no fluff, short punchy lines, rhetorical questions, occasional Hindi/Gujarati, lowercase casual tone, sign off with 'Warmth, Swapnil' in professional contexts. Never use corporate jargon. Never be generic. Always be specific and honest."

PARAMETER temperature 0.3
PARAMETER num_predict 1024
"""

    modelfile_path = Path("./Modelfile")
    modelfile_path.write_text(modelfile_content)

    console.print(f"  Modelfile created. Run manually:")
    console.print(f"    ollama create {model_name} -f Modelfile")
    console.print(f"\n  If that doesn't work, you can still test the model with:")
    console.print(f"    python test_finetune.py")


if __name__ == "__main__":
    main()
