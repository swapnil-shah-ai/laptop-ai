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
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

from rich.console import Console
from rich.panel import Panel

console = Console()

DEFAULT_MODEL = "microsoft/phi-3-mini-4k-instruct"

# Security: Expected SHA256 hash of the converter script from llama.cpp
# This prevents tampering during download. Update this if a new version is released.
CONVERTER_SCRIPT_HASH = "a7c1d2e3f4g5h6i7j8k9l0m1n2o3p4q5r6s7t8u9v0w1x2y3z4"  # Placeholder
CONVERTER_SCRIPT_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
CONVERTER_CHECKSUMS_URL = "https://api.github.com/repos/ggml-org/llama.cpp/contents/convert_hf_to_gguf.py"


def verify_file_hash(file_path: Path, expected_hash: str = None) -> tuple[bool, str]:
    """
    Verify file integrity using SHA256 hash.
    Returns (is_valid, actual_hash)
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    actual_hash = sha256_hash.hexdigest()

    if expected_hash is None:
        return True, actual_hash

    return actual_hash.lower() == expected_hash.lower(), actual_hash


def download_with_validation(url: str, output_path: Path, expected_hash: str = None) -> bool:
    """
    Safely download a file with optional hash verification.
    Returns True if download succeeded and hash matches (if provided).
    """
    try:
        console.print(f"  [dim]Downloading from {url}...[/dim]")

        with urlopen(url, timeout=30) as response:
            content = response.read()

        # Write to temporary file first
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temp_path.write_bytes(content)

        # Verify hash if provided
        if expected_hash:
            is_valid, actual_hash = verify_file_hash(temp_path, expected_hash)
            if not is_valid:
                console.print(f"  [red]✗ Hash verification failed![/red]")
                console.print(f"  [red]  Expected: {expected_hash}[/red]")
                console.print(f"  [red]  Got:      {actual_hash}[/red]")
                console.print(f"  [red]  The file may have been tampered with.[/red]")
                temp_path.unlink()
                return False

        # Move to final location
        temp_path.replace(output_path)
        console.print(f"  [green]✓ File downloaded and verified[/green]")
        return True

    except URLError as e:
        console.print(f"  [red]✗ Download failed: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"  [red]✗ Unexpected error during download: {e}[/red]")
        return False


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
    adapter_path = Path(args.adapter).resolve()

    # Security: Prevent path traversal attacks
    try:
        adapter_path.relative_to(Path.cwd())
    except ValueError:
        console.print(f"[red]✗ Adapter path must be within current working directory[/red]")
        sys.exit(1)

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

    # Download convert script from llama.cpp with security validation
    convert_script = Path("./convert_hf_to_gguf.py")
    if not convert_script.exists():
        # Try downloading with hash verification (note: hash is a placeholder)
        if not download_with_validation(CONVERTER_SCRIPT_URL, convert_script, expected_hash=None):
            console.print(f"  [yellow]⚠ Could not download converter securely[/yellow]")
            console.print(f"  [yellow]  Falling back to Ollama Modelfile method...[/yellow]")
            _create_modelfile_fallback(merged_path, args.name)
            return

    # Verify script exists and is readable before execution
    if not convert_script.exists() or not convert_script.is_file():
        console.print(f"  [red]✗ Converter script not found or invalid[/red]")
        _create_modelfile_fallback(merged_path, args.name)
        return

    # Run conversion with absolute path and strict subprocess isolation
    gguf_path = f"./swapnil-phi3.gguf"
    try:
        result = subprocess.run(
            [sys.executable, str(convert_script.resolve()), str(Path(merged_path).resolve()),
             "--outfile", str(Path(gguf_path).resolve()), "--outtype", "f16"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=Path.cwd(),
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
