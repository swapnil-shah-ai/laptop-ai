"""
Laptop AI - Phase 2: Test Fine-Tuned Model
Compares base model vs your fine-tuned model side by side.

Usage:
    python test_finetune.py
    python test_finetune.py --prompt "Write a cold DM to a startup CEO"
"""

import argparse
import json
import urllib.request
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

console = Console()


def query_ollama(prompt: str, model: str, system: str = None) -> str:
    """Query Ollama and get response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "prompt": prompt if not system else f"{system}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": 0.3,
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
            return data.get("response", "No response.")
    except Exception as e:
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model vs base")
    parser.add_argument("--prompt", type=str, help="Custom prompt to test")
    parser.add_argument("--base", type=str, default="mistral", help="Base model name in Ollama")
    parser.add_argument("--finetuned", type=str, default="swapnil-phi3", help="Fine-tuned model name in Ollama")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]LAPTOP AI[/bold cyan] — Fine-Tuning Comparison\n"
        f"[dim]Base: {args.base} vs Fine-tuned: {args.finetuned}[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # Test prompts
    test_prompts = [
        "Write a cold DM to a startup founder about a product leadership role",
        "Write a LinkedIn post about why most AI startups will fail",
        "Reply to a friend who asks what you're doing after your startup shutdown",
        "Write an email pushing back on a business proposal you disagree with",
    ]

    if args.prompt:
        test_prompts = [args.prompt]

    for i, prompt in enumerate(test_prompts, 1):
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Test {i}: {prompt}[/bold]")
        console.print(f"{'=' * 60}\n")

        # Base model
        console.print(f"  [dim]Generating from {args.base}...[/dim]")
        base_response = query_ollama(prompt, args.base)

        console.print(Panel(
            base_response,
            title=f"[yellow]{args.base} (base)[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))

        # Fine-tuned model
        console.print(f"  [dim]Generating from {args.finetuned}...[/dim]")
        ft_response = query_ollama(prompt, args.finetuned)

        console.print(Panel(
            ft_response,
            title=f"[green]{args.finetuned} (fine-tuned)[/green]",
            border_style="green",
            padding=(1, 2),
        ))

        console.print()

    console.print("[bold]Look for differences in:[/bold]")
    console.print("  - Tone (punchy vs generic)")
    console.print("  - Line length (short vs paragraph)")
    console.print("  - Sign-off style ('Warmth, Swapnil' vs generic)")
    console.print("  - Hindi/Gujarati sprinkled in")
    console.print("  - Directness (no fluff vs corporate speak)")
    console.print()


if __name__ == "__main__":
    main()
