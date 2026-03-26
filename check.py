"""
Laptop AI - Pre-flight Check
Run this first to make sure everything is installed and ready.

Usage:
    python check.py
"""

import sys
import subprocess
from rich.console import Console
from rich.panel import Panel

console = Console()


def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    version = f"{v.major}.{v.minor}.{v.micro}"
    if ok:
        console.print(f"  [green]✓[/green] Python {version}")
    else:
        console.print(f"  [red]✗[/red] Python {version} — need 3.10+")
    return ok


def check_package(name):
    try:
        __import__(name)
        console.print(f"  [green]✓[/green] {name}")
        return True
    except ImportError:
        console.print(f"  [red]✗[/red] {name} — run: pip install -r requirements.txt")
        return False


def check_ollama():
    import urllib.request
    import json

    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            console.print(f"  [green]✓[/green] Ollama running ({len(models)} models)")
            return True, models
    except Exception:
        console.print("  [red]✗[/red] Ollama not running — start with: ollama serve")
        return False, []


def check_model(models, model_name):
    found = any(m == model_name or m.startswith(f"{model_name}:") for m in models)
    if found:
        console.print(f"  [green]✓[/green] {model_name}")
    else:
        console.print(f"  [red]✗[/red] {model_name} — run: ollama pull {model_name}")
    return found


def main():
    console.print(Panel(
        "[bold cyan]LAPTOP AI[/bold cyan] — Pre-flight Check",
        border_style="cyan",
    ))

    all_ok = True

    # Python
    console.print("\n[bold]Python:[/bold]")
    all_ok &= check_python()

    # Packages
    console.print("\n[bold]Python packages:[/bold]")
    for pkg in ["yaml", "chromadb", "pypdf", "docx", "openpyxl", "pptx", "rich", "tqdm"]:
        all_ok &= check_package(pkg)

    # Ollama
    console.print("\n[bold]Ollama:[/bold]")
    ollama_ok, models = check_ollama()
    all_ok &= ollama_ok

    if ollama_ok:
        console.print("\n[bold]Models:[/bold]")
        all_ok &= check_model(models, "nomic-embed-text")
        all_ok &= check_model(models, "mistral")

    # Verdict
    console.print()
    if all_ok:
        console.print(Panel(
            "[bold green]All checks passed![/bold green]\n"
            "Run [bold]python scan.py[/bold] to start indexing your files.",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[bold yellow]Some checks failed.[/bold yellow]\n"
            "Fix the issues above and run [bold]python check.py[/bold] again.",
            border_style="yellow",
        ))

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
