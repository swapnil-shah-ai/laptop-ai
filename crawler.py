"""
Laptop AI - File Crawler
Walks your file system, finds indexable files, skips junk.
"""

import os
import time
from pathlib import Path
from typing import Generator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()


def discover_files(config: dict) -> list[dict]:
    """
    Walk scan_paths, find all supported files, skip excluded folders/files.
    Returns list of dicts: {path, name, extension, size_mb}
    """
    scan_paths = config.get("scan_paths", [])
    supported = set(ext.lower() for ext in config.get("supported_extensions", []))
    skip_folders = set(f.lower() for f in config.get("skip_folders", []))
    skip_files = set(f.lower() for f in config.get("skip_files", []))
    max_size = config.get("max_file_size_mb", 50) * 1024 * 1024  # convert to bytes

    found_files = []
    skipped_count = 0
    error_count = 0

    console.print("\n[bold cyan]Scanning file system...[/bold cyan]\n")

    for scan_path in scan_paths:
        scan_path = Path(scan_path)
        if not scan_path.exists():
            console.print(f"  [yellow]⚠ Path not found: {scan_path}[/yellow]")
            continue

        console.print(f"  [dim]Scanning: {scan_path}[/dim]")

        for root, dirs, files in os.walk(scan_path, topdown=True):
            # Filter out skip folders IN PLACE (prevents os.walk from descending)
            dirs[:] = [
                d for d in dirs
                if d.lower() not in skip_folders
                and not d.startswith(".")
            ]

            for filename in files:
                try:
                    filepath = Path(root) / filename
                    ext = filepath.suffix.lower()

                    # Skip unsupported extensions
                    if ext not in supported:
                        skipped_count += 1
                        continue

                    # Skip excluded filenames
                    if filename.lower() in skip_files:
                        skipped_count += 1
                        continue

                    # Skip files too large
                    file_size = filepath.stat().st_size
                    if file_size > max_size:
                        skipped_count += 1
                        continue

                    # Skip empty files
                    if file_size == 0:
                        skipped_count += 1
                        continue

                    found_files.append({
                        "path": str(filepath),
                        "name": filename,
                        "extension": ext,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                    })

                except (PermissionError, OSError):
                    error_count += 1
                    continue

    # Summary
    console.print(f"\n  [green]✓ Found {len(found_files)} indexable files[/green]")
    console.print(f"  [dim]  Skipped: {skipped_count} | Errors: {error_count}[/dim]")

    # Breakdown by type
    ext_counts = {}
    for f in found_files:
        ext_counts[f["extension"]] = ext_counts.get(f["extension"], 0) + 1
    
    console.print(f"\n  [bold]By type:[/bold]")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        console.print(f"    {ext:8s}  {count:>6,}")

    total_size = sum(f["size_mb"] for f in found_files)
    console.print(f"\n  [bold]Total size:[/bold] {total_size:,.1f} MB")

    return found_files
