"""
Laptop AI — Phase 4: Query Logger
Silently logs every query from query.py into a CSV file.
This CSV becomes the training data for the prediction model.

Logged fields:
    - timestamp (ISO format)
    - day_of_week (0=Monday ... 6=Sunday)
    - hour (0-23)
    - question (what you typed)
    - retrieved_files (semicolon-separated file paths returned by ChromaDB)
    - num_results (how many chunks were retrieved)

Usage:
    # In query.py, add these two lines:
    from phase4_predict.logger import log_query
    log_query(question, retrieved_results)
"""

import csv
import os
from datetime import datetime
from pathlib import Path


# Log file lives in the project root, not inside phase4-predict/
# This way all phases can find it easily
DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "query_log.csv"
)

FIELDNAMES = [
    "timestamp",
    "day_of_week",
    "hour",
    "question",
    "retrieved_files",
    "num_results",
]


def log_query(
    question: str,
    retrieved_results: list,
    log_path: str = DEFAULT_LOG_PATH,
) -> None:
    """
    Silently append one row to the query log CSV.

    Args:
        question: The user's question string.
        retrieved_results: List of dicts from ChromaDB results.
            Each dict should have a 'metadata' key with a 'source' field.
            OR a list of file path strings.
        log_path: Path to the CSV log file.
    """
    try:
        now = datetime.now()

        # Extract file paths from results
        files = []
        for r in retrieved_results:
            if isinstance(r, dict):
                source = r.get("metadata", {}).get("source", "")
                if not source:
                    source = r.get("source", "")
                files.append(source)
            elif isinstance(r, str):
                files.append(r)

        row = {
            "timestamp": now.isoformat(),
            "day_of_week": now.weekday(),  # 0=Monday
            "hour": now.hour,
            "question": question.strip(),
            "retrieved_files": ";".join(files),
            "num_results": len(files),
        }

        # Create file with headers if it doesn't exist
        file_exists = os.path.isfile(log_path)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    except Exception:
        # Logger should NEVER crash the main app
        # If logging fails, the query still works — user doesn't notice
        pass


def get_log_stats(log_path: str = DEFAULT_LOG_PATH) -> dict:
    """
    Quick stats on the query log. Used by predict.py to check readiness.

    Returns:
        dict with total_queries, unique_files, date_range, ready (bool)
    """
    if not os.path.isfile(log_path):
        return {"total_queries": 0, "unique_files": 0, "date_range": None, "ready": False}

    total = 0
    all_files = set()
    first_ts = None
    last_ts = None

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            for fp in row.get("retrieved_files", "").split(";"):
                if fp.strip():
                    all_files.add(fp.strip())
            ts = row.get("timestamp", "")
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

    return {
        "total_queries": total,
        "unique_files": len(all_files),
        "date_range": f"{first_ts[:10]} to {last_ts[:10]}" if first_ts else None,
        "ready": total >= 50,  # Minimum for any signal; 200+ for reliable predictions
    }


if __name__ == "__main__":
    # Quick self-test
    stats = get_log_stats()
    print(f"Query log stats: {stats}")
    if stats["total_queries"] == 0:
        print("No queries logged yet. Use query.py to start building your log.")
    elif not stats["ready"]:
        print(f"Need at least 50 queries for basic predictions. You have {stats['total_queries']}.")
    else:
        print("Ready for prediction training!")
