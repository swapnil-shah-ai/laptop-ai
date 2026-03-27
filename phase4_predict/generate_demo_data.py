"""
Laptop AI — Phase 4: Demo Data Generator
Creates realistic synthetic query log data so you can test
predict.py immediately — no need to wait 3 weeks for real data.

The synthetic data mimics real work patterns:
    - Monday mornings: project docs, meeting notes
    - Midweek: technical files, spreadsheets
    - Fridays: reports, summaries
    - Late nights: personal docs, learning material

Usage:
    python generate_demo_data.py              # Generate 300 entries
    python generate_demo_data.py --count 500  # Generate 500 entries
"""

import csv
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import os

# Simulated file paths — feel free to edit these to match YOUR files
SAMPLE_FILES = {
    "work_docs": [
        r"C:\Users\You\Documents\Q4_Strategy_Deck.pptx",
        r"C:\Users\You\Documents\Board_Meeting_Notes_2025.docx",
        r"C:\Users\You\Documents\Product_Roadmap.xlsx",
        r"C:\Users\You\Documents\Team_OKRs.docx",
        r"C:\Users\You\Documents\Client_Proposal_Acme.pdf",
    ],
    "technical": [
        r"C:\Users\You\Documents\API_Documentation.md",
        r"C:\Users\You\Documents\Architecture_Diagram.pdf",
        r"C:\Users\You\Documents\Database_Schema.sql",
        r"C:\Users\You\Code\main.py",
        r"C:\Users\You\Documents\Tech_Spec_v3.docx",
    ],
    "finance": [
        r"C:\Users\You\Documents\Budget_2025.xlsx",
        r"C:\Users\You\Documents\Invoice_Template.xlsx",
        r"C:\Users\You\Documents\Expense_Report_Q3.pdf",
        r"C:\Users\You\Documents\Revenue_Forecast.xlsx",
    ],
    "personal": [
        r"C:\Users\You\Documents\Resume_2025.pdf",
        r"C:\Users\You\Documents\Learning_Notes_AI.md",
        r"C:\Users\You\Documents\Book_Summaries.txt",
        r"C:\Users\You\Documents\Side_Project_Ideas.md",
    ],
    "reports": [
        r"C:\Users\You\Documents\Weekly_Report_Template.docx",
        r"C:\Users\You\Documents\Monthly_KPIs.xlsx",
        r"C:\Users\You\Documents\Quarterly_Review.pptx",
    ],
}

# Simulated questions — these match the file categories above
SAMPLE_QUESTIONS = {
    "work_docs": [
        "What's our strategy for next quarter?",
        "Find the board meeting notes",
        "What are the product priorities?",
        "Show me the team OKRs",
        "Pull up the Acme proposal",
        "What did we decide in the last board meeting?",
        "What's the product roadmap timeline?",
    ],
    "technical": [
        "How does the API authentication work?",
        "Show me the database schema",
        "What's the system architecture?",
        "Find the tech spec for the new feature",
        "What endpoints are available?",
        "How is the data pipeline structured?",
    ],
    "finance": [
        "What's our budget for Q4?",
        "Show me the revenue forecast",
        "Pull up the expense report",
        "What are this month's numbers?",
        "Find the invoice template",
    ],
    "personal": [
        "What was that AI concept I noted down?",
        "Find my learning notes",
        "Show me the book summaries",
        "What side project ideas did I write down?",
        "Pull up my resume",
    ],
    "reports": [
        "Where's the weekly report template?",
        "Show me last month's KPIs",
        "Find the quarterly review deck",
        "What metrics did we track?",
    ],
}

# Work patterns — probability of each category by (day_of_week, hour_bucket)
# This is what makes the predictions non-random
PATTERNS = {
    # Monday morning: strategy + meetings
    (0, "morning"): {"work_docs": 0.5, "technical": 0.1, "finance": 0.1, "reports": 0.2, "personal": 0.1},
    (0, "afternoon"): {"work_docs": 0.3, "technical": 0.3, "finance": 0.1, "reports": 0.2, "personal": 0.1},
    # Tuesday-Thursday: heavy technical + work
    (1, "morning"): {"work_docs": 0.2, "technical": 0.4, "finance": 0.1, "reports": 0.1, "personal": 0.2},
    (1, "afternoon"): {"work_docs": 0.2, "technical": 0.4, "finance": 0.2, "reports": 0.1, "personal": 0.1},
    (2, "morning"): {"work_docs": 0.2, "technical": 0.4, "finance": 0.1, "reports": 0.1, "personal": 0.2},
    (2, "afternoon"): {"work_docs": 0.3, "technical": 0.3, "finance": 0.2, "reports": 0.1, "personal": 0.1},
    (3, "morning"): {"work_docs": 0.2, "technical": 0.4, "finance": 0.1, "reports": 0.1, "personal": 0.2},
    (3, "afternoon"): {"work_docs": 0.2, "technical": 0.3, "finance": 0.2, "reports": 0.2, "personal": 0.1},
    # Friday: reports + wrap-up
    (4, "morning"): {"work_docs": 0.2, "technical": 0.1, "finance": 0.2, "reports": 0.4, "personal": 0.1},
    (4, "afternoon"): {"work_docs": 0.1, "technical": 0.1, "finance": 0.1, "reports": 0.4, "personal": 0.3},
    # Weekend: personal + learning
    (5, "morning"): {"work_docs": 0.05, "technical": 0.2, "finance": 0.05, "reports": 0.0, "personal": 0.7},
    (5, "afternoon"): {"work_docs": 0.05, "technical": 0.15, "finance": 0.0, "reports": 0.0, "personal": 0.8},
    (6, "morning"): {"work_docs": 0.1, "technical": 0.1, "finance": 0.0, "reports": 0.0, "personal": 0.8},
    (6, "afternoon"): {"work_docs": 0.1, "technical": 0.1, "finance": 0.05, "reports": 0.05, "personal": 0.7},
}

# Evening patterns (same across all days, skews personal)
EVENING_PATTERN = {"work_docs": 0.05, "technical": 0.15, "finance": 0.0, "reports": 0.0, "personal": 0.8}


def get_hour_bucket(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"


def generate_entry(timestamp: datetime) -> dict:
    """Generate one synthetic query log entry."""
    day = timestamp.weekday()
    hour = timestamp.hour
    bucket = get_hour_bucket(hour)

    # Pick category based on day/time patterns
    if bucket == "evening":
        pattern = EVENING_PATTERN
    else:
        pattern = PATTERNS.get((day, bucket), PATTERNS.get((0, "morning")))

    categories = list(pattern.keys())
    weights = list(pattern.values())
    category = random.choices(categories, weights=weights, k=1)[0]

    # Pick a question and 2-4 files from that category
    question = random.choice(SAMPLE_QUESTIONS[category])
    num_files = random.randint(2, min(4, len(SAMPLE_FILES[category])))
    files = random.sample(SAMPLE_FILES[category], num_files)

    # Occasionally mix in a file from another category (realistic noise)
    if random.random() < 0.15:
        other_cat = random.choice([c for c in categories if c != category])
        files.append(random.choice(SAMPLE_FILES[other_cat]))

    return {
        "timestamp": timestamp.isoformat(),
        "day_of_week": day,
        "hour": hour,
        "question": question,
        "retrieved_files": ";".join(files),
        "num_results": len(files),
    }


def generate_demo_log(count: int = 300, output_path: str = None):
    """Generate a full synthetic query log CSV."""
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "query_log.csv"
        )

    # Spread entries over the last 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    total_seconds = int((end_time - start_time).total_seconds())

    # Generate random timestamps, biased toward work hours
    entries = []
    for _ in range(count):
        # Random day offset
        day_offset = random.randint(0, 29)
        day_base = start_time + timedelta(days=day_offset)

        # Hour — biased toward work hours (9-18) with some evening
        hour_weights = [0] * 6 + [1, 2, 3, 5, 5, 5, 4, 4, 4, 3, 3, 2, 1, 1, 1, 0, 0, 0]
        hour = random.choices(range(24), weights=hour_weights, k=1)[0]
        minute = random.randint(0, 59)

        ts = day_base.replace(hour=hour, minute=minute, second=random.randint(0, 59))
        entries.append(generate_entry(ts))

    # Sort by timestamp
    entries.sort(key=lambda x: x["timestamp"])

    # Write CSV
    fieldnames = ["timestamp", "day_of_week", "hour", "question", "retrieved_files", "num_results"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)

    print(f"Generated {count} synthetic query log entries")
    print(f"Saved to: {output_path}")
    print(f"Date range: {entries[0]['timestamp'][:10]} to {entries[-1]['timestamp'][:10]}")
    print(f"\nRun 'python predict.py' to train the model and see predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic query log for Phase 4 testing")
    parser.add_argument("--count", type=int, default=300, help="Number of entries to generate (default: 300)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: ../query_log.csv)")
    args = parser.parse_args()

    generate_demo_log(count=args.count, output_path=args.output)
