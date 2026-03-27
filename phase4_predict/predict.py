"""
Laptop AI — Phase 4: Prediction Engine
Learns your file access patterns and predicts which documents
you'll need — before you ask.

How it works:
    1. Reads your query_log.csv (built by logger.py from Phase 1)
    2. Extracts features: day of week, hour, keywords from questions
    3. Trains a Random Forest classifier for each frequently-accessed file
    4. Predicts: "Right now (this day, this hour), which files will you need?"

What you learn:
    - Feature engineering (turning raw data into ML inputs)
    - TF-IDF vectorization (turning text into numbers)
    - Multi-label classification (predicting multiple outputs)
    - Model evaluation (precision, recall, feature importance)
    - scikit-learn pipeline (end-to-end ML in Python)

Usage:
    python predict.py                  # Train + predict for right now
    python predict.py --day Monday     # Predict for a specific day
    python predict.py --hour 9         # Predict for a specific hour
    python predict.py --stats          # Show model performance stats
    python predict.py --retrain        # Force retrain (ignores cached model)
"""

import csv
import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np

# -------------------------------------------------------------------
# STEP 1: Load and parse the query log
# -------------------------------------------------------------------

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "query_log.csv"
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "predictor.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

# Minimum queries needed. Below this, predictions are noise.
MIN_QUERIES = 50
# Only build classifiers for files accessed at least this many times
MIN_FILE_FREQUENCY = 5
# Number of top files to predict
TOP_K = 5


def load_query_log(log_path: str = LOG_PATH) -> list[dict]:
    """Load query log CSV into list of dicts."""
    if not os.path.isfile(log_path):
        print(f"No query log found at: {log_path}")
        print("Run 'python generate_demo_data.py' to create demo data,")
        print("or use query.py (Phase 1) to build real data over time.")
        sys.exit(1)

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["day_of_week"] = int(row["day_of_week"])
            row["hour"] = int(row["hour"])
            row["num_results"] = int(row["num_results"])
            row["files"] = [
                fp.strip() for fp in row.get("retrieved_files", "").split(";")
                if fp.strip()
            ]
            entries.append(row)

    return entries


# -------------------------------------------------------------------
# STEP 2: Feature engineering
# -------------------------------------------------------------------

def build_features(entries: list[dict]) -> tuple:
    """
    Turn raw log entries into ML features.

    Features per entry:
        - day_of_week one-hot (7 features)
        - hour_bucket one-hot (4 buckets: early_morning, morning, afternoon, evening)
        - TF-IDF of question text (captures topic patterns)

    Returns:
        X: numpy array of features
        file_labels: dict mapping file_path -> binary array (1 if retrieved, 0 if not)
        vectorizer: fitted TF-IDF vectorizer
        feature_names: list of feature names for interpretation
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # --- Day of week: one-hot ---
    days = np.zeros((len(entries), 7))
    for i, e in enumerate(entries):
        days[i, e["day_of_week"]] = 1.0

    # --- Hour bucket: one-hot ---
    def hour_to_bucket(h: int) -> int:
        if h < 6:
            return 0     # early morning / late night
        elif h < 12:
            return 1     # morning
        elif h < 18:
            return 2     # afternoon
        else:
            return 3     # evening

    hours = np.zeros((len(entries), 4))
    for i, e in enumerate(entries):
        hours[i, hour_to_bucket(e["hour"])] = 1.0

    # --- Question text: TF-IDF ---
    questions = [e["question"] for e in entries]
    vectorizer = TfidfVectorizer(
        max_features=100,     # keep it small — we don't have millions of rows
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams ("board meeting", not just "board")
    )
    tfidf = vectorizer.fit_transform(questions).toarray()

    # --- Combine all features ---
    X = np.hstack([days, hours, tfidf])

    # Feature names for interpretation
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    bucket_names = ["late_night", "morning", "afternoon", "evening"]
    tfidf_names = vectorizer.get_feature_names_out().tolist()
    feature_names = day_names + bucket_names + tfidf_names

    # --- Build per-file binary labels ---
    # Count file frequency first
    all_files = []
    for e in entries:
        all_files.extend(e["files"])
    file_counts = Counter(all_files)

    # Only model files that appear often enough
    frequent_files = [
        fp for fp, count in file_counts.most_common()
        if count >= MIN_FILE_FREQUENCY
    ]

    file_labels = {}
    for fp in frequent_files:
        labels = np.array([1 if fp in e["files"] else 0 for e in entries])
        file_labels[fp] = labels

    return X, file_labels, vectorizer, feature_names


# -------------------------------------------------------------------
# STEP 3: Train
# -------------------------------------------------------------------

def train_models(
    X: np.ndarray,
    file_labels: dict,
    feature_names: list,
) -> dict:
    """
    Train one Random Forest classifier per file.

    Why Random Forest?
        - Works well on small datasets (we have hundreds, not millions)
        - Handles mixed features (one-hot + TF-IDF) natively
        - Gives feature importance (so we can explain WHY a file was predicted)
        - No hyperparameter tuning needed for a first pass

    Returns:
        dict of {file_path: trained_model}
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    models = {}
    stats = {}

    print(f"\nTraining classifiers for {len(file_labels)} files...\n")

    for fp, y in file_labels.items():
        # Skip if file was retrieved in almost every query (no signal)
        positive_rate = y.mean()
        if positive_rate > 0.9 or positive_rate < 0.05:
            continue

        clf = RandomForestClassifier(
            n_estimators=50,       # 50 trees — enough for small data
            max_depth=8,           # prevent overfitting
            min_samples_leaf=3,    # need at least 3 examples per leaf
            random_state=42,
            n_jobs=-1,             # use all CPU cores
        )
        clf.fit(X, y)

        # Quick cross-validation score (how reliable is this classifier?)
        if len(y) >= 20:
            cv_scores = cross_val_score(clf, X, y, cv=3, scoring="f1")
            f1 = cv_scores.mean()
        else:
            f1 = 0.0

        # Top 3 most important features for this file
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[-3:][::-1]
        top_features = [(feature_names[i], round(importances[i], 3)) for i in top_idx]

        models[fp] = clf
        stats[fp] = {
            "f1_score": round(f1, 3),
            "positive_rate": round(positive_rate, 3),
            "top_features": top_features,
        }

        filename = Path(fp).name
        print(f"  {filename}")
        print(f"    F1 score: {f1:.2f}  |  Access rate: {positive_rate:.0%}")
        print(f"    Key drivers: {', '.join(f[0] for f in top_features)}")
        print()

    return models, stats


# -------------------------------------------------------------------
# STEP 4: Predict
# -------------------------------------------------------------------

def predict_now(
    models: dict,
    vectorizer,
    day: int = None,
    hour: int = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Predict which files you'll need right now (or at a given day/hour).

    Returns list of {file, probability, reason} sorted by probability.
    """
    now = datetime.now()
    if day is None:
        day = now.weekday()
    if hour is None:
        hour = now.hour

    # Build feature vector for "right now"
    day_vec = np.zeros(7)
    day_vec[day] = 1.0

    def hour_to_bucket(h):
        if h < 6: return 0
        elif h < 12: return 1
        elif h < 18: return 2
        else: return 3

    hour_vec = np.zeros(4)
    hour_vec[hour_to_bucket(hour)] = 1.0

    # Empty question — we're predicting without a specific query
    # The TF-IDF features will all be 0, so prediction relies on day/hour
    tfidf_vec = np.zeros(vectorizer.transform([""]).shape[1])

    X_now = np.concatenate([day_vec, hour_vec, tfidf_vec]).reshape(1, -1)

    # Get probability from each file's classifier
    predictions = []
    for fp, clf in models.items():
        prob = clf.predict_proba(X_now)[0]
        # predict_proba returns [P(0), P(1)] — we want P(1)
        if len(prob) > 1:
            p = prob[1]
        else:
            p = prob[0] if clf.classes_[0] == 1 else 0.0

        predictions.append({
            "file": fp,
            "filename": Path(fp).name,
            "probability": round(p, 3),
        })

    # Sort by probability, return top K
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    return predictions[:top_k]


# -------------------------------------------------------------------
# STEP 5: Save / Load (so you don't retrain every morning)
# -------------------------------------------------------------------

def save_models(models, vectorizer, stats):
    """Save trained models to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(META_PATH, "w") as f:
        # Convert stats to JSON-serializable format
        serializable_stats = {}
        for fp, s in stats.items():
            serializable_stats[fp] = {
                "f1_score": s["f1_score"],
                "positive_rate": s["positive_rate"],
                "top_features": s["top_features"],
            }
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "num_files": len(models),
            "stats": serializable_stats,
        }, f, indent=2)

    print(f"Models saved to {MODEL_DIR}/")


def load_models():
    """Load trained models from disk. Returns None if not found."""
    if not all(os.path.isfile(p) for p in [MODEL_PATH, VECTORIZER_PATH]):
        return None, None, None

    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    return models, vectorizer, meta


# -------------------------------------------------------------------
# STEP 6: Display
# -------------------------------------------------------------------

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def display_predictions(predictions: list[dict], day: int, hour: int):
    """Pretty-print predictions."""
    print()
    print("=" * 60)
    print(f"  PREDICTED FILES — {DAY_NAMES[day]}, {hour:02d}:00")
    print("=" * 60)
    print()

    if not predictions:
        print("  No predictions available. Need more query data.")
        return

    for i, p in enumerate(predictions, 1):
        bar_len = int(p["probability"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct = p["probability"] * 100

        print(f"  {i}. {p['filename']}")
        print(f"     {bar} {pct:.0f}%")
        print(f"     {p['file']}")
        print()

    print("-" * 60)
    print("  Based on your past query patterns at this day/time.")
    print("  Retrain with: python predict.py --retrain")
    print()


def display_stats(stats: dict, meta: dict):
    """Show model performance details."""
    print()
    print("=" * 60)
    print("  MODEL PERFORMANCE")
    print("=" * 60)
    print()
    print(f"  Trained: {meta.get('trained_at', 'unknown')[:19]}")
    print(f"  Files modeled: {meta.get('num_files', 0)}")
    print()

    for fp, s in stats.items():
        filename = Path(fp).name
        f1 = s["f1_score"]
        rate = s["positive_rate"]
        features = s.get("top_features", [])

        quality = "good" if f1 > 0.6 else "okay" if f1 > 0.3 else "weak"

        print(f"  {filename}")
        print(f"    Prediction quality: {quality} (F1={f1:.2f})")
        print(f"    You access this file {rate:.0%} of the time")
        if features:
            print(f"    Key drivers: {', '.join(f[0] for f in features)}")
        print()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Laptop AI Phase 4 — Predict which files you'll need"
    )
    parser.add_argument("--day", type=str, default=None,
                        help="Day to predict for (Monday, Tuesday, etc.)")
    parser.add_argument("--hour", type=int, default=None,
                        help="Hour to predict for (0-23)")
    parser.add_argument("--stats", action="store_true",
                        help="Show model performance statistics")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if cached model exists")
    parser.add_argument("--log", type=str, default=LOG_PATH,
                        help="Path to query log CSV")
    args = parser.parse_args()

    # Parse day name to int
    predict_day = None
    if args.day:
        day_map = {name.lower(): i for i, name in enumerate(DAY_NAMES)}
        predict_day = day_map.get(args.day.lower())
        if predict_day is None:
            print(f"Unknown day: {args.day}. Use Monday, Tuesday, etc.")
            sys.exit(1)

    # Try loading cached model first
    models, vectorizer, meta = None, None, None
    if not args.retrain:
        models, vectorizer, meta = load_models()
        if models:
            print(f"Loaded cached model (trained {meta.get('trained_at', '?')[:10]})")

    # Train if needed
    if models is None:
        entries = load_query_log(args.log)
        print(f"Loaded {len(entries)} query log entries.")

        if len(entries) < MIN_QUERIES:
            print(f"\nNot enough data. Need at least {MIN_QUERIES} queries, have {len(entries)}.")
            print("Keep using query.py, or run 'python generate_demo_data.py' for demo data.")
            sys.exit(1)

        X, file_labels, vectorizer, feature_names = build_features(entries)

        if not file_labels:
            print("\nNo files accessed frequently enough to model.")
            print(f"Need at least {MIN_FILE_FREQUENCY} accesses per file.")
            sys.exit(1)

        models, stats = train_models(X, file_labels, feature_names)
        save_models(models, vectorizer, stats)

    # Show stats if requested
    if args.stats:
        if meta and "stats" in meta:
            display_stats(meta["stats"], meta)
        else:
            print("No stats available. Run with --retrain first.")
        return

    # Predict
    now = datetime.now()
    day = predict_day if predict_day is not None else now.weekday()
    hour = args.hour if args.hour is not None else now.hour

    predictions = predict_now(models, vectorizer, day=day, hour=hour)
    display_predictions(predictions, day, hour)


if __name__ == "__main__":
    main()
