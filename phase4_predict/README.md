# Phase 4 — ML Prediction

> Your laptop learns your patterns and predicts which files you'll need — before you ask.

## What You Learn

This phase teaches you how predictive ML works in production systems:

- **Feature engineering** — turning messy real-world data (timestamps, text) into numbers a model can use
- **TF-IDF vectorization** — how machines "read" text by measuring word importance
- **Multi-label classification** — predicting multiple things at once (which files, not just one file)
- **Random Forest** — an ensemble model that's forgiving on small datasets
- **Cross-validation** — measuring if your model is actually good or just memorizing
- **Feature importance** — understanding WHY the model made a prediction
- **Model persistence** — saving/loading trained models so you don't retrain every time
- **Scheduled inference** — running predictions automatically without human intervention

## How It Works

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐
│  query.py   │────▶│ logger   │────▶│ CSV log  │────▶│  predict  │
│  (Phase 1)  │     │ (silent) │     │ (data)   │     │  (ML)     │
└─────────────┘     └──────────┘     └──────────┘     └─────┬─────┘
                                                            │
                                                   ┌───────▼────────┐
                                                   │  "Today you'll │
                                                   │   need these   │
                                                   │   3 files"     │
                                                   └────────────────┘
```

Every time you query your files (Phase 1), the logger silently records what you asked and what was retrieved. After enough queries, the prediction model learns patterns like:

- **Monday mornings → board meeting notes, strategy deck**
- **Wednesday afternoons → technical specs, API docs**
- **Fridays → weekly report template, KPIs**

## Quick Start

### Option A: Try with demo data (instant)

```bash
cd phase4-predict

# Generate 300 synthetic query entries
python generate_demo_data.py

# Train the model and see predictions
python predict.py
```

### Option B: Use your real data (needs 50+ queries first)

```bash
# 1. Retrofit the logger into your Phase 1 query.py (see below)
# 2. Use query.py normally for a few weeks
# 3. When ready:
python predict.py
```

## Setup

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Retrofit logger into query.py

Add these two lines to your Phase 1 `query.py`:

**At the top, with the other imports:**
```python
from phase4_predict.logger import log_query
```

**After the model returns an answer (inside the query loop), add:**
```python
log_query(question, retrieved_results)
```

That's it. Two lines. The logger handles everything else silently.

### Step 3: Accumulate data

Use `query.py` as normal. The logger silently builds `query_log.csv` in the project root. You need at least 50 queries for basic predictions, 200+ for reliable ones.

Check your progress:
```bash
python -c "from phase4_predict.logger import get_log_stats; print(get_log_stats())"
```

### Step 4: Train and predict

```bash
# Train the model and predict for right now
python predict.py

# Predict for a specific day/time
python predict.py --day Monday --hour 9

# See model performance
python predict.py --stats

# Force retrain after more data accumulates
python predict.py --retrain
```

### Step 5: Schedule daily predictions (optional)

```bash
# Set up 8 AM daily predictions with Windows toast notification
python schedule.py --install

# Change time to 9:30 AM
python schedule.py --install --time 09:30

# Test it now
python schedule.py --test

# Remove the scheduled task
python schedule.py --uninstall
```

## Files

| File | What It Does |
|------|-------------|
| `logger.py` | Silently logs every query to CSV. Add 2 lines to query.py. |
| `predict.py` | Trains ML model, predicts your top files for any day/time. |
| `schedule.py` | Sets up Windows Task Scheduler for daily predictions. |
| `generate_demo_data.py` | Creates synthetic data so you can test immediately. |
| `requirements.txt` | Python dependencies (scikit-learn, numpy). |

## The ML Pipeline Explained

### Feature Engineering (what the model sees)

Raw log entry:
```
timestamp: 2025-03-24T09:15:00
question: "What was the board decision on pricing?"
files: Board_Notes.docx, Pricing_Strategy.xlsx
```

Transformed into features:
```
[0, 1, 0, 0, 0, 0, 0]    ← Monday (one-hot: 7 features)
[0, 1, 0, 0]              ← Morning (one-hot: 4 buckets)  
[0.0, 0.82, 0.0, ...]     ← TF-IDF of "board decision pricing"
```

### Why Random Forest?

| Consideration | Why RF Wins Here |
|--------------|-----------------|
| Small dataset (hundreds, not millions) | RF handles small data well |
| Mixed features (numbers + text) | RF doesn't care about feature types |
| No hyperparameter tuning | Works well out of the box |
| Explainability | Feature importance tells you WHY |
| Overfitting risk | Built-in regularization via bagging |

### Model Evaluation

The script uses 3-fold cross-validation and reports F1 score per file:

- **F1 > 0.6** → good predictions, the pattern is real
- **F1 0.3-0.6** → okay, some signal but noisy
- **F1 < 0.3** → weak, need more data or the access pattern is random

## Example Output

```
============================================================
  PREDICTED FILES — Monday, 09:00
============================================================

  1. Board_Meeting_Notes_2025.docx
     ████████████████████████░░░░░░ 80%
     C:\Users\You\Documents\Board_Meeting_Notes_2025.docx

  2. Q4_Strategy_Deck.pptx
     ██████████████████░░░░░░░░░░░░ 62%
     C:\Users\You\Documents\Q4_Strategy_Deck.pptx

  3. Team_OKRs.docx
     ██████████████░░░░░░░░░░░░░░░░ 45%
     C:\Users\You\Documents\Team_OKRs.docx

------------------------------------------------------------
  Based on your past query patterns at this day/time.
  Retrain with: python predict.py --retrain
```

## This Is Not an Agent

Phase 4 **predicts** — it doesn't **act**. It tells you which files you'll probably need. It doesn't open them, email them, or prepare summaries. That distinction matters:

| Capability | Tool | Agent |
|-----------|------|-------|
| Answer when asked | Phase 1 (RAG) ✓ | |
| Speak in your voice | Phase 2 (Fine-tune) ✓ | |
| Work offline | Phase 3 (Distill) ✓ | |
| Predict what you need | Phase 4 (Predict) ✓ | |
| Decide + act on its own | | Phase 5 (not built yet) |

Phase 5 would take Phase 4's predictions and *do something* — open the files, prepare a morning briefing, draft an email. That's the agent layer. This phase is the prerequisite.

## License

MIT
