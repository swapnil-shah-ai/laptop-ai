# Laptop AI — Build the Enterprise AI Stack on Your Laptop

> Scan your files. Ask questions. Get answers in YOUR voice. Predict what you'll need tomorrow. Zero cloud. Zero API keys. Everything runs on YOUR machine.

## What This Is

A hands-on learning project that builds the enterprise AI stack from scratch — on a regular laptop. Not a framework. Not a tutorial. A working system you run on your own files.

Four phases. Four weekends. Four levels of the enterprise AI stack.

| Phase | What It Does | What You Learn | Status |
|-------|-------------|---------------|--------|
| **1. RAG** | Search your files, get answers with sources | Embeddings, vector databases, retrieval, generation | ✅ Live |
| **2. Fine-tune** | Model learns your writing style via LoRA | Adapter training, behavior vs knowledge, PEFT | ✅ Live |
| **3. Distill** | Big model teaches small model your domain | Knowledge distillation, teacher-student training | ✅ Live |
| **4. Predict** | Anticipate which files you'll need | Feature engineering, Random Forest, TF-IDF, scheduled inference | ✅ Live |

Each phase builds on the previous. By the end, you have a complete enterprise AI stack — running on your laptop.

---

## Phase 1: RAG — Knowledge Retrieval

Crawls your laptop, chunks every document, embeds locally, stores in ChromaDB. Ask questions about YOUR files via terminal. Answers come with source citations.

### Stack

| Component | Tool | Why |
|-----------|------|-----|
| Embeddings | Ollama + nomic-embed-text | Local, free, fast |
| LLM | Ollama + Mistral 7B | Runs on 16GB RAM |
| Vector DB | ChromaDB | Local, no server needed |
| File parsing | pypdf, python-docx, openpyxl, python-pptx | One loader per file type |

### Setup

**1. Install Ollama**

Download from [ollama.com](https://ollama.com/download). Install. Then:

```bash
ollama pull nomic-embed-text
ollama pull mistral
```

**2. Clone and install**

```bash
git clone https://github.com/swapnil-shah-ai/laptop-ai.git
cd laptop-ai
pip install -r requirements.txt
```

**3. Configure**

Edit `config.yaml` — set your scan paths:

```yaml
scan_paths:
  - C:/Users/YourName/Documents
  - C:/Users/YourName/Desktop
```

**4. Pre-flight check**

```bash
python check.py
```

Green across the board = ready to go.

**5. Scan**

```bash
python scan.py
```

Crawls → chunks → embeds → stores. First scan takes 1-6 hours depending on file count and hardware. Close the laptop lid if needed — just don't close the terminal.

**6. Query**

```bash
python query.py
```

Ask anything:
```
You: What was the final pricing we submitted?
You: What does the Enterprise AI Bible say about RAG?
You: Find all references to Q3 in my documents
You: stats
You: sources
```

### Architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Your Files │────▶│ Crawler  │────▶│ Chunker  │────▶│ Embedder │
│  (laptop)   │     │          │     │          │     │ (Ollama) │
└─────────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                           │
                                                           ▼
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│   Answer    │◀────│   LLM    │◀────│ Retrieve │◀────│ ChromaDB │
│ + Sources   │     │ (Ollama) │     │  Top-K   │     │          │
└─────────────┘     └──────────┘     └──────────┘     └──────────┘
```

### File Types Supported

PDF, DOCX, XLSX, XLS, PPTX, CSV, TXT, MD, JSON, YAML, PY, JS, TS, HTML, CSS, LOG — each chunked intelligently by file type.

---

## Phase 2: Fine-Tuning — Behavior Change

Teaches a local model YOUR writing style using LoRA (Low-Rank Adaptation). You provide writing samples, the model learns your tone, rhythm, vocabulary. After training, it writes like you — not like a generic AI.

### How It Works

1. You collect your writing — LinkedIn posts, DMs, emails, WhatsApp messages
2. `prepare_data.py` creates instruction-response training pairs
3. `finetune.py` trains a LoRA adapter on your writing (0.1% of model weights)
4. `export_ollama.py` merges the adapter and creates a custom Ollama model
5. `test_finetune.py` compares base model vs your fine-tuned model side by side

### Setup

**1. Install Phase 2 dependencies**

```bash
pip install -r requirements_phase2.txt
```

This adds PyTorch, HuggingFace Transformers, PEFT (LoRA library), and datasets.

**2. Collect your writing**

Create a text file with your writing samples — the more, the better. Minimum 50 samples recommended. Include LinkedIn posts, emails, DMs, anything where your voice comes through.

```bash
# Save as my_writing.txt in the laptop-ai folder
```

**3. Prepare training data**

```bash
python prepare_data.py --input my_writing.txt --output training_data.jsonl
```

**How to collect your training data:**

The script needs instruction-response pairs in JSONL format. Here's how to build them manually — this is the real skill, not the script.

1. Go to your LinkedIn profile → Posts & Activity. Copy your best 30-50 posts into a text file.
2. For each post, write a short instruction that would have prompted it. Example:
```json
   {"instruction": "Write a contrarian take on why enterprise AI projects fail", "response": "Your actual LinkedIn post text goes here..."}
```

3. Do the same with DMs, emails, WhatsApp messages — anything where your voice comes through.
4. Save the file as `training_data.jsonl` in the laptop-ai folder (one JSON object per line).
5. The more diverse the examples (formal + casual + short + long), the better the model captures your range.

Minimum 30 pairs to see any effect. 100+ is where it gets good. The current `prepare_data.py` has reference examples showing the format — swap them with your own.

**4. Fine-tune**

```bash
# For 16GB RAM (no GPU)
python finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --lora-rank 4

# For 32GB RAM or GPU
python finetune.py --model microsoft/phi-3-mini-4k-instruct

# Quick test (1 epoch)
python finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --epochs 1
```

Training takes 1-3 hours on CPU for TinyLlama, longer for larger models.

**5. Export to Ollama**

```bash
python export_ollama.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --name my-custom-model
```

**6. Test**

```bash
python test_finetune.py --finetuned my-custom-model
```

**7. Use in RAG pipeline**

Update `config.yaml`:

```yaml
llm_model: my-custom-model
```

Now your RAG answers come in YOUR voice.

### Key Concepts

- **RAG = Knowledge.** What the model knows about (your files). Changes frequently, no retraining needed.
- **Fine-tuning = Behavior.** How the model writes (your style). Baked into weights, permanent.
- **LoRA = Efficient fine-tuning.** Freezes 99.9% of the model, trains only 0.1%. Makes fine-tuning possible on consumer hardware.

### Tips for Better Results

- **More data = better results.** 30 samples is the minimum. 100+ is where it gets good.
- **Diverse examples.** Include formal emails, casual messages, long posts, short replies.
- **More epochs.** Try 5-10 epochs instead of 3 if results are weak.
- **Bigger model.** If you have 32GB RAM or a GPU, use Phi-3 Mini or Mistral 7B instead of TinyLlama.

---

## Phase 3: Distillation — Knowledge Transfer

Uses your larger local model (Mistral 7B) as a teacher. The teacher answers questions about your files via RAG. Those high-quality answers become training data for a smaller student model (TinyLlama 1.1B). Result: the small model learns to answer like the big model — faster, lighter, and without needing the big model running.

### How It Works

1. `make_questions.py` reads your indexed chunks and generates questions about them
2. `distill.py` runs a two-step loop:
   - **Teacher pass:** sends each question through RAG (retrieves chunks + Mistral answers)
   - **Student training:** trains TinyLlama on the teacher's question-answer pairs via LoRA
3. After training, the student model answers questions about your files without needing Mistral

### Setup

**1. Make sure Phase 1 is working** — you need a scanned ChromaDB database with your files.

**2. Generate questions from your documents**

```bash
python make_questions.py
```

This reads your indexed chunks and creates questions. Output: `distill_questions.jsonl`.

**3. Run distillation**

```bash
python distill.py
```

This takes time — it asks each question via RAG (teacher), collects answers, then trains the student model. Output: `distill_training_data.jsonl` (the Q&A pairs) and a LoRA adapter in `distill_checkpoints/`.

### Key Concepts

- **Distillation = transferring intelligence.** Big model's knowledge gets compressed into a small model.
- **Teacher-student pattern.** Used everywhere in production AI — OpenAI, Google, Anthropic all use this to make smaller, faster models.
- **Why not just use the big model?** Speed and resources. Mistral 7B uses 4.4GB RAM and takes 10-15 seconds per answer on CPU. TinyLlama uses 1.1GB and answers in 2-3 seconds. In production, that difference is the difference between viable and not.
- **Zero API cost.** The teacher is your local Mistral — no Claude or GPT API calls needed. Everything stays on your machine.

---

## Phase 4: Predictive ML — Anticipation

Your laptop learns your file access patterns and predicts which documents you'll need — before you ask. This is the jump from "tool that waits" to "system that anticipates."

### How It Works

1. `logger.py` silently logs every query from Phase 1: what you asked, when, which files came back
2. After enough queries (50+), `predict.py` trains a Random Forest classifier on your patterns
3. It learns things like: "Monday mornings → board notes" or "Friday afternoons → weekly report"
4. `schedule.py` sets up a daily 8 AM Windows notification with your predicted files

### Quick Start (demo data)

```bash
cd phase4_predict

# Generate 300 synthetic query entries
python generate_demo_data.py

# Train and predict
python predict.py

# See how predictions shift by day
python predict.py --day Monday --hour 9
python predict.py --day Saturday --hour 10

# View model performance
python predict.py --stats
```

### Setup for Real Data

**1. Install Phase 4 dependencies**

```bash
pip install -r phase4_predict/requirements.txt
```

**2. Use Phase 1 normally.** The logger is already retrofitted into `query.py` — every question you ask silently builds the training data CSV.

**3. Check your progress**

```bash
python -c "from phase4_predict.logger import get_log_stats; print(get_log_stats())"
```

Need 50 queries for basic predictions, 200+ for reliable ones.

**4. Train and predict**

```bash
python phase4_predict/predict.py --retrain
```

**5. Schedule daily predictions (optional)**

```bash
python phase4_predict/schedule.py --install            # Daily 8 AM predictions
python phase4_predict/schedule.py --install --time 09:30  # Change time
python phase4_predict/schedule.py --test               # Test now
```

### Key Concepts

- **Feature engineering.** Turning raw data (timestamps, text) into numbers a model can learn from.
- **TF-IDF.** How machines measure word importance — "budget" in a question is more predictive than "show" or "me."
- **Random Forest.** An ensemble of decision trees. Works well on small datasets. Gives feature importance so you know WHY a prediction was made.
- **Cross-validation.** Measuring if the model actually learned patterns or just memorized the data.
- **Scheduled inference.** Running predictions automatically — how ML works in production, not when a human remembers to trigger it.

### This Is Not an Agent

Phase 4 predicts — it doesn't act. It tells you which files you'll probably need. It doesn't open them, email them, or prepare summaries. A Phase 5 agent layer would take these predictions and do something with them. That's not built yet.

---

## Hardware Requirements

| Setup | RAM | What You Can Run |
|-------|-----|-----------------|
| Minimum | 8GB | Embedding + small LLM |
| Recommended | 16GB | Full RAG + TinyLlama fine-tuning + distillation + prediction |
| Ideal | 32GB or GPU | Phi-3/Mistral fine-tuning + faster everything |

## Project Structure

```
laptop-ai/
├── config.yaml              # All settings — scan paths, models, chunk size
├── check.py                 # Pre-flight verification
├── crawler.py               # Walks file system, finds indexable files
├── chunker.py               # Extracts text from 10+ file types, splits into chunks
├── scan.py                  # Phase 1: Crawl → chunk → embed → store
├── query.py                 # Phase 1: Terminal chat with RAG (+ Phase 4 logger)
├── prepare_data.py          # Phase 2: Create training pairs from writing samples
├── finetune.py              # Phase 2: LoRA fine-tuning
├── export_ollama.py         # Phase 2: Export fine-tuned model to Ollama
├── test_finetune.py         # Phase 2: Compare base vs fine-tuned model
├── make_questions.py        # Phase 3: Generate questions from your indexed chunks
├── distill.py               # Phase 3: Teacher-student distillation pipeline
├── phase4_predict/          # Phase 4: ML prediction module
│   ├── logger.py            # Silently logs every query to CSV
│   ├── predict.py           # Trains Random Forest, predicts files by day/time
│   ├── generate_demo_data.py # Synthetic data for instant testing
│   ├── schedule.py          # Windows Task Scheduler for daily predictions
│   └── requirements.txt     # Phase 4 dependencies (scikit-learn, numpy)
├── requirements.txt         # Phase 1 dependencies
├── requirements_phase2.txt  # Phase 2 dependencies
├── CHEATSHEET.md            # All AI concepts explained in one page
└── .gitignore
```

## Reset

```bash
# Wipe RAG database and re-scan
python scan.py --reset

# Wipe fine-tuning — delete the LoRA and merged model folders manually

# Wipe Phase 4 predictions — retrain from scratch
python phase4_predict/predict.py --retrain
```

## License

MIT

---

*Built by a non-engineer on a ₹50K laptop with 16GB RAM and no GPU. Not another RAG tutorial — a 4-phase learning journey through the full enterprise AI stack, from retrieval to prediction.*
