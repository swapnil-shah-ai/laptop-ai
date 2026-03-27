# Laptop AI — Build the Enterprise AI Stack on Your Laptop

> Scan your files. Ask questions. Get answers in YOUR voice. Zero cloud. Zero API keys. Everything runs on YOUR machine.

## What This Is

A hands-on learning project that builds the enterprise AI stack from scratch — on a regular laptop. Not a framework. Not a tutorial. A working system you run on your own files.

Two phases are live. Two more are coming.

| Phase | What It Does | Status |
|-------|-------------|--------|
| **1. RAG** | Search your files, get answers with sources | ✅ Live |
| **2. Fine-tune** | Model learns your writing style via LoRA | ✅ Live |
| **3. Distill** | Big model teaches small model | ✅ Live |
| **4. Predict** | Anticipate which files you'll need (coming) | 🔜 |

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

Note: The current `prepare_data.py` has hardcoded training pairs as a reference implementation. For your own writing, you'll need to modify the `create_training_pairs()` function to parse your text file and create instruction-response pairs. Each pair needs a context ("Write a cold DM to a startup CEO") and your actual writing as the response.

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

## What's Coming

### Phase 3: Distillation — Knowledge Transfer

Use a powerful API model (Claude, GPT-4) to answer questions via RAG. Save those high-quality answers. Train your local model on them. Result: local model answers like Claude, without ever calling Claude again. API bill drops to zero.

### Phase 4: Predictive ML — Anticipation

System learns your query patterns — what you ask, when, how often. Predicts what files you'll need before you ask. Basic ML (scikit-learn), not deep learning.

---

## Hardware Requirements

| Setup | RAM | What You Can Run |
|-------|-----|-----------------|
| Minimum | 8GB | Embedding + small LLM |
| Recommended | 16GB | Full RAG + TinyLlama fine-tuning |
| Ideal | 32GB or GPU | Phi-3/Mistral fine-tuning + faster everything |

## Project Structure

```
laptop-ai/
├── config.yaml              # All settings — scan paths, models, chunk size
├── check.py                 # Pre-flight verification
├── crawler.py               # Walks file system, finds indexable files
├── chunker.py               # Extracts text from 10+ file types, splits into chunks
├── scan.py                  # Phase 1: Crawl → chunk → embed → store
├── query.py                 # Phase 1: Terminal chat with RAG
├── prepare_data.py          # Phase 2: Create training pairs from writing samples
├── finetune.py              # Phase 2: LoRA fine-tuning
├── export_ollama.py         # Phase 2: Export fine-tuned model to Ollama
├── test_finetune.py         # Phase 2: Compare base vs fine-tuned model
├── requirements.txt         # Phase 1 dependencies
├── requirements_phase2.txt  # Phase 2 dependencies
├── CHEATSHEET.md            # All AI concepts explained in one page
└── .gitignore
```

## Reset

```bash
# Wipe RAG database and re-scan
python scan.py --reset

# Wipe fine-tuning and start fresh
# Delete swapnil-lora/ and swapnil-merged/ folders manually
```

## License

MIT

---

*Built by a non-engineer on a ₹50K laptop with 16GB RAM and no GPU. Not another RAG tutorial — a working system you run on your own files, that answers in your own voice.*
