# Laptop AI — Build the Enterprise AI Stack on Your Laptop

> Phase 1: RAG — Scan your entire laptop, chunk every document, embed locally, query via terminal. Zero cloud. Zero API keys. Everything runs on YOUR machine.

## What This Does

You run one command. It crawls your laptop — every PDF, Word doc, Excel sheet, PowerPoint, text file, code file. Chunks them intelligently. Embeds them locally using Ollama. Stores them in ChromaDB. Then you open a terminal chat and ask questions about YOUR files. It answers with source citations.

No data leaves your machine. Ever.

## Stack

| Component | Tool | Why |
|-----------|------|-----|
| Embeddings | Ollama + nomic-embed-text | Local, free, fast |
| LLM | Ollama + Mistral 7B | Runs on 16GB RAM |
| Vector DB | ChromaDB | Local, no server needed |
| Orchestration | LangChain | Industry standard |
| File parsing | pypdf, python-docx, openpyxl, python-pptx | One loader per file type |

## Setup (Windows)

### 1. Install Ollama

Download from [ollama.com](https://ollama.com/download). Install. Then open a terminal:

```bash
ollama pull nomic-embed-text
ollama pull mistral
```

This downloads the embedding model (~274MB) and the LLM (~4.1GB). One-time download.

### 2. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/laptop-ai.git
cd laptop-ai
pip install -r requirements.txt
```

### 3. Configure

Edit `config.yaml` — set your scan paths:

```yaml
scan_paths:
  - C:/Users/YourName/Documents
  - C:/Users/YourName/Desktop
  - D:/Projects
```

Or scan everything under your user folder (default):

```yaml
scan_paths:
  - C:/Users/YourName
```

### 4. Scan

Make sure Ollama is running (it starts automatically on install, or run `ollama serve`).

```bash
python scan.py
```

This will:
1. **Crawl** — find every supported file
2. **Chunk** — extract text and split intelligently
3. **Embed** — create vector embeddings via Ollama
4. **Store** — save to local ChromaDB

First scan takes time depending on file count. Subsequent runs are faster.

### 5. Query

```bash
python query.py
```

Ask anything:
```
You: What does the Enterprise AI Bible say about RAG architecture?
You: Find all references to Mistral in my documents
You: What were the key decisions in the Q3 board deck?
You: Show me everything related to customer churn
```

Special commands:
- `stats` — show database statistics
- `sources` — list all indexed files
- `quit` — exit

### Single Question Mode

```bash
python query.py --ask "What is our pricing strategy?"
```

## File Types Supported

| Type | Extensions | How It's Chunked |
|------|-----------|-----------------|
| PDF | .pdf | By page, then paragraphs |
| Word | .docx | Paragraphs + tables |
| Excel | .xlsx, .xls | By sheet, rows with headers |
| PowerPoint | .pptx | By slide |
| CSV | .csv | Rows with column headers |
| Text/Code | .txt, .md, .py, .js, .ts, .html, .css | Paragraphs |
| Config | .json, .yaml, .yml | Structured text |
| Logs | .log | Paragraphs |

## Architecture

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

## The 4-Weekend Learning Journey

This is **Phase 1** of a 4-phase project:

| Phase | What You Build | What You Learn |
|-------|---------------|----------------|
| **1. RAG** (this) | Scan laptop → chunk → embed → query | How enterprise RAG actually works |
| **2. Fine-tune** | Train on your writing style | How fine-tuning changes model behavior |
| **3. Distill** | Claude answers → train local Mistral 7B | How knowledge distillation works |
| **4. ML Predict** | Anticipate which files you'll need | How predictive ML layers work |

Each phase builds on the previous. By the end, you have a complete enterprise AI stack — running on your laptop.

## Reset

To wipe the database and re-scan:

```bash
python scan.py --reset
```

## Hardware Requirements

- **Minimum:** 8GB RAM, any modern CPU (embedding only, use smaller LLM)
- **Recommended:** 16GB RAM, i5 or better (runs Mistral 7B comfortably)
- **Storage:** ~2GB for models + database size depends on your files

## License

MIT

---

*Built as an open-source learning project. Not another RAG tutorial — a working system you run on your own files.*
