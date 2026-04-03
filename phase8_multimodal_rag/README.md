# Phase 8 — Conversational Multimodal RAG

Ask questions about your downloaded files — text documents AND images — using a locally-running multimodal LLM.

## What this does

Scans your `~/Downloads` folder, embeds text chunks and images into a shared vector space, then lets you ask questions in a conversational loop. When you ask a question, the system retrieves the most relevant text chunks AND images, passes them all to Gemma 4 (which can natively see images), and generates an answer grounded in your actual files.

## Architecture

```
~/Downloads (text + images)
         │
    ┌────┴────┐
    ▼         ▼
Text chunks   Images
    │         │
    ▼         ▼
nomic-embed   nomic-embed-vision
  (Ollama)     (HF transformers)
    │         │
    └────┬────┘
         ▼
   ChromaDB (single collection)
   768-dim shared latent space
         │
         ▼ ← user query (text embedding)
   Top-k retrieval (mixed text + images)
         │
         ▼
   Gemma 4 E4B-IT (multimodal generation)
         │
         ▼
   Response (grounded in retrieved context)
         │
         ▼
   Conversation history (appended, trimmed at 20 turns)
```

## Key concepts (what's new vs Phase 1)

1. **Shared latent space** — nomic-embed-text and nomic-embed-vision produce 768-dim vectors in the same vector space. A text query can match against both text chunks and images. This is the core trick that makes multimodal retrieval work without separate search pipelines.

2. **Multimodal generation** — Gemma 4 E4B-IT natively accepts images alongside text. When retrieved results include images, we pass the actual image bytes to the model. No image-to-text conversion needed.

3. **Single collection** — Because the embeddings share a latent space, we use one ChromaDB collection for both text and images. Metadata field `type` distinguishes them. One query, one search, mixed results ranked by cosine similarity.

4. **Nomic text embed prefixes** — nomic-embed-text requires prefixes: `search_document:` for indexing, `search_query:` for querying. Without these, retrieval quality drops significantly. This is a common gotcha.

5. **Conversation history** — 128K context window (vs 4K in Phase 1) means we can hold ~20 full turns without hitting limits. History is trimmed from the front when it exceeds the cap.

6. **Per-Layer Embeddings (PLE)** — Gemma 4 E4B has 8B total params but only 4B "effective" params active at inference, thanks to PLE. Not MoE — different technique. The compute and memory footprint behaves like a 4B model.

## Setup

### Prerequisites
- [Ollama](https://ollama.com/) installed and running
- Python 3.10+
- ~10GB disk for Gemma 4 E4B model
- ~500MB for nomic-embed-vision weights (auto-downloaded)

### Install

```bash
# Pull models
ollama pull gemma4:e4b
ollama pull nomic-embed-text

# Install Python deps
pip install -r requirements.txt
```

### Run

```bash
# Step 1: Ingest your Downloads folder
python ingest.py

# Step 2: Chat
python chat.py
```

### Chat commands
- `/clear` — reset conversation history
- `/count` — show indexed items and history length
- `/quit` — exit

## Models used

| Model | Role | Size | Via |
|-------|------|------|----|
| Gemma 4 E4B-IT | Multimodal generation | 9.6GB (Q4_K_M) | Ollama |
| nomic-embed-text | Text embedding (768-dim) | 274MB | Ollama |
| nomic-embed-vision-v1.5 | Image embedding (768-dim) | ~360MB | HF transformers |

## File support

**Text**: .txt, .md, .csv, .json, .py, .js, .html, .css, .log, .pdf, .docx

**Images**: .png, .jpg, .jpeg, .webp, .bmp, .gif

## Token math (Phase 8 vs Phase 1)

Phase 1 (Mistral 7B, 4K context):
- Turn 1: system(100) + history(0) + chunks(600) + question(50) = 750 tokens
- Turn 6-7: exceeded 4K context → broke

Phase 8 (Gemma 4 E4B, 128K context):
- Turn 1: system(200) + context(2000) + history(0) + question(50) = 2,250 tokens
- Turn 20: system(200) + context(2000) + history(8000) + question(50) = 10,250 tokens
- Headroom: ~118K tokens remaining at turn 20

The 128K context changes the game from "how do I fit everything" to "I have room for everything."

## Phases 1-8 progression

| Phase | Concept | Model |
|-------|---------|-------|
| 1 | RAG (text retrieval + generation) | Mistral 7B |
| 2 | LoRA fine-tuning | TinyLlama 1.1B |
| 3 | Knowledge distillation | Mistral → TinyLlama |
| 4 | ML prediction (Random Forest) | scikit-learn |
| 5 | Standard agent (tool use) | Mistral 7B |
| 6 | Autonomous agent (reflection, ReAct) | Mistral 7B |
| 7 | Multi-agent debate | Mistral vs Phi-3 |
| **8** | **Conversational multimodal RAG** | **Gemma 4 E4B-IT** |
