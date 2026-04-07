# Laptop AI — Build the Enterprise AI Stack on Your Laptop

> Scan your files. Ask questions. Get answers in YOUR voice. Predict what you'll need tomorrow. Give it a goal — it figures out the rest. Watch it catch its own mistakes. Watch two AI models debate a decision. Ask about your images and documents together. Give it permanent memory — it remembers who you are across sessions. Zero cloud. Zero API keys. Everything runs on YOUR machine.

## What This Is

A hands-on learning project that builds the enterprise AI stack from scratch — on a regular laptop. Not a framework. Not a tutorial. A working system you run on your own files.

Nine phases. Nine levels of the enterprise AI stack.

| Phase | What It Does | What You Learn | Status |
|-------|-------------|----------------|--------|
| **1. RAG** | Search your files, get answers with sources | Embeddings, vector databases, retrieval, generation | ✅ Live |
| **2. Fine-tune** | Model learns your writing style via LoRA | Adapter training, behavior vs knowledge, PEFT | ✅ Live |
| **3. Distill** | Big model teaches small model your domain | Knowledge distillation, teacher-student training | ✅ Live |
| **4. Predict** | Anticipate which files you'll need | Feature engineering, Random Forest, TF-IDF, scheduled inference | ✅ Live |
| **5. Agent** | Give it a goal, it plans and executes | Task decomposition, state management, tool selection, planning loop | ✅ Live |
| **6. Autonomous** | Agent catches its own mistakes and retries | Reflection, self-correction, ReAct loop, goal convergence | ✅ Live |
| **7. Multi-agent** | Two different models debate a decision | Agent identity, agent communication, convergence, judge pattern | ✅ Live |
| **8. Multimodal RAG** | Ask about text files AND images together | Multimodal embeddings, shared latent space, cross-modal retrieval, conversational context | ✅ Live |
| **9. Memory** | Laptop remembers you across sessions | Entity extraction, knowledge graph, cross-session persistence, memory deduplication | ✅ Live |

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

```
ollama pull nomic-embed-text
ollama pull mistral
```

**2. Clone and install**

```
git clone https://github.com/swapnil-shah-ai/laptop-ai.git
cd laptop-ai
pip install -r requirements.txt
```

**3. Configure**

Edit `config.yaml` — set your scan paths:

```
scan_paths:
  - C:/Users/YourName/Documents
  - C:/Users/YourName/Desktop
```

**4. Pre-flight check**

```
python check.py
```

Green across the board = ready to go.

**5. Scan**

```
python scan.py
```

Crawls → chunks → embeds → stores. First scan takes 1-6 hours depending on file count and hardware. Close the laptop lid if needed — just don't close the terminal.

**6. Query**

```
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

```
pip install -r requirements_phase2.txt
```

This adds PyTorch, HuggingFace Transformers, PEFT (LoRA library), and datasets.

**2. Collect your writing**

Create a text file with your writing samples — the more, the better. Minimum 50 samples recommended. Include LinkedIn posts, emails, DMs, anything where your voice comes through.

```
# Save as my_writing.txt in the laptop-ai folder
```

**3. Prepare training data**

```
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

```
# For 16GB RAM (no GPU)
python finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --lora-rank 4

# For 32GB RAM or GPU
python finetune.py --model microsoft/phi-3-mini-4k-instruct

# Quick test (1 epoch)
python finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --epochs 1
```

Training takes 1-3 hours on CPU for TinyLlama, longer for larger models.

**5. Export to Ollama**

```
python export_ollama.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --name my-custom-model
```

**6. Test**

```
python test_finetune.py --finetuned my-custom-model
```

**7. Use in RAG pipeline**

Update `config.yaml`:

```
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

```
python make_questions.py
```

This reads your indexed chunks and creates questions. Output: `distill_questions.jsonl`.

**3. Run distillation**

```
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

```
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

```
pip install -r phase4_predict/requirements.txt
```

**2. Use Phase 1 normally.** The logger is already retrofitted into `query.py` — every question you ask silently builds the training data CSV.

**3. Check your progress**

```
python -c "from phase4_predict.logger import get_log_stats; print(get_log_stats())"
```

Need 50 queries for basic predictions, 200+ for reliable ones.

**4. Train and predict**

```
python phase4_predict/predict.py --retrain
```

**5. Schedule daily predictions (optional)**

```
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

---

## Phase 5: Agent — Goal-Driven Execution

You give the agent a goal in plain English. It breaks the goal into steps, picks the right tool for each step, runs them in order, handles failures, and gives you one combined result. Think of it as a project manager that coordinates Phases 1, 2 and 4 to get a job done.

### How It Works

1. You give it a goal: "prepare me for tomorrow's meeting"
2. The planner (Phi3 Mini) breaks it into steps: predict files → search documents → write a briefing
3. For each step, the agent picks a tool: SEARCH (Phase 1 RAG), WRITE (Phase 2 fine-tuned model), PREDICT (Phase 4 ML), or SUMMARIZE
4. Results from each step feed into the next — the agent chains them together
5. If a step fails, the agent logs it and moves on — it doesn't crash

### Quick Start

```
cd phase5_agent

# Interactive mode
python agent.py

# Single goal
python agent.py "find everything about pricing and summarize it"
python agent.py "prepare me for tomorrow's meeting"
python agent.py "what will I need this Monday morning?"
```

### Architecture

```
┌──────────────┐
│  Your Goal   │
│  (plain      │
│   English)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌─────────────────────────────────┐
│   Planner    │────▶│  Step 1: PREDICT  [Phase 4 ML]  │
│  (Phi3 Mini) │     │  Step 2: SEARCH   [Phase 1 RAG] │
│              │     │  Step 3: WRITE    [Phase 2 LoRA] │
│  Breaks goal │     │  Step 4: SUMMARIZE [Phi3]        │
│  into steps  │     └─────────────┬───────────────────┘
└──────────────┘                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Results from each step  │
                    │  chained together into   │
                    │  one combined output     │
                    └──────────────────────────┘
```

### The 5 Agent Concepts

These are the building blocks of any AI agent — from this project to ChatGPT plugins to enterprise automation:

- **Task decomposition.** Breaking "prepare me for tomorrow" into concrete steps. The planner LLM does this.
- **State management.** Tracking what's done and what's next. A simple checklist — pending, running, done, failed.
- **Tool selection.** For each step, picking the right function to call. SEARCH for finding information, WRITE for generating text, PREDICT for anticipating needs, SUMMARIZE for condensing.
- **Error recovery.** When a tool fails, the agent doesn't stop. It logs the failure, skips the step, and continues. You see what worked and what didn't.
- **Planning loop.** Each step receives results from all previous steps as context. Step 3 knows what step 1 predicted and what step 2 found. The agent chains — it doesn't just run steps in isolation.

### Key Insight

The agent's quality depends on two things: the planner model's intelligence (how well it breaks down goals) and the tool quality (how well each phase works). Swap Phi3 for a larger model and the plans get smarter. Improve your RAG data and search results get better. The agent framework stays the same — it's the orchestration layer, not the intelligence layer.

---

## Phase 6: Autonomous Agent — Self-Correction

Phase 5 plans and executes. If a step produces garbage, it moves on without noticing. Phase 6 adds reflection — after every step, the agent evaluates its own output. If the result is poor, it retries with a different approach. It catches its own mistakes without you telling it to.

Phase 5 = student who submits the first draft. Phase 6 = student who rereads, fixes weak parts, then submits.

### How It Works

1. Same planning and tool selection as Phase 5
2. After each step executes, the agent evaluates the result: "was this good enough?"
3. Evaluation uses deterministic checks first (is it empty? too short? contains an error?) then asks the LLM a binary relevance question
4. If the result fails evaluation, the agent retries with a simpler instruction
5. Maximum 2 retries per step — then it accepts the best available and flags it as "accepted with issues"
6. Every evaluation is logged in a reflection log so you can see the agent's reasoning

### Quick Start

```
cd phase6_autonomous

# Interactive mode
python autonomous_agent.py

# Single goal
python autonomous_agent.py "find everything about pricing and summarize it"

# Side-by-side comparison: Phase 5 vs Phase 6 on the same goal
python autonomous_agent.py --compare "prepare me for tomorrow's meeting"
```

### The 4 New Concepts

These build on Phase 5's foundation. The industry calls this the ReAct pattern (reason, act, observe, reason again):

- **Evaluate.** After each step, check the result. Deterministic rules catch obvious failures (empty, too short, error messages). The LLM handles subjective checks (is this relevant?).
- **Retry.** When evaluation fails, don't repeat the same instruction. Change the approach — simplify, rephrase, make it more specific. Each retry is different from the last.
- **Reflect.** Log every attempt — what was tried, what happened, pass or fail, why. This is the agent's thinking-about-its-own-thinking, visible in the output.
- **Converge.** The agent needs to know when to stop. Max 2 retries prevents infinite loops. If retries are exhausted, accept the best result and be honest about it.

### What the Comparison Shows

Running `--compare` on the same goal reveals the difference:

- Phase 5 marks garbage output as "done" — it doesn't know the difference
- Phase 6 catches that same garbage, retries with a simpler approach, and produces a better result
- Phase 5 says "4 clean, 0 failed" when 2 results were actually unusable
- Phase 6 says "3 clean, 0 with issues" and tells you exactly what it caught

The trade-off: Phase 6 takes longer (evaluation + retries add time). But the output is honest and better.

### Key Insight

Reflection can't fix a broken tool. If the underlying model is too small or the data is too sparse, no amount of retrying will produce a good result. But reflection tells you the tool is broken — instead of silently passing garbage to the next step. That's the difference between an agent that fails silently and an agent you can trust.

---

## Phase 7: Multi-Agent Debate — Different Models, Different Reasoning

Phase 5 is one agent calling tools. Phase 6 is one agent evaluating itself. Phase 7 is agents talking to each other — two genuinely different AI models debating a business decision from opposing perspectives, with a judge synthesizing the result.

This isn't one model pretending to be two people. Mistral 7B and Phi3 3.8B have different architectures, different training data, different reasoning patterns. The debate produces genuinely different arguments, not stylistic variations of the same logic.

### How It Works

1. You type a business decision: "should we build or buy our AI stack?"
2. The LLM dynamically generates two opposing roles — e.g. "Enterprise CTO" vs "CFO". Different topic = different roles. Nothing hardcoded.
3. Agent A (Mistral 7B) argues its position. Agent B (Phi3 3.8B) counters. They go back and forth for 3 rounds, each seeing what the other just said.
4. After each round, both agents rate their confidence (1-10). The system tracks whether they're converging or polarizing.
5. A judge reads the full debate transcript and delivers a verdict: where they agreed, where they disagreed, strongest arguments from each side, and a recommendation.

### Quick Start

```
cd phase7_multiagent

# Interactive mode
python debate.py

# Single topic
python debate.py "should we build or buy our AI stack"

# More rounds
python debate.py "should startups raise VC or bootstrap" --rounds 5
```

### The 4 New Concepts

- **Agent identity.** The LLM generates roles dynamically based on the topic. Different decision = different debaters. Industry term: agent roles or agent personas.
- **Agent communication.** One agent's output becomes another agent's input. Agent B sees what Agent A just argued and responds to it specifically. Industry term: message passing or inter-agent communication.
- **Convergence.** After each round, agents rate their confidence 1-10. The system tracks whether they're moving toward agreement (converging) or staying apart (polarizing). This tells you if the debate is productive or stuck.
- **Judge pattern.** A third call reads the full debate and synthesizes. Where did they agree? Where did they disagree? What should a decision-maker consider? Industry term: LLM-as-judge. This is exactly what Microsoft shipped as "Critique" in Copilot Researcher.

### Key Insight

Most multi-agent demos use one model with different prompts — fake diversity. This uses two genuinely different models producing genuinely different reasoning. Two actually different brains is more diverse than a thousand copies of one brain with different names.

---

## Phase 8: Conversational Multimodal RAG — Text + Images

Phase 1 searched text. Phase 8 searches text AND images together. Ask questions about your downloaded files — PDFs, documents, screenshots, photos — and get answers grounded in both. Gemma 4 E4B-IT sees the retrieved images natively. Conversational: it remembers what you asked before.

### How It Works

1. `ingest.py` scans your Downloads folder, embeds text chunks (nomic-embed-text via Ollama) and images (nomic-embed-vision via HuggingFace) into one ChromaDB collection — both produce 768-dim vectors in the same latent space
2. `chat.py` takes your question, queries text and images separately (reserved slots for each so text doesn't dominate), passes retrieved context to Gemma 4 E4B-IT
3. Gemma 4 reads both the text chunks and the actual images, generates an answer with source citations
4. Conversation history is maintained across turns — 128K context window means ~200 turns before hitting the wall

### Quick Start

```bash
# Pull models (one-time)
ollama pull gemma4:e4b
ollama pull nomic-embed-text

# Install deps
cd phase8_multimodal_rag
pip install -r requirements.txt

# Ingest your Downloads folder
python ingest.py

# Chat
python chat.py
```

### The 6 New Concepts

- **Shared latent space.** Text and image embedders produce vectors in the same coordinate system. A text query matches against both text chunks and images without separate pipelines.
- **Cross-modal retrieval gap.** Text-to-image similarity (0.08) always scores lower than text-to-text (0.68). Fix: query text and images separately with reserved slots, or generate text captions for images during ingestion.
- **Multimodal generation.** Gemma 4 natively sees images — no image-to-text conversion. Retrieved images are passed as actual image bytes alongside text chunks.
- **Quantization.** Compressing model weights: FP16 → Q8 (half memory) → Q4 (quarter memory). Gemma 4 E4B at Q4 = 9.6GB. This is how large models run on laptops.
- **Per-Layer Embeddings (PLE).** Gemma 4 has 8B total params but 4B effective at inference. Not MoE — different technique. Parameter count ≠ capability.
- **Debug order.** When RAG answers are wrong, diagnose bottom-up: extraction (did the PDF text come out clean?) → retrieval (did the right chunks surface?) → generation (was the answer good?). Most teams waste months on prompt engineering when retrieval is broken.

### Key Insight

The hardest part of multimodal RAG isn't the model — it's the data pipeline. PDF extraction breaks on encrypted files, scanned documents and layout-heavy slides. Image retrieval fails when cross-modal similarity is too weak. Fix the data first, then fix the retrieval, then — and only then — touch the prompt.

---

## Phase 9: Memory Layer — Your Laptop Remembers You

Phase 1-8 gave your laptop intelligence about your FILES. Phase 9 gives your laptop intelligence about YOU. It extracts facts and entity relationships from every conversation, stores them permanently and retrieves relevant ones in future sessions — without you re-explaining anything.

A chatbot forgets you every session. An assistant remembers who you are. Phase 9 is the difference.

### How It Works

1. `memory.py` is the memory engine — three core functions: extract (pull facts and entities from a conversation using Gemma 4), store (save to ChromaDB + SQLite with deduplication), retrieve (query both before each inference)
2. `chat_memory.py` wraps Phase 8's document retrieval and adds the memory layer on top. Three retrieval sources feed one answer: your documents (ChromaDB collection 1), your conversation memories (ChromaDB collection 2) and your entity relationships (SQLite)
3. After each conversation turn, Gemma 4 runs an extraction prompt in the background — identifies facts worth remembering, entities (people, projects, companies) and relationships between them
4. On next session startup, the system greets you with context from your last conversation

### Quick Start

```bash
# Models already pulled from Phase 8
# No new dependencies needed

# Chat with memory
python chat_memory.py
```

Commands during chat:
```
/memory        - view all stored facts, entities and relationships
/memory clear  - wipe all memories and start fresh
/quit          - exit
```

### Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│  Your query │────▶│ Retrieve from 3 sources:                     │
└─────────────┘     │  1. Documents (ChromaDB collection 1)        │
                    │  2. Memories  (ChromaDB collection 2)        │
                    │  3. Entity graph (SQLite relationships)      │
                    └──────────────────┬────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │ Gemma 4 generates answer using all 3 sources │
                    └──────────────────┬───────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │ Background: extract facts + entities from    │
                    │ this exchange → store in collection 2 +      │
                    │ SQLite → available for next turn / session    │
                    └──────────────────────────────────────────────┘
```

### The 5 New Concepts

- **Entity extraction.** After each turn, the LLM identifies people, projects, companies and topics mentioned in the exchange. "Get Soham's approval on Q3 budget" extracts Soham (person), Q3 budget (topic) and the relationship (needs approval from).
- **Knowledge graph.** SQLite stores entity relationships with confidence scores. Mention Soham three times across three sessions and the confidence that he is your approver rises from 0.5 to 0.8. The graph gets smarter with use.
- **Cross-session persistence.** Close the terminal, come back tomorrow, ask "anything urgent?" — the memory layer surfaces pending approvals, people and deadlines from yesterday without you re-explaining.
- **Memory deduplication.** Same fact mentioned five times should not create five entries. Cosine similarity above 0.85 between a new fact and an existing one means duplicate — update timestamp, don't create new entry.
- **Multi-source retrieval.** Documents answer "what's in my files." Memories answer "what have I been working on." Entity graph answers "who is connected to what." All three feed one prompt, one answer.

### Key Insight

Within a single session, memory adds zero value — conversation history handles context. Memory's entire value is cross-session persistence. The problem isn't tokens or context windows. The problem is that LLMs are stateless — every session starts from zero. Memory makes them stateful. That is the difference between a chatbot and an assistant.

---

## Hardware Requirements

| Setup | RAM | What You Can Run |
|-------|-----|-----------------|
| Minimum | 8GB | Embedding + small LLM |
| Recommended | 16GB | Full RAG + TinyLlama fine-tuning + distillation + prediction + agent + autonomous + multi-agent debate + multimodal RAG + memory |
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
├── phase5_agent/            # Phase 5: Goal-driven agent
│   └── agent.py             # Planner + tools + state + execution loop
├── phase6_autonomous/       # Phase 6: Autonomous agent with reflection
│   └── autonomous_agent.py  # ReAct loop + evaluation + retry + compare mode
├── phase7_multiagent/       # Phase 7: Multi-agent debate
│   └── debate.py            # Two models debate + convergence + judge verdict
├── phase8_multimodal_rag/   # Phase 8: Conversational multimodal RAG
│   ├── config.py            # Models, paths, chunking params
│   ├── ingest.py            # Scan Downloads, embed text + images into ChromaDB
│   └── chat.py              # Conversational interface with Gemma 4 E4B-IT
├── memory.py                # Phase 9: Memory engine — extract, store, retrieve
├── chat_memory.py           # Phase 9: Chat with persistent memory layer
├── memory.db                # Phase 9: SQLite entity graph (auto-created, gitignored)
├── CHEATSHEET.pdf           # Enterprise AI concepts — 11-page reference
├── requirements.txt         # Phase 1 dependencies
├── requirements_phase2.txt  # Phase 2 dependencies
└── .gitignore
```

## Reset

```bash
# Wipe RAG database and re-scan
python scan.py --reset

# Wipe fine-tuning — delete the LoRA and merged model folders manually

# Wipe Phase 4 predictions — retrain from scratch
python phase4_predict/predict.py --retrain

# Wipe Phase 9 memories — start fresh
python -c "from memory import clear_memories; clear_memories()"
```

## License

MIT

---

*Built by a non-engineer on a regular laptop with 16GB RAM and no GPU. Not another RAG tutorial — a 9-phase learning journey through the full enterprise AI stack, from retrieval to memory.*
