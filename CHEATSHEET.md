# Laptop AI — Concept Cheat Sheet
## Everything you learned in 2 days, in one page

---

### THE 4 PHASES

| Phase | What it does | One-liner |
|-------|-------------|-----------|
| RAG | Search your files, answer with sources | Knowledge retrieval |
| Fine-tuning | Model learns your writing style | Behavior change |
| Distillation | Big model teaches small model | Knowledge transfer |
| Predictive ML | System anticipates what you need | Pattern prediction |

---

### RAG = KNOWLEDGE (Phase 1)

**Pipeline:** Crawl files → Chunk text → Embed chunks → Store in ChromaDB → Retrieve → Generate answer

- **Chunk:** Paragraph-sized piece of text, searchable independently
- **Embedding:** Text converted to numbers (768 numbers) that capture MEANING, not words
- **Vector search:** Find chunks whose numbers are closest to your question's numbers
- **Context window:** How much text the LLM can read at once (its working memory)
- **RAG happens at inference time** — when you ask a question, not during preparation
- **Preparation (scan.py)** builds the library. **RAG (query.py)** searches it.

---

### FINE-TUNING = BEHAVIOR (Phase 2)

- Changes the model's weights so it writes like YOU
- **LoRA:** Freeze 99.9% of model, train a tiny adapter layer (0.1%)
- Needs YOUR writing as training data (instruction + your response = one training pair)
- **Loss:** Number showing how different model's output is from yours. Lower = better.
- Fine-tuning does NOT help with changing knowledge — use RAG for that

---

### DISTILLATION = KNOWLEDGE TRANSFER (Phase 3)

- Big model (Claude) answers questions using RAG chunks → save answers
- Train small local model (TinyLlama) on those answers
- Result: local model answers like Claude, without calling Claude
- RAG is mandatory for distillation — without chunks, big model has no context

---

### PREDICTIVE ML = ANTICIPATION (Phase 4)

- Watches your query patterns (what you ask, when, how often)
- Predicts what files you'll need before you ask
- Uses basic ML (scikit-learn), not deep learning
- Simple patterns (time, frequency) = basic ML. Complex patterns (language) = deep learning.

---

### THE AI STACK — WHO DOES WHAT

| Role | What they do |
|------|-------------|
| ML Engineer | Trains models from scratch, fine-tunes, distills |
| AI Engineer | Builds products using models — RAG, agents, evals, guardrails |
| Software Engineer | Builds infrastructure — APIs, databases, frontend, backend |
| Product Leader (YOU) | Decides what to build, which approach fits, hires the right people |

---

### TOOLS — WHAT EACH ONE IS

| Tool | Old world analogy | What it does |
|------|------------------|-------------|
| Ollama | Blu-ray player | Runs AI models locally, handles all optimization |
| HuggingFace | Library/bookstore | Where models are stored and shared |
| PyTorch | Workshop with power tools | Does the math of training/fine-tuning |
| ChromaDB | Filing cabinet with index cards | Stores and searches vector embeddings |
| LangChain | Assembly line connector | Connects different AI components together |

---

### MODEL LANDSCAPE

| Model | Size | What we used it for |
|-------|------|-------------------|
| nomic-embed-text | 274 MB | Converting text to vectors (embedding) |
| Mistral 7B | 4.4 GB | Answering questions in RAG (Phase 1) |
| Phi-3 Mini | 7.6 GB | Too big for our RAM — skipped |
| TinyLlama 1.1B | 2.2 GB | Fine-tuning target (Phase 2) |

---

### KEY CONCEPTS — INTERVIEW READY

- **Temperature:** How creative vs deterministic the model is. Low (0.1) = factual. High (0.9) = creative. Set at inference time, one line of code.
- **Hallucination:** Model generating confident but wrong text. Fix with: constrained prompts, source citations, confidence thresholds, low temperature.
- **Pipeline:** Steps connected in sequence where output of one feeds into the next.
- **Tokens:** How LLMs read text. Roughly 1 word = 1 token.
- **Weights:** Numbers that determine how much importance the model gives to patterns.
- **Bias:** Default tendency before seeing any input.
- **Prompt engineering:** Writing clear instructions for the model — scoping the task, setting boundaries, defining format.
- **Context window ≠ Vector database:** Completely separate. Context window = LLM's reading limit. Vector DB = storage for embeddings.

---

### DECISION FRAMEWORKS — FOR INTERVIEWS

**RAG vs Fine-tuning:**
- Data changes frequently? → RAG
- Need different tone/style/format? → Fine-tuning
- RAG = knowledge. Fine-tuning = behavior.

**Cloud API vs Open-source:**
- Sensitive data? → Open-source on your infra
- Need to ship fast? → Cloud API
- High volume? → Open-source (cheaper at scale)
- Complex reasoning? → Cloud API (still better today)

**Basic ML vs Deep Learning:**
- Simple patterns (time, frequency)? → Basic ML (scikit-learn)
- Complex patterns (language, images)? → Deep learning (PyTorch)
