"""
Phase 8 — Conversational Multimodal RAG
Configuration: models, paths, chunking params
"""

import os

# ── Models ──────────────────────────────────────────────
GENERATION_MODEL = "gemma4:e4b"          # Gemma 4 E4B-IT via Ollama (9.6GB, Q4_K_M, 128K context)
TEXT_EMBED_MODEL = "nomic-embed-text"     # Text embedder via Ollama (768-dim)
VISION_EMBED_MODEL = "nomic-ai/nomic-embed-vision-v1.5"  # Image embedder via HF transformers (768-dim, same latent space)

# ── Paths ───────────────────────────────────────────────
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_multimodal_db")

# ── Supported file types ────────────────────────────────
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".css", ".log"}
DOC_EXTENSIONS = {".pdf", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# ── Chunking ────────────────────────────────────────────
CHUNK_SIZE = 500           # characters per chunk
CHUNK_OVERLAP = 50         # overlap between chunks
MAX_FILE_SIZE_MB = 50      # skip files larger than this

# ── Retrieval ───────────────────────────────────────────
TOP_K_TEXT = 5             # text chunks to retrieve
TOP_K_IMAGES = 3           # images to retrieve
TOP_K_TOTAL = 6            # total results after merging (text + image)

# ── Generation ──────────────────────────────────────────
MAX_HISTORY_TURNS = 20     # conversation turns to keep (128K context gives headroom)
TEMPERATURE = 1.0          # Gemma 4 recommended
TOP_P = 0.95               # Gemma 4 recommended
TOP_K = 64                 # Gemma 4 recommended

# ── ChromaDB ────────────────────────────────────────────
COLLECTION_NAME = "multimodal_downloads"  # single collection, shared latent space
EMBEDDING_DIM = 768

# ── Nomic text embed requires prefixes ──────────────────
# search_document: for indexing, search_query: for querying
TEXT_EMBED_PREFIX_DOC = "search_document: "
TEXT_EMBED_PREFIX_QUERY = "search_query: "
