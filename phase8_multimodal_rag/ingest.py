"""
Phase 8 — Conversational Multimodal RAG
Ingestion: scan ~/Downloads, embed text chunks + images into ChromaDB
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path

import chromadb
import ollama
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

import config

# ── Text extraction ─────────────────────────────────────

def extract_text_from_file(filepath: str) -> str:
    """Extract text content from supported file types."""
    ext = Path(filepath).suffix.lower()
    try:
        if ext in config.TEXT_EXTENSIONS:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            import PyPDF2
            text_parts = []
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        elif ext == ".docx":
            import docx
            doc = docx.Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"  ⚠ Error reading {filepath}: {e}")
    return ""


# ── Chunking ────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ── Embedding: text (Ollama) ────────────────────────────

def embed_texts(texts: list[str], prefix: str = config.TEXT_EMBED_PREFIX_DOC) -> list[list[float]]:
    """Embed text chunks using nomic-embed-text via Ollama.
    Nomic requires prefixes: 'search_document: ' for indexing, 'search_query: ' for querying.
    """
    prefixed = [f"{prefix}{t}" for t in texts]
    # Ollama embed supports batch input
    response = ollama.embed(model=config.TEXT_EMBED_MODEL, input=prefixed)
    return response["embeddings"]


# ── Embedding: images (HF transformers) ─────────────────

class VisionEmbedder:
    """Embed images using nomic-embed-vision-v1.5 via HF transformers.
    Produces 768-dim vectors in the same latent space as nomic-embed-text.
    """
    def __init__(self):
        print("  Loading vision embedder...")
        self.processor = AutoImageProcessor.from_pretrained(config.VISION_EMBED_MODEL)
        self.model = AutoModel.from_pretrained(config.VISION_EMBED_MODEL, trust_remote_code=True)
        self.model.eval()
        print("  Vision embedder ready.")

    def embed_image(self, image_path: str) -> list[float]:
        """Embed a single image file into a 768-dim vector."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).last_hidden_state
        # Take CLS token, normalize
        embedding = F.normalize(output[:, 0], p=2, dim=1)
        return embedding[0].tolist()

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        """Embed multiple images."""
        embeddings = []
        for path in image_paths:
            try:
                emb = self.embed_image(path)
                embeddings.append(emb)
            except Exception as e:
                print(f"  ⚠ Error embedding image {path}: {e}")
                embeddings.append(None)
        return embeddings


# ── File scanning ───────────────────────────────────────

def scan_downloads() -> tuple[list[str], list[str]]:
    """Scan Downloads folder. Return (text_files, image_files).
    Skips files over MAX_FILE_SIZE_MB.
    """
    text_files = []
    image_files = []
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024

    downloads = Path(config.DOWNLOADS_DIR)
    if not downloads.exists():
        print(f"Downloads folder not found: {config.DOWNLOADS_DIR}")
        return [], []

    for filepath in downloads.iterdir():
        if filepath.is_dir():
            continue
        if filepath.name.startswith("."):
            continue

        try:
            size = filepath.stat().st_size
        except OSError:
            continue

        if size > max_bytes or size == 0:
            continue

        ext = filepath.suffix.lower()
        if ext in config.TEXT_EXTENSIONS or ext in config.DOC_EXTENSIONS:
            text_files.append(str(filepath))
        elif ext in config.IMAGE_EXTENSIONS:
            image_files.append(str(filepath))

    return text_files, image_files


def file_id(filepath: str) -> str:
    """Generate a stable ID from filepath for dedup."""
    return hashlib.md5(filepath.encode()).hexdigest()[:12]


# ── Main ingestion ──────────────────────────────────────

def ingest():
    images_only = "--images-only" in sys.argv

    print("=" * 60)
    print("Phase 8 — Multimodal RAG Ingestion")
    if images_only:
        print("MODE: images only (preserving existing text embeddings)")
    print("=" * 60)

    # Scan
    text_files, image_files = scan_downloads()
    print(f"\nFound {len(text_files)} text/doc files, {len(image_files)} image files in {config.DOWNLOADS_DIR}\n")

    if not text_files and not image_files:
        print("Nothing to ingest.")
        return

    # Init ChromaDB
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    if images_only:
        # Preserve existing collection, just add images
        try:
            collection = client.get_collection(config.COLLECTION_NAME)
            print(f"Existing collection: {collection.count()} items. Adding images.\n")
        except Exception:
            print("No existing collection found. Run without --images-only first.")
            return
    else:
        # Full rebuild
        try:
            client.delete_collection(config.COLLECTION_NAME)
            print("Cleared existing collection.\n")
        except Exception:
            pass
        collection = client.create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    total_added = 0

    # ── Ingest text files ───────────────────────────────
    if text_files and not images_only:
        print(f"Ingesting {len(text_files)} text/doc files...")
        for i, filepath in enumerate(text_files):
            filename = Path(filepath).name
            print(f"  [{i+1}/{len(text_files)}] {filename}")

            text = extract_text_from_file(filepath)
            if not text.strip():
                print(f"    → empty, skipping")
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            # Embed in batches of 32
            batch_size = 32
            for b_start in range(0, len(chunks), batch_size):
                batch = chunks[b_start:b_start + batch_size]
                embeddings = embed_texts(batch)

                ids = [f"{file_id(filepath)}_c{b_start + j}" for j in range(len(batch))]
                metadatas = [{
                    "type": "text",
                    "source": filepath,
                    "filename": filename,
                    "chunk_index": b_start + j
                } for j in range(len(batch))]

                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=batch,
                    metadatas=metadatas
                )
                total_added += len(batch)

            print(f"    → {len(chunks)} chunks indexed")

    # ── Ingest images ───────────────────────────────────
    if image_files:
        print(f"\nIngesting {len(image_files)} image files...")
        vision_embedder = VisionEmbedder()

        for i, filepath in enumerate(image_files):
            filename = Path(filepath).name
            print(f"  [{i+1}/{len(image_files)}] {filename}")

            embedding = vision_embedder.embed_image(filepath)
            if embedding is None:
                continue

            img_id = f"img_{file_id(filepath)}"
            collection.add(
                ids=[img_id],
                embeddings=[embedding],
                documents=[f"[IMAGE] {filename}"],  # placeholder text for display
                metadatas=[{
                    "type": "image",
                    "source": filepath,
                    "filename": filename
                }]
            )
            total_added += 1
            print(f"    → embedded")

    print(f"\n{'=' * 60}")
    print(f"Ingestion complete. {total_added} items indexed in ChromaDB.")
    print(f"DB path: {config.CHROMA_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    ingest()
