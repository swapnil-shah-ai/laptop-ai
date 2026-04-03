"""
Phase 8 — Conversational Multimodal RAG
Chat: retrieve text + images from ChromaDB, generate with Gemma 4 E4B-IT
"""

import os
import sys
import base64
from pathlib import Path

import chromadb
import ollama

import config


# ── System prompt ───────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant with access to the user's downloaded files.
When answering questions, use the retrieved context provided below.
If images are included in the context, describe what you see and relate it to the question.
If the context doesn't contain relevant information, say so honestly.
Always cite which file the information came from when possible."""


# ── Retrieval ───────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """Embed a user query using nomic-embed-text with search_query prefix."""
    response = ollama.embed(
        model=config.TEXT_EMBED_MODEL,
        input=[f"{config.TEXT_EMBED_PREFIX_QUERY}{query}"]
    )
    return response["embeddings"][0]


def retrieve(collection, query: str) -> list[dict]:
    """Retrieve results from ChromaDB by querying text and images SEPARATELY,
    then merging. This prevents text from always dominating images in ranking.
    """
    query_embedding = embed_query(query)

    # Query text chunks separately
    text_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config.TOP_K_TEXT,
        where={"type": "text"},
        include=["documents", "metadatas", "distances"]
    )

    # Query images separately
    image_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config.TOP_K_IMAGES,
        where={"type": "image"},
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    # Process text results
    for i in range(len(text_results["ids"][0])):
        retrieved.append({
            "id": text_results["ids"][0][i],
            "document": text_results["documents"][0][i],
            "metadata": text_results["metadatas"][0][i],
            "distance": text_results["distances"][0][i],
            "similarity": 1 - text_results["distances"][0][i]
        })

    # Process image results
    for i in range(len(image_results["ids"][0])):
        retrieved.append({
            "id": image_results["ids"][0][i],
            "document": image_results["documents"][0][i],
            "metadata": image_results["metadatas"][0][i],
            "distance": image_results["distances"][0][i],
            "similarity": 1 - image_results["distances"][0][i]
        })

    return retrieved


# ── Build prompt with retrieved context ─────────────────

def build_messages(query: str, retrieved: list[dict],
                   history: list[dict]) -> list[dict]:
    """Build the message list for Gemma 4 with:
    - System prompt + retrieved text context
    - Conversation history
    - Current user message (with images if retrieved)
    """
    # Build context block from retrieved text chunks
    text_context_parts = []
    image_paths = []

    for item in retrieved:
        if item["metadata"]["type"] == "text":
            filename = item["metadata"]["filename"]
            sim = item["similarity"]
            text_context_parts.append(
                f"[From: {filename} | Relevance: {sim:.2f}]\n{item['document']}"
            )
        elif item["metadata"]["type"] == "image":
            source = item["metadata"]["source"]
            if os.path.exists(source):
                image_paths.append(source)
                text_context_parts.append(
                    f"[Image: {item['metadata']['filename']} | Relevance: {item['similarity']:.2f}] "
                    f"(image attached below)"
                )

    context_block = "\n\n---\n\n".join(text_context_parts) if text_context_parts else "(no relevant context found)"

    # System message with context
    system_content = f"{SYSTEM_PROMPT}\n\n--- RETRIEVED CONTEXT ---\n\n{context_block}"

    messages = [{"role": "system", "content": system_content}]

    # Add conversation history
    messages.extend(history)

    # Current user message — attach images if any were retrieved
    user_message = {"role": "user", "content": query}

    if image_paths:
        # Ollama multimodal: pass image paths directly
        # Ollama handles base64 encoding internally when given file paths
        user_message["images"] = image_paths

    messages.append(user_message)

    return messages


# ── Generation ──────────────────────────────────────────

def generate(messages: list[dict]) -> str:
    """Generate response using Gemma 4 E4B-IT via Ollama."""
    response = ollama.chat(
        model=config.GENERATION_MODEL,
        messages=messages,
        options={
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "top_k": config.TOP_K,
        }
    )
    return response["message"]["content"]


# ── Conversation loop ──────────────────────────────────

def chat():
    print("=" * 60)
    print("Phase 8 — Conversational Multimodal RAG")
    print("Gemma 4 E4B-IT + nomic-embed + ChromaDB")
    print("=" * 60)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
        count = collection.count()
        print(f"\nLoaded collection: {count} items indexed")
    except Exception:
        print("\n⚠ No collection found. Run ingest.py first.")
        print(f"  python ingest.py")
        return

    # Conversation history
    history: list[dict] = []

    print(f"\nReady. Ask questions about your downloaded files.")
    print(f"Commands: /clear (reset history), /count, /quit\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not query:
            continue

        # Commands
        if query.lower() == "/quit":
            print("Bye.")
            break
        elif query.lower() == "/clear":
            history.clear()
            print("History cleared.\n")
            continue
        elif query.lower() == "/count":
            print(f"Collection: {collection.count()} items, History: {len(history)} turns\n")
            continue

        # Retrieve
        print("  Retrieving...")
        retrieved = retrieve(collection, query)

        n_text = sum(1 for r in retrieved if r["metadata"]["type"] == "text")
        n_img = sum(1 for r in retrieved if r["metadata"]["type"] == "image")
        print(f"  Found {n_text} text chunks, {n_img} images")

        # Show sources
        seen_files = set()
        for r in retrieved:
            fn = r["metadata"]["filename"]
            if fn not in seen_files:
                seen_files.add(fn)
                print(f"    → {fn} ({r['similarity']:.2f})")

        # Build messages
        messages = build_messages(query, retrieved, history)

        # Generate
        print("  Generating...\n")
        try:
            response = generate(messages)
        except Exception as e:
            print(f"  ⚠ Generation error: {e}")
            print(f"  Make sure Gemma 4 is pulled: ollama pull gemma4:e4b\n")
            continue

        print(f"Gemma: {response}\n")

        # Update history (trim if needed)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})

        # Trim history if exceeds max turns
        if len(history) > config.MAX_HISTORY_TURNS * 2:
            history = history[-(config.MAX_HISTORY_TURNS * 2):]
            print("  (older history trimmed)\n")


if __name__ == "__main__":
    chat()
