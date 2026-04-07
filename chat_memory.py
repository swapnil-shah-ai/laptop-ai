"""
Phase 9: Chat with memory
Combines Phase 8 document retrieval with persistent memory layer.
Your laptop remembers what you discussed across sessions.
"""

import chromadb
import ollama
from memory import (
    extract_memories, store_memories, retrieve_memories,
    get_last_session_summary, show_memories, clear_memories,
    CHROMA_PATH, EXTRACT_MODEL, EMBED_MODEL
)

DOC_COLLECTION = "laptop_files"
MODEL = EXTRACT_MODEL
TOP_K_DOCS = 5
MAX_HISTORY = 20
SYSTEM_BASE = """You are a helpful AI assistant running locally on a laptop.
You have access to the user's documents and memories from previous conversations.
Use the provided context to give accurate, relevant answers.
If memories from previous conversations are relevant, use them naturally.
Do not mention that you are retrieving from a database or memory system."""


def get_doc_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection(name=DOC_COLLECTION)
    except Exception:
        return None


def retrieve_documents(query):
    collection = get_doc_collection()
    if not collection or collection.count() == 0:
        return ""
    try:
        embed_resp = ollama.embed(model=EMBED_MODEL, input=query)
        results = collection.query(
            query_embeddings=[embed_resp["embeddings"][0]],
            n_results=min(TOP_K_DOCS, collection.count()))
        if results["documents"] and results["documents"][0]:
            parts = ["Relevant information from your documents:"]
            for i, chunk in enumerate(results["documents"][0], 1):
                text = chunk[:500] if len(chunk) > 500 else chunk
                parts.append(f"  [{i}] {text}")
            return "\n".join(parts)
    except Exception:
        pass
    return ""


def build_prompt(query, conversation_history):
    parts = [SYSTEM_BASE]

    memories = retrieve_memories(query)
    if memories:
        parts.append("")
        parts.append(memories)

    docs = retrieve_documents(query)
    if docs:
        parts.append("")
        parts.append(docs)

    messages = [{"role": "system", "content": "\n".join(parts)}]

    for turn in conversation_history[-MAX_HISTORY:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": query})
    return messages


def main():
    print("\n" + "=" * 55)
    print("  laptop-ai | phase 9 | chat with memory")
    print("  your laptop remembers what you discussed")
    print("=" * 55)

    doc_collection = get_doc_collection()
    if doc_collection:
        print(f"  documents: {doc_collection.count()} chunks indexed")
    else:
        print("  documents: none (run ingest.py from phase 8 first)")

    summary = get_last_session_summary()
    if summary:
        print(f"\n  {summary}")
    else:
        print("\n  no previous sessions found, starting fresh")

    print("\n  commands:")
    print("    /memory       - view stored memories")
    print("    /memory clear - wipe all memories")
    print("    /quit         - exit")
    print("=" * 55 + "\n")

    conversation_history = []

    while True:
        try:
            query = input("you: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\ngoodbye.\n")
            break

        if not query:
            continue

        if query.lower() == "/quit":
            print("\ngoodbye.\n")
            break

        if query.lower() == "/memory":
            show_memories()
            continue

        if query.lower() == "/memory clear":
            confirm = input("are you sure? this deletes all memories (y/n): ").strip().lower()
            if confirm == "y":
                clear_memories()
            continue

        # Step 1: build prompt with memories + documents
        messages = build_prompt(query, conversation_history)

        # Step 2: generate answer
        try:
            response = ollama.chat(
                model=MODEL, messages=messages,
                options={"num_predict": 1024, "temperature": 0.7})
            answer = response["message"]["content"]
        except Exception as e:
            print(f"\nerror generating response: {e}\n")
            continue

        print(f"\nassistant: {answer}\n")

        # Step 3: save to history
        conversation_history.append({"user": query, "assistant": answer})

        # Step 4: extract and store memories
        print("  [extracting memories...]")
        try:
            extracted = extract_memories(query, answer)
            if extracted:
                count = store_memories(extracted)
                if count > 0:
                    print(f"  [{count} new memory stored]\n")
                else:
                    print("  [memories already known]\n")
            else:
                print("  [no new memories from this exchange]\n")
        except Exception as e:
            print(f"  [memory error: {e}]\n")


if __name__ == "__main__":
    main()
