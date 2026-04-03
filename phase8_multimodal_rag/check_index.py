import chromadb
import config

client = chromadb.PersistentClient(path=config.CHROMA_DIR)
col = client.get_collection(config.COLLECTION_NAME)
results = col.get(where={"type": "text"}, include=["metadatas"])
files = set(m["filename"] for m in results["metadatas"])
unapp = [f for f in sorted(files) if "un" in f.lower() or "app" in f.lower() or "pitch" in f.lower()]

print(f"Total files indexed: {len(files)}")
print(f"\nun-app related files found:")
for f in unapp:
    count = sum(1 for m in results["metadatas"] if m["filename"] == f)
    print(f"  {f} ({count} chunks)")

if not unapp:
    print("  NONE found.")
