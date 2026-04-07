"""
Phase 9: Memory layer for laptop-ai
Extracts facts and entity relationships from conversations.
Stores in ChromaDB (semantic search) + SQLite (entity graph).
Retrieves relevant memories before each inference.
"""

import sqlite3
import json
import time
import chromadb
import ollama

MEMORY_DB_PATH = "memory.db"
CHROMA_PATH = "chroma_db"
MEMORY_COLLECTION = "user_memory"
EMBED_MODEL = "nomic-embed-text"
EXTRACT_MODEL = "gemma4:e4b"
MAX_MEMORY_RESULTS = 5
SIMILARITY_THRESHOLD = 0.85
SESSION_ID = str(int(time.time()))


def init_db():
    conn = sqlite3.connect(MEMORY_DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
        chromadb_id TEXT, created_at REAL, session_id TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
        entity_type TEXT, first_seen REAL, last_seen REAL,
        mention_count INTEGER DEFAULT 1)""")
    c.execute("""CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_a TEXT NOT NULL, relationship_type TEXT NOT NULL,
        entity_b TEXT NOT NULL, first_seen REAL, last_seen REAL,
        confidence REAL DEFAULT 0.5,
        UNIQUE(entity_a, relationship_type, entity_b))""")
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, started_at REAL, summary TEXT)""")
    conn.commit()
    conn.close()


def get_memory_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=MEMORY_COLLECTION, metadata={"hnsw:space": "cosine"})


def extract_memories(question, answer):
    answer_trimmed = answer[:600] if len(answer) > 600 else answer

    prompt = """You are a memory extraction system. Extract information worth remembering for future conversations.

Rules:
- Only extract facts useful in a FUTURE conversation
- Skip greetings and small talk
- Extract entities (people, projects, companies) and relationships

Return ONLY valid JSON, no markdown, no backticks, no explanation:
{"facts": ["fact 1", "fact 2"], "entities": [{"name": "entity name", "type": "person/project/company/topic"}], "relationships": [{"entity_a": "name1", "relationship": "relationship type", "entity_b": "name2"}]}

Conversation:
User: """ + question + """
Assistant: """ + answer_trimmed

    response = ollama.generate(
        model=EXTRACT_MODEL, prompt=prompt,
        options={"temperature": 0.1, "num_predict": 500})

    raw = response["response"].strip()

    # Strip markdown code fences
    if "```" in raw:
        lines = raw.split("\n")
        cleaned = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(cleaned).strip()

    json_start = raw.find("{")
    json_end = raw.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        print("  [extraction: no JSON found]")
        return None

    json_str = raw[json_start:json_end]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  [extraction: JSON parse failed - {e}]")
        return None

    if "facts" not in parsed:
        parsed["facts"] = []
    if "entities" not in parsed:
        parsed["entities"] = []
    if "relationships" not in parsed:
        parsed["relationships"] = []

    parsed["facts"] = [f for f in parsed["facts"] if isinstance(f, str) and len(f.strip()) > 5]

    if not parsed["facts"] and not parsed["entities"] and not parsed["relationships"]:
        print("  [extraction: nothing worth remembering]")
        return None

    return parsed


def store_memories(extracted):
    if not extracted:
        return 0

    collection = get_memory_collection()
    conn = sqlite3.connect(MEMORY_DB_PATH)
    c = conn.cursor()
    now = time.time()
    stored_count = 0
    col_count = collection.count()

    for fact in extracted.get("facts", []):
        fact = fact.strip()
        if not fact or len(fact) < 5:
            continue

        try:
            embed_resp = ollama.embed(model=EMBED_MODEL, input=fact)
            fact_embedding = embed_resp["embeddings"][0]
        except Exception as e:
            print(f"  [embed error: {e}]")
            continue

        # Dedup check only if collection is not empty
        is_duplicate = False
        if col_count > 0:
            try:
                existing = collection.query(
                    query_embeddings=[fact_embedding], n_results=1)
                if (existing["distances"] and existing["distances"][0]
                    and len(existing["distances"][0]) > 0):
                    similarity = 1 - existing["distances"][0][0]
                    if similarity > SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        old_id = existing["ids"][0][0]
                        c.execute("UPDATE facts SET created_at = ? WHERE chromadb_id = ?",
                                  (now, old_id))
            except Exception as e:
                print(f"  [dedup error: {e}]")

        if not is_duplicate:
            fact_id = f"mem_{int(now)}_{stored_count}"
            try:
                collection.add(
                    ids=[fact_id], embeddings=[fact_embedding],
                    documents=[fact],
                    metadatas=[{"session_id": SESSION_ID, "created_at": now}])
                c.execute(
                    "INSERT INTO facts (text, chromadb_id, created_at, session_id) VALUES (?, ?, ?, ?)",
                    (fact, fact_id, now, SESSION_ID))
                stored_count += 1
                col_count += 1
            except Exception as e:
                print(f"  [store fact error: {e}]")

    for entity in extracted.get("entities", []):
        name = entity.get("name", "").strip()
        etype = entity.get("type", "unknown").strip()
        if not name:
            continue
        try:
            c.execute("""INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count)
                VALUES (?, ?, ?, ?, 1) ON CONFLICT(name) DO UPDATE SET
                last_seen = ?, mention_count = mention_count + 1""",
                (name, etype, now, now, now))
        except Exception as e:
            print(f"  [entity error: {e}]")

    for rel in extracted.get("relationships", []):
        ea = rel.get("entity_a", "").strip()
        rt = rel.get("relationship", "").strip()
        eb = rel.get("entity_b", "").strip()
        if not ea or not rt or not eb:
            continue
        try:
            c.execute("""INSERT INTO relationships
                (entity_a, relationship_type, entity_b, first_seen, last_seen, confidence)
                VALUES (?, ?, ?, ?, ?, 0.5)
                ON CONFLICT(entity_a, relationship_type, entity_b) DO UPDATE SET
                last_seen = ?, confidence = MIN(confidence + 0.1, 1.0)""",
                (ea, rt, eb, now, now, now))
        except Exception as e:
            print(f"  [relationship error: {e}]")

    try:
        summary = "; ".join(extracted.get("facts", [])[:3])
        c.execute("INSERT OR REPLACE INTO sessions (id, started_at, summary) VALUES (?, ?, ?)",
                  (SESSION_ID, now, summary))
    except Exception:
        pass

    conn.commit()
    conn.close()
    return stored_count


def retrieve_memories(query):
    parts = []
    collection = get_memory_collection()

    if collection.count() > 0:
        try:
            embed_resp = ollama.embed(model=EMBED_MODEL, input=query)
            results = collection.query(
                query_embeddings=[embed_resp["embeddings"][0]],
                n_results=min(MAX_MEMORY_RESULTS, collection.count()))
            if results["documents"] and results["documents"][0]:
                parts.append("Relevant memories from previous conversations:")
                for i, fact in enumerate(results["documents"][0], 1):
                    parts.append(f"  {i}. {fact}")
        except Exception:
            pass

    conn = sqlite3.connect(MEMORY_DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT name, entity_type FROM entities ORDER BY mention_count DESC LIMIT 20")
        all_entities = c.fetchall()
        query_upper = query.upper()
        relevant = [e for e in all_entities if e[0].upper() in query_upper]

        if relevant:
            rel_parts = []
            for name, etype in relevant:
                c.execute("""SELECT entity_a, relationship_type, entity_b
                    FROM relationships WHERE entity_a = ? OR entity_b = ?
                    ORDER BY confidence DESC LIMIT 5""", (name, name))
                for ea, rt, eb in c.fetchall():
                    rel_parts.append(f"  {ea} -> {rt} -> {eb}")
            if rel_parts:
                parts.append("Known entity relationships:")
                parts.extend(rel_parts)
        elif all_entities:
            c.execute("""SELECT entity_a, relationship_type, entity_b
                FROM relationships ORDER BY confidence DESC LIMIT 3""")
            rows = c.fetchall()
            if rows:
                parts.append("Known entity relationships:")
                for ea, rt, eb in rows:
                    parts.append(f"  {ea} -> {rt} -> {eb}")
    except Exception:
        pass
    finally:
        conn.close()

    return "\n".join(parts) if parts else ""


def get_last_session_summary():
    conn = sqlite3.connect(MEMORY_DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT summary, started_at FROM sessions WHERE id != ? ORDER BY started_at DESC LIMIT 1",
                  (SESSION_ID,))
        row = c.fetchone()
        if row and row[0]:
            ago = time.time() - row[1]
            if ago < 3600: t = f"{int(ago/60)} minutes ago"
            elif ago < 86400: t = f"{int(ago/3600)} hours ago"
            else: t = f"{int(ago/86400)} days ago"
            return f"last session ({t}): {row[0]}"
    except Exception:
        pass
    finally:
        conn.close()
    return None


def show_memories():
    conn = sqlite3.connect(MEMORY_DB_PATH)
    c = conn.cursor()
    print("\n" + "=" * 50)
    print("STORED MEMORIES")
    print("=" * 50)

    c.execute("SELECT text, created_at FROM facts ORDER BY created_at DESC")
    facts = c.fetchall()
    if facts:
        from datetime import datetime
        print(f"\nFacts ({len(facts)} total):")
        for text, ts in facts:
            dt = datetime.fromtimestamp(ts).strftime("%d-%m-%Y %H:%M")
            print(f"  - {text}  [{dt}]")
    else:
        print("\nNo facts stored yet.")

    c.execute("SELECT name, entity_type, mention_count FROM entities ORDER BY mention_count DESC")
    for name, etype, count in c.fetchall():
        print(f"  entity: {name} ({etype}) - mentioned {count}x")

    c.execute("SELECT entity_a, relationship_type, entity_b, confidence FROM relationships ORDER BY confidence DESC")
    rels = c.fetchall()
    if rels:
        print(f"\nRelationships ({len(rels)} total):")
        for ea, rt, eb, conf in rels:
            print(f"  - {ea} -> {rt} -> {eb}  [confidence: {conf:.1f}]")

    print("=" * 50 + "\n")
    conn.close()


def clear_memories():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(MEMORY_COLLECTION)
    except Exception:
        pass
    conn = sqlite3.connect(MEMORY_DB_PATH)
    c = conn.cursor()
    for t in ["facts", "entities", "relationships", "sessions"]:
        c.execute(f"DELETE FROM {t}")
    conn.commit()
    conn.close()
    print("All memories cleared.\n")


init_db()
