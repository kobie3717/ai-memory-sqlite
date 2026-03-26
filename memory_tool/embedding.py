"""Vector embedding and semantic search functionality."""

import sys
import sqlite3
from .config import MODEL_DIR, EMBEDDING_DIM, RRF_K

# Lazy imports for optional dependencies
_EMBEDDING_MODEL = None

try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    _VEC_LIBS_AVAILABLE = True
except ImportError:
    _VEC_LIBS_AVAILABLE = False


def has_vec_support():
    """Check if vector search dependencies are available."""
    # Import from database module to keep single source of truth
    from .database import has_vec_support as db_has_vec_support
    return db_has_vec_support()


def get_embedding_model():
    """Lazy-load the embedding model (singleton)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    if not _VEC_LIBS_AVAILABLE:
        return None

    # Check if model files exist
    if not MODEL_DIR.exists():
        return None

    required_files = [
        MODEL_DIR / "tokenizer.json",
        MODEL_DIR / "onnx" / "model.onnx"
    ]
    if not all(f.exists() for f in required_files):
        return None

    try:
        tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        tokenizer.enable_truncation(max_length=256)

        session = ort.InferenceSession(
            str(MODEL_DIR / "onnx" / "model.onnx"),
            providers=['CPUExecutionProvider']
        )

        _EMBEDDING_MODEL = (tokenizer, session)
        return _EMBEDDING_MODEL
    except Exception as e:
        print(f"Warning: Failed to load embedding model: {e}", file=sys.stderr)
        return None


def embed_text(text):
    """Generate embedding for a single text string. Returns bytes for sqlite-vec."""
    model = get_embedding_model()
    if model is None:
        return None

    tokenizer, session = model

    try:
        # Encode text
        encoding = tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Run inference
        outputs = session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        })

        # Mean pooling
        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
        embeddings = summed / counts

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / norms).astype(np.float32)

        # Return as bytes
        return embeddings[0].tobytes()
    except Exception as e:
        print(f"Warning: Embedding failed: {e}", file=sys.stderr)
        return None


def embed_texts_batch(texts):
    """Generate embeddings for multiple texts. Returns list of bytes."""
    model = get_embedding_model()
    if model is None:
        return [None] * len(texts)

    tokenizer, session = model

    try:
        # Encode batch
        encodings = tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Run inference
        outputs = session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        })

        # Mean pooling
        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
        embeddings = summed / counts

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / norms).astype(np.float32)

        # Return as list of bytes
        return [emb.tobytes() for emb in embeddings]
    except Exception as e:
        print(f"Warning: Batch embedding failed: {e}", file=sys.stderr)
        return [None] * len(texts)


def embed_and_store(conn, mem_id, content):
    """Generate embedding for content and store in vector table."""
    if not has_vec_support():
        return

    embedding = embed_text(content)
    if embedding is None:
        return

    try:
        # Insert or replace embedding (rowid must match memory id)
        conn.execute(
            "INSERT OR REPLACE INTO memory_vec(rowid, embedding) VALUES (?, ?)",
            (mem_id, embedding)
        )
    except Exception as e:
        # Silently fail if vec table doesn't exist
        pass


def semantic_search(conn, query, limit=20):
    """Perform semantic vector search."""
    if not has_vec_support():
        return []

    query_embedding = embed_text(query)
    if query_embedding is None:
        return []

    try:
        results = conn.execute("""
            SELECT m.*, vec_distance_cosine(v.embedding, ?) as distance
            FROM memory_vec v
            JOIN memories m ON m.id = v.rowid
            WHERE m.active = 1
            ORDER BY distance
            LIMIT ?
        """, (query_embedding, limit)).fetchall()
        return results
    except Exception as e:
        return []


def reindex_embeddings(conn):
    """Bulk-embed all active memories for vector search."""
    if not has_vec_support():
        print("Vector search not available. Install: pip install sqlite-vec onnxruntime tokenizers numpy")
        print(f"Also download model files to {MODEL_DIR}")
        return

    rows = conn.execute("""
        SELECT id, content FROM memories WHERE active = 1
        ORDER BY id
    """).fetchall()

    if not rows:
        print("No active memories to index.")
        return

    print(f"Reindexing {len(rows)} memories...")
    batch_size = 32
    total = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = [r["content"] for r in batch]
        embeddings = embed_texts_batch(texts)

        for j, emb in enumerate(embeddings):
            if emb is not None:
                mem_id = batch[j]["id"]
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO memory_vec(rowid, embedding) VALUES (?, ?)",
                        (mem_id, emb)
                    )
                    total += 1
                except sqlite3.Error:
                    pass  # Skip this embedding if insert fails

        if (i + batch_size) % 100 == 0:
            print(f"  Processed {min(i + batch_size, len(rows))}/{len(rows)}...")

    conn.commit()
    print(f"Reindexing complete. {total} embeddings created.")
