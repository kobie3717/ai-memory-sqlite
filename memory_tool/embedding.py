"""Vector embedding and semantic search functionality."""

import sys
import sqlite3
import math
from typing import Optional, List, Tuple, Any, Iterable
from .config import MODEL_DIR, EMBEDDING_DIM, RRF_K, get_logger

logger = get_logger(__name__)

# Lazy imports for optional dependencies
_EMBEDDING_MODEL = None

try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    _VEC_LIBS_AVAILABLE = True
except ImportError:
    _VEC_LIBS_AVAILABLE = False


def sanitize_and_normalize_embedding(vec: Iterable[float]) -> List[float]:
    """
    Defensive pre-cosine: replace NaN/Inf with 0.0, then L2-normalize.

    Empty or all-zero vectors return as-is (no division by zero).
    Returns a fresh list.
    """
    cleaned = [
        0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v)
        for v in vec
    ]
    norm = math.sqrt(sum(v * v for v in cleaned))
    if norm == 0.0:
        return cleaned
    return [v / norm for v in cleaned]


def has_vec_support() -> bool:
    """Check if vector search dependencies are available."""
    # Import from database module to keep single source of truth
    from .database import has_vec_support as db_has_vec_support
    return db_has_vec_support()


def get_embedding_model() -> Optional[Tuple[Any, Any]]:
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
        tokenizer.enable_truncation(max_length=512)

        session = ort.InferenceSession(
            str(MODEL_DIR / "onnx" / "model.onnx"),
            providers=['CPUExecutionProvider']
        )

        _EMBEDDING_MODEL = (tokenizer, session)
        return _EMBEDDING_MODEL
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        return None


def embed_text(text: str) -> Optional[bytes]:
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

        # Run inference — only pass token_type_ids if model declares it
        session_inputs = {i.name for i in session.get_inputs()}
        feed = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'token_type_ids' in session_inputs:
            feed['token_type_ids'] = token_type_ids
        outputs = session.run(None, feed)

        # Mean pooling
        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
        embeddings = summed / counts

        # Defensive scrub before normalize: replace NaN/Inf with 0
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)  # Prevent division by zero
        embeddings = (embeddings / norms).astype(np.float32)

        # Return as bytes
        return embeddings[0].tobytes()
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def embed_texts_batch(texts: List[str]) -> List[Optional[bytes]]:
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

        # Run inference — only pass token_type_ids if model declares it
        session_inputs = {i.name for i in session.get_inputs()}
        feed = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'token_type_ids' in session_inputs:
            feed['token_type_ids'] = token_type_ids
        outputs = session.run(None, feed)

        # Mean pooling
        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
        embeddings = summed / counts

        # Defensive scrub before normalize: replace NaN/Inf with 0
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)  # Prevent division by zero
        embeddings = (embeddings / norms).astype(np.float32)

        # Return as list of bytes
        return [emb.tobytes() for emb in embeddings]
    except Exception as e:
        logger.warning(f"Batch embedding failed: {e}")
        return [None] * len(texts)


def embed_and_store(conn: sqlite3.Connection, mem_id: int, content: str) -> None:
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


def semantic_search(conn: sqlite3.Connection, query: str, limit: int = 20) -> List[sqlite3.Row]:
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


def reindex_embeddings(conn: sqlite3.Connection) -> None:
    """Bulk-embed all active memories for vector search."""
    if not has_vec_support():
        logger.error("Vector search not available. Install: pip install sqlite-vec onnxruntime tokenizers numpy")
        logger.error(f"Also download model files to {MODEL_DIR}")
        return

    rows = conn.execute("""
        SELECT id, content FROM memories WHERE active = 1
        ORDER BY id
    """).fetchall()

    if not rows:
        logger.info("No active memories to index.")
        return

    logger.info(f"Reindexing {len(rows)} memories...")
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
                    # vec0 virtual tables don't support INSERT OR REPLACE; delete first
                    conn.execute("DELETE FROM memory_vec WHERE rowid = ?", (mem_id,))
                    conn.execute(
                        "INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)",
                        (mem_id, emb)
                    )
                    total += 1
                except sqlite3.Error as e:
                    logger.warning(f"Failed to store embedding for #{mem_id}: {e}")

        if (i + batch_size) % 100 == 0:
            logger.debug(f"  Processed {min(i + batch_size, len(rows))}/{len(rows)}...")

    conn.commit()
    logger.info(f"Reindexing complete. {total} embeddings created.")
