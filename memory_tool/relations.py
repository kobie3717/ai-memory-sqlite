"""Memory relationship and conflict management."""

import sqlite3
import sys
import os
import re
import json
import shutil
import subprocess
import hashlib
import math
from datetime import datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher

# Import from our modular components
from .config import *
from .database import get_db, has_vec_support
from .utils import auto_tag, word_set, normalize, find_similar, word_overlap, similarity
from .fsrs import fsrs_retention, fsrs_new_stability, fsrs_new_difficulty, fsrs_next_interval, fsrs_auto_rating
from .importance import update_importance
from .embedding import embed_and_store, embed_text, semantic_search

# Lazy imports for optional dependencies
try:
    import numpy as np
    import sqlite_vec
except ImportError:
    pass


# Lazy import to avoid circular dependency
def _get_export_memory_md():
    """Lazy import of export_memory_md to avoid circular dependency."""
    from .export import export_memory_md
    return export_memory_md


def relate_memories(id1, id2, relation_type="related"):
    conn = get_db()
    try:
        conn.execute("INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, ?)",
                     (id1, id2, relation_type))
        conn.execute("INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, ?)",
                     (id2, id1, relation_type))
        conn.commit()
        print(f"Linked #{id1} <-> #{id2} ({relation_type})")
    except sqlite3.IntegrityError as e:
        print(f"Failed: {e}")
    conn.close()




def get_related(mem_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT m.*, mr.relation_type FROM memories m
        JOIN memory_relations mr ON m.id = mr.target_id
        WHERE mr.source_id = ? AND m.active = 1
    """, (mem_id,)).fetchall()
    conn.close()
    return rows


# ── Conflict Detection (v4 Feature #3) ──



def find_conflicts():
    """Find memories with 50-85% similarity (potential conflicts)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, content, category, project FROM memories WHERE active = 1 ORDER BY project, category"
    ).fetchall()
    conn.close()

    conflicts = []
    seen = set()

    for i, row1 in enumerate(rows):
        for row2 in rows[i+1:]:
            pair_key = tuple(sorted([row1["id"], row2["id"]]))
            if pair_key in seen:
                continue

            words1 = word_set(row1["content"])
            words2 = word_set(row2["content"])
            if not words1 or not words2:
                continue

            intersection = words1 & words2
            union = words1 | words2
            jaccard = len(intersection) / len(union) if union else 0
            seq_score = SequenceMatcher(None, normalize(row1["content"]), normalize(row2["content"])).ratio()
            score = max(jaccard, seq_score)

            # Only report 50-85% (below dedup threshold but suspicious)
            if 0.50 <= score < 0.85:
                conflicts.append({
                    "id1": row1["id"],
                    "id2": row2["id"],
                    "content1": row1["content"],
                    "content2": row2["content"],
                    "score": score,
                    "project": row1["project"] or row2["project"] or "Unknown",
                    "category": f"{row1['category']}/{row2['category']}" if row1["category"] != row2["category"] else row1["category"],
                })
                seen.add(pair_key)

    return sorted(conflicts, key=lambda x: (-x["score"], x["project"]))




def merge_memories(id1, id2):
    """Merge two memories: keep newer, deactivate older, merge tags and relations."""
    conn = get_db()
    mem1 = conn.execute("SELECT * FROM memories WHERE id = ?", (id1,)).fetchone()
    mem2 = conn.execute("SELECT * FROM memories WHERE id = ?", (id2,)).fetchone()

    if not mem1 or not mem2:
        print("One or both memories not found.")
        conn.close()
        return

    # Determine newer (higher updated_at)
    if mem1["updated_at"] >= mem2["updated_at"]:
        keep_id, discard_id = id1, id2
        keep_mem, discard_mem = mem1, mem2
    else:
        keep_id, discard_id = id2, id1
        keep_mem, discard_mem = mem2, mem1

    # Merge tags
    tags1 = set(filter(None, keep_mem["tags"].split(",")))
    tags2 = set(filter(None, discard_mem["tags"].split(",")))
    merged_tags = ",".join(sorted(tags1 | tags2))

    # Update keeper
    conn.execute(
        "UPDATE memories SET tags = ?, updated_at = datetime('now') WHERE id = ?",
        (merged_tags, keep_id)
    )

    # Deactivate discarded
    conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (discard_id,))

    # Transfer relations from discarded to keeper
    conn.execute("""
        UPDATE memory_relations SET source_id = ?
        WHERE source_id = ? AND target_id != ?
    """, (keep_id, discard_id, keep_id))
    conn.execute("""
        UPDATE memory_relations SET target_id = ?
        WHERE target_id = ? AND source_id != ?
    """, (keep_id, discard_id, keep_id))

    # Remove any duplicate relations
    conn.execute("DELETE FROM memory_relations WHERE source_id = target_id")

    conn.commit()
    conn.close()
    _get_export_memory_md()()
    print(f"Merged #{discard_id} into #{keep_id} (deactivated #{discard_id})")




def supersede_memory(old_id, new_id):
    """Mark old memory as superseded by new."""
    conn = get_db()

    # Verify both exist
    old = conn.execute("SELECT id FROM memories WHERE id = ?", (old_id,)).fetchone()
    new = conn.execute("SELECT id FROM memories WHERE id = ?", (new_id,)).fetchone()

    if not old or not new:
        print("One or both memories not found.")
        conn.close()
        return

    # Deactivate old
    conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (old_id,))

    # Create supersedes relation
    conn.execute(
        "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'supersedes')",
        (new_id, old_id)
    )

    conn.commit()
    conn.close()
    _get_export_memory_md()()
    print(f"#{new_id} supersedes #{old_id} (deactivated #{old_id})")


# ── Topic File Export (v4 Feature #5) ──


