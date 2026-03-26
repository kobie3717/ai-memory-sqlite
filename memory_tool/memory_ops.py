"""Core CRUD operations for memories."""

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


# Lazy imports to avoid circular dependencies
def _get_export_memory_md():
    """Lazy import of export_memory_md to avoid circular dependency."""
    from .export import export_memory_md
    return export_memory_md


def _get_auto_link_memory():
    """Lazy import of auto_link_memory to avoid circular dependency."""
    from .graph import auto_link_memory
    return auto_link_memory


def touch_memory(conn, mem_id):
    """Update access tracking and FSRS state when a memory is accessed."""
    # First update basic access tracking
    conn.execute("""
        UPDATE memories SET
            accessed_at = datetime('now'),
            access_count = access_count + 1,
            stale = 0
        WHERE id = ?
    """, (mem_id,))

    # FSRS update on access
    row = conn.execute("""
        SELECT fsrs_stability, fsrs_difficulty, fsrs_reps, last_accessed_at,
               access_count, priority, category, updated_at
        FROM memories WHERE id = ?
    """, (mem_id,)).fetchone()

    if row:
        old_s = row["fsrs_stability"] or 1.0
        old_d = row["fsrs_difficulty"] or 5.0
        reps = (row["fsrs_reps"] or 0)
        last_acc = row["last_accessed_at"]

        # Calculate days since last access
        if last_acc:
            try:
                last_dt = datetime.fromisoformat(last_acc.replace('Z', '+00:00'))
                elapsed = (datetime.now() - last_dt.replace(tzinfo=None)).total_seconds() / 86400
            except (ValueError, AttributeError):
                elapsed = 1.0  # Fallback for invalid date format
        else:
            elapsed = 1.0

        rating = fsrs_auto_rating(row["category"], row["access_count"], row["priority"])
        new_s = fsrs_new_stability(old_s, old_d, rating, elapsed)
        new_d = fsrs_new_difficulty(old_d, rating)
        new_interval = fsrs_next_interval(new_s)

        conn.execute("""
            UPDATE memories SET
                fsrs_stability = ?, fsrs_difficulty = ?, fsrs_interval = ?,
                fsrs_reps = ?, last_accessed_at = datetime('now')
            WHERE id = ?
        """, (new_s, new_d, new_interval, reps + 1, mem_id))

        # Update importance after FSRS update
        update_importance(mem_id, conn)




def auto_adjust_priority(conn, mem_id):
    row = conn.execute(
        "SELECT access_count, priority FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()
    if row:
        suggested = min(10, row["access_count"] // 5)
        if suggested > row["priority"]:
            conn.execute("UPDATE memories SET priority = ? WHERE id = ?", (suggested, mem_id))


# ── Smart Ingest (v4 Feature #4) ──



def smart_ingest(category, content, tags="", project=None, priority=0, related_to=None,
                 expires_at=None, source="manual", topic_key=None, derived_from=None,
                 citations=None, reasoning=None):
    """
    Smart ingestion with 4-tier similarity handling:
    - SKIP: >85% (duplicate blocked)
    - UPDATE: 70-85% same category/project (auto-update existing)
    - SUPERSEDE: 50-70% same project (insert new, mark old superseded)
    - CREATE: <50% (normal insert)
    """
    tags = auto_tag(content, tags)

    # Check for topic_key upsert
    if topic_key:
        conn = get_db()
        existing = conn.execute(
            "SELECT id, tags, revision_count FROM memories WHERE topic_key = ? AND active = 1",
            (topic_key,)
        ).fetchone()

        if existing:
            # Upsert: update content, merge tags, bump revision
            existing_tags = set(filter(None, existing["tags"].split(",")))
            new_tags = set(filter(None, tags.split(",")))
            merged_tags = ",".join(sorted(existing_tags | new_tags))
            new_revision = existing["revision_count"] + 1

            conn.execute("""
                UPDATE memories SET
                    content = ?,
                    tags = ?,
                    updated_at = datetime('now'),
                    revision_count = ?,
                    stale = 0
                WHERE id = ?
            """, (content, merged_tags, new_revision, existing["id"]))
            touch_memory(conn, existing["id"])
            embed_and_store(conn, existing["id"], content)
            conn.commit()
            conn.close()
            _get_export_memory_md()()
            print(f"Updated memory #{existing['id']} (revision {new_revision}) key:{topic_key}")
            return existing["id"]
        else:
            # New topic_key, insert normally
            conn.close()
            # Fall through to normal insert with topic_key set

    # Similarity-based dedup/smart-ingest
    similar = find_similar(content, category, project, threshold=0.5)

    if similar:
        best_id, best_content, score, best_cat, best_proj = similar[0]

        # SKIP: >85% (blocked)
        if score > 0.85:
            print(f"DUPLICATE BLOCKED (score={score:.0%}): similar to #{best_id}")
            print(f"  Existing: {best_content}")
            print(f"  Use 'memory-tool update {best_id} \"{content}\"' to update instead.")
            return None

        # UPDATE: 70-85% same category and project
        elif score > 0.70 and category == best_cat and project == best_proj:
            conn = get_db()
            existing = conn.execute(
                "SELECT tags, revision_count FROM memories WHERE id = ?", (best_id,)
            ).fetchone()
            existing_tags = set(filter(None, existing["tags"].split(",")))
            new_tags = set(filter(None, tags.split(",")))
            merged_tags = ",".join(sorted(existing_tags | new_tags))
            new_revision = existing["revision_count"] + 1

            conn.execute("""
                UPDATE memories SET
                    content = ?,
                    tags = ?,
                    updated_at = datetime('now'),
                    revision_count = ?,
                    stale = 0
                WHERE id = ?
            """, (content, merged_tags, new_revision, best_id))
            touch_memory(conn, best_id)
            embed_and_store(conn, best_id, content)
            conn.commit()
            conn.close()
            _get_export_memory_md()()
            print(f"AUTO-UPDATED memory #{best_id} ({score:.0%} match, revision {new_revision})")
            return best_id

        # SUPERSEDE: 50-70% same project
        elif score > 0.50 and project == best_proj:
            # Insert new, mark old as superseded
            conn = get_db()
            cur = conn.execute(
                """INSERT INTO memories (category, content, tags, project, priority, accessed_at, expires_at, source, topic_key, derived_from, citations, reasoning)
                   VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)""",
                (category, content, tags, project, priority, expires_at, source, topic_key, derived_from, citations, reasoning)
            )
            new_id = cur.lastrowid

            # Deactivate old
            conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (best_id,))

            # Create supersedes relation
            conn.execute(
                "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'supersedes')",
                (new_id, best_id)
            )

            if related_to:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'related')",
                        (new_id, int(related_to))
                    )
                except (ValueError, sqlite3.IntegrityError):
                    pass

            embed_and_store(conn, new_id, content)
            update_importance(new_id, conn)
            conn.commit()
            conn.close()
            _get_export_memory_md()()
            print(f"Added memory #{new_id}, supersedes #{best_id} ({score:.0%} overlap, different content)")
            return new_id

        # CREATE with warning: <50% or different category/project
        else:
            print(f"WARNING: Similar memory (score={score:.0%}): #{best_id}: {best_content}")

    # CREATE: Normal insert
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO memories (category, content, tags, project, priority, accessed_at, expires_at, source, topic_key, derived_from, citations, reasoning)
           VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)""",
        (category, content, tags, project, priority, expires_at, source, topic_key, derived_from, citations, reasoning)
    )
    mem_id = cur.lastrowid

    if related_to:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'related')",
                (mem_id, int(related_to))
            )
        except (ValueError, sqlite3.IntegrityError):
            pass

    embed_and_store(conn, mem_id, content)
    update_importance(mem_id, conn)
    conn.commit()
    conn.close()

    # Phase 3: Auto-link to graph entities
    _get_auto_link_memory()(mem_id, content)

    _get_export_memory_md()()
    key_str = f" key:{topic_key}" if topic_key else ""
    print(f"Added memory #{mem_id} [{category}]{key_str}{' tags:' + tags if tags else ''}")
    return mem_id




def add_memory(category, content, tags="", project=None, priority=0, related_to=None,
               expires_at=None, source="manual", topic_key=None, skip_dedup=False,
               derived_from=None, citations=None, reasoning=None):
    """Legacy add_memory wrapper for backward compatibility."""
    if skip_dedup:
        # Old behavior: skip dedup entirely
        tags = auto_tag(content, tags)
        conn = get_db()
        cur = conn.execute(
            """INSERT INTO memories (category, content, tags, project, priority, accessed_at, expires_at, source, topic_key, derived_from, citations, reasoning)
               VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)""",
            (category, content, tags, project, priority, expires_at, source, topic_key, derived_from, citations, reasoning)
        )
        mem_id = cur.lastrowid
        if related_to:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'related')",
                    (mem_id, int(related_to))
                )
            except (ValueError, sqlite3.IntegrityError):
                pass
        embed_and_store(conn, mem_id, content)
        update_importance(mem_id, conn)
        conn.commit()
        conn.close()

        # Phase 3: Auto-link to graph entities
        _get_auto_link_memory()(mem_id, content)

        _get_export_memory_md()()
        print(f"Added memory #{mem_id} [{category}]{' tags:' + tags if tags else ''}")
        return mem_id
    else:
        return smart_ingest(category, content, tags, project, priority, related_to, expires_at, source, topic_key, derived_from, citations, reasoning)




def search_memories(query, mode="hybrid"):
    """
    Search memories with multiple modes:
    - hybrid: Combine FTS and vector search with RRF (default)
    - keyword: FTS only
    - semantic: Vector only
    """
    conn = get_db()
    fts_results = []
    vec_results = []

    # 1. FTS keyword search
    if mode in ("hybrid", "keyword"):
        try:
            rows = conn.execute("""
                SELECT m.id FROM memories m
                JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ? AND m.active = 1
                ORDER BY rank LIMIT 20
            """, (query,)).fetchall()
            fts_results = [(r['id'], i) for i, r in enumerate(rows)]
        except sqlite3.OperationalError:
            pass

    # 2. Vector semantic search
    if mode in ("hybrid", "semantic") and has_vec_support():
        query_vec = embed_text(query)
        if query_vec is not None:
            try:
                # Get vec results (sqlite-vec requires k parameter)
                rows = conn.execute("""
                    SELECT rowid as id, distance FROM memory_vec
                    WHERE embedding MATCH ?
                    AND k = 20
                    ORDER BY distance
                """, (query_vec,)).fetchall()

                # Filter to active only
                active_ids = set(r['id'] for r in conn.execute(
                    "SELECT id FROM memories WHERE active = 1"
                ).fetchall())
                vec_results = [(r['id'], i) for i, r in enumerate(rows) if r['id'] in active_ids]
            except Exception:
                # Silently fail if vec table doesn't exist yet
                pass

    # 3. Reciprocal Rank Fusion (combine scores)
    if mode == "hybrid" and (fts_results or vec_results):
        scores = {}
        for mem_id, rank in fts_results:
            scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (RRF_K + rank + 1)
        for mem_id, rank in vec_results:
            scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (RRF_K + rank + 1)

        # Sort by combined RRF score
        ranked_ids = sorted(scores.keys(), key=lambda x: -scores[x])[:20]

        # Fetch full rows
        if ranked_ids:
            placeholders = ','.join('?' * len(ranked_ids))
            rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", ranked_ids).fetchall()
            # Re-sort by RRF score
            id_to_row = {r['id']: r for r in rows}
            rows = [id_to_row[mid] for mid in ranked_ids if mid in id_to_row]
        else:
            rows = []
    elif mode == "keyword" and fts_results:
        # Keyword-only mode: use FTS results
        mem_ids = [mid for mid, _ in fts_results]
        placeholders = ','.join('?' * len(mem_ids))
        rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", mem_ids).fetchall()
        id_to_row = {r['id']: r for r in rows}
        rows = [id_to_row[mid] for mid in mem_ids if mid in id_to_row]
    elif mode == "semantic" and vec_results:
        # Semantic-only mode: use vector results
        mem_ids = [mid for mid, _ in vec_results]
        placeholders = ','.join('?' * len(mem_ids))
        rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", mem_ids).fetchall()
        id_to_row = {r['id']: r for r in rows}
        rows = [id_to_row[mid] for mid in mem_ids if mid in id_to_row]
    else:
        rows = []

    # Fallback to LIKE if no results
    if not rows:
        rows = conn.execute("""
            SELECT * FROM memories
            WHERE active = 1 AND (content LIKE ? OR tags LIKE ? OR project LIKE ?)
            ORDER BY updated_at DESC LIMIT 20
        """, (f"%{query}%", f"%{query}%", f"%{query}%")).fetchall()

    # Touch accessed memories
    for r in rows:
        touch_memory(conn, r["id"])
        auto_adjust_priority(conn, r["id"])
    conn.commit()
    conn.close()
    return rows




def get_memory(mem_id):
    """Get full detail for a single memory."""
    conn = get_db()
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    conn.close()
    return row




def list_memories(category=None, project=None, tag=None, stale_only=False, expired_only=False):
    conn = get_db()
    query = "SELECT * FROM memories WHERE active = 1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if project:
        query += " AND project = ?"
        params.append(project)
    if tag:
        query += " AND tags LIKE ?"
        params.append(f"%{tag}%")
    if stale_only:
        query += " AND stale = 1"
    if expired_only:
        query += " AND expires_at IS NOT NULL AND expires_at < datetime('now')"
    query += " ORDER BY priority DESC, updated_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows




def update_memory(mem_id, content):
    conn = get_db()
    # Auto-tag the new content
    existing = conn.execute("SELECT tags, revision_count FROM memories WHERE id = ?", (mem_id,)).fetchone()
    tags = auto_tag(content, existing["tags"] if existing else "")
    new_revision = existing["revision_count"] + 1 if existing else 1
    conn.execute(
        "UPDATE memories SET content = ?, tags = ?, updated_at = datetime('now'), revision_count = ?, stale = 0 WHERE id = ?",
        (content, tags, new_revision, mem_id)
    )
    touch_memory(conn, mem_id)
    embed_and_store(conn, mem_id, content)
    update_importance(mem_id, conn)
    conn.commit()
    conn.close()
    _get_export_memory_md()()
    print(f"Updated memory #{mem_id} (revision {new_revision})")




def delete_memory(mem_id):
    conn = get_db()
    conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (mem_id,))
    conn.commit()
    conn.close()
    _get_export_memory_md()()
    print(f"Deactivated memory #{mem_id}")




def tag_memory(mem_id, tags):
    conn = get_db()
    existing = conn.execute("SELECT tags FROM memories WHERE id = ?", (mem_id,)).fetchone()
    if existing:
        current = set(filter(None, existing["tags"].split(",")))
        new_tags = set(filter(None, tags.split(",")))
        merged = ",".join(sorted(current | new_tags))
        conn.execute("UPDATE memories SET tags = ?, updated_at = datetime('now') WHERE id = ?", (merged, mem_id))
        conn.commit()
        print(f"Tagged memory #{mem_id}: {merged}")
    conn.close()
    _get_export_memory_md()()


# ── Relationships ──



def show_importance_ranking():
    """Show all memories ranked by importance score."""
    conn = get_db()

    # First recalculate all
    rows = conn.execute("SELECT id FROM memories WHERE active = 1").fetchall()
    for r in rows:
        update_importance(r["id"], conn)
    conn.commit()

    # Now display top and bottom
    rows = conn.execute("""
        SELECT id, category, project, content, imp_novelty, imp_relevance,
               imp_frequency, imp_impact, imp_score
        FROM memories WHERE active = 1
        ORDER BY imp_score DESC
    """).fetchall()
    conn.close()

    print(f"Importance Ranking ({len(rows)} memories)")
    print("=" * 70)
    print(f"  {'#':>3} {'Score':>5} {'N':>3} {'R':>3} {'F':>3} {'I':>3} {'Cat':<8} Content")
    print(f"  {'':>3} {'':>5} {'ov':>3} {'el':>3} {'rq':>3} {'mp':>3}")
    print("-" * 70)

    # Show top 15
    for r in rows[:15]:
        content = r["content"][:45]
        cat = (r["category"] or "?")[:8]
        print(f"  #{r['id']:>3} {r['imp_score']:>5.1f} {r['imp_novelty']:>3.0f} {r['imp_relevance']:>3.0f} {r['imp_frequency']:>3.0f} {r['imp_impact']:>3.0f} {cat:<8} {content}")

    if len(rows) > 15:
        print(f"\n  ... {len(rows) - 15} more memories")
        print(f"\n  Bottom 5 (candidates for cleanup):")
        for r in rows[-5:]:
            content = r["content"][:45]
            cat = (r["category"] or "?")[:8]
            print(f"  #{r['id']:>3} {r['imp_score']:>5.1f} {r['imp_novelty']:>3.0f} {r['imp_relevance']:>3.0f} {r['imp_frequency']:>3.0f} {r['imp_impact']:>3.0f} {cat:<8} {content}")



