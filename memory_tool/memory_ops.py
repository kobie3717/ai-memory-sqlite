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
from typing import Optional, List, Dict, Tuple, Any, Union

# Import from our modular components
from .config import *
from .database import get_db, has_vec_support
from .utils import auto_tag, word_set, normalize, find_similar, word_overlap, similarity
from .fsrs import fsrs_retention, fsrs_new_stability, fsrs_new_difficulty, fsrs_next_interval, fsrs_auto_rating
from .importance import update_importance
from .embedding import embed_and_store, embed_text, semantic_search
from .ppr import ppr_boost_search_results

logger = get_logger(__name__)

# Lazy imports for optional dependencies
try:
    import numpy as np
    import sqlite_vec
except ImportError:
    pass


# Causal query classifier
_CAUSAL_PATTERNS = re.compile(
    r'\b(why|caused?|because|how did|led to|result(?:ed|s)? in|due to|'
    r'reason|trigger(?:ed|s)?|brought about|stem(?:med|s)? from|'
    r'consequence|impact of|effect of|what made)\b',
    re.IGNORECASE
)

def _is_causal_query(query: str) -> bool:
    """Return True if query is asking about causes/effects rather than similarity."""
    return bool(_CAUSAL_PATTERNS.search(query))


def _graph_search(query: str, conn: sqlite3.Connection, limit: int = 15) -> List[Tuple[int, float]]:
    """
    Find memories via graph spreading activation.
    1. Extract noun-like tokens from query
    2. Find matching graph entities
    3. Run graph_spread from each matched entity
    4. Resolve entities → memories via memory_entity_links
    Returns list of (memory_id, rank) tuples.
    """
    from .graph import graph_spread

    # Extract candidate entity names: tokens ≥4 chars, skip stop words
    stop = {'what', 'when', 'where', 'which', 'that', 'this', 'with', 'from',
            'have', 'been', 'were', 'they', 'their', 'there', 'about', 'into',
            'will', 'would', 'could', 'should', 'because', 'result', 'caused'}
    tokens = [t for t in re.findall(r'\b[a-zA-Z]{4,}\b', query) if t.lower() not in stop]

    if not tokens:
        return []

    # Find matching entities (case-insensitive partial match, limit to 5 seed entities)
    placeholders = ' OR '.join(['LOWER(name) LIKE ?' for _ in tokens[:5]])
    params = [f'%{t.lower()}%' for t in tokens[:5]]
    try:
        entities = conn.execute(
            f"SELECT id, name FROM graph_entities WHERE {placeholders} LIMIT 5",
            params
        ).fetchall()
    except Exception:
        return []

    if not entities:
        return []

    # Run graph_spread from each matched entity, collect memory scores
    mem_scores: Dict[int, float] = {}
    seen_entities = set()

    for ent in entities[:3]:  # max 3 seeds
        ent_name = ent['name'] if hasattr(ent, 'keys') else ent[1]
        if ent_name in seen_entities:
            continue
        seen_entities.add(ent_name)

        spread = graph_spread(ent_name, depth=2)  # returns List[Dict] with 'id' key

        # Get memory IDs linked to each spread entity
        for spread_ent in spread:
            ent_id = spread_ent.get('id')
            activation = spread_ent.get('activation', 0.5)
            if ent_id is None:
                continue
            try:
                linked = conn.execute(
                    "SELECT memory_id FROM memory_entity_links WHERE entity_id = ?",
                    (ent_id,)
                ).fetchall()
                for row in linked:
                    mid = row[0]
                    mem_scores[mid] = max(mem_scores.get(mid, 0.0), activation)
            except Exception:
                pass

    # Sort and return top results as (memory_id, rank) tuples
    sorted_mems = sorted(mem_scores.items(), key=lambda x: -x[1])
    return [(mid, i) for i, (mid, _) in enumerate(sorted_mems[:limit])]


def recency_boost(created_at: datetime, alpha: float = 0.2) -> float:
    """Multiplicative recency boost for search results. Recent = higher score.

    Args:
        created_at: datetime when memory was created
        alpha: boost strength (0.2 = 20% max boost/penalty)

    Returns:
        Multiplier between ~0.9 and ~1.1
    """
    if not created_at:
        return 1.0
    days_ago = (datetime.now() - created_at).total_seconds() / 86400
    # Normalize to 0-1 range (recent = 1, old = 0)
    # Use 365 days as the reference window
    recency = max(0.1, min(1.0, 1.0 - (days_ago / 365)))
    # Convert to multiplier: 1.0 + alpha * (recency - 0.5)
    # Recent memories get +10% boost, old memories get -10% penalty
    return 1.0 + alpha * (recency - 0.5)


# Lazy imports to avoid circular dependencies
def _get_export_memory_md() -> Any:
    """Lazy import of export_memory_md to avoid circular dependency."""
    from .export import export_memory_md
    return export_memory_md


def _get_auto_link_memory() -> Any:
    """Lazy import of auto_link_memory to avoid circular dependency."""
    from .graph import auto_link_memory
    return auto_link_memory


def touch_memory(conn: sqlite3.Connection, mem_id: int) -> None:
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

        # NOTE: fsrs_reps increment removed from touch path (2026-07 citation_count migration).
        # touch_memory writes surface-count semantics (times FTS returned this memory).
        # fsrs_reps now only increments via echo feedback (apply_echo_feedback) — the
        # legitimate FSRS repetition signal. Pre-migration DBs have contaminated reps
        # (touch equality), but new increments are citation-gated.
        conn.execute("""
            UPDATE memories SET
                fsrs_stability = ?, fsrs_difficulty = ?, fsrs_interval = ?,
                last_accessed_at = datetime('now')
            WHERE id = ?
        """, (new_s, new_d, new_interval, mem_id))

        # Update importance after FSRS update
        update_importance(mem_id, conn)




def auto_adjust_priority(conn: sqlite3.Connection, mem_id: int) -> None:
    row = conn.execute(
        "SELECT access_count, priority FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()
    if row:
        suggested = min(10, row["access_count"] // 5)
        if suggested > row["priority"]:
            conn.execute("UPDATE memories SET priority = ? WHERE id = ?", (suggested, mem_id))


# ── Smart Ingest (v4 Feature #4) ──


def check_contradictions(content: str, category: Optional[str] = None, project: Optional[str] = None) -> Optional[str]:
    """Check for potential contradictions with existing memories.
    Returns a warning string if contradictions are detected, None otherwise."""

    # Contradiction patterns (negation indicators)
    CONTRADICTION_PATTERNS = [
        r'\bnot\b', r'\bdon\'?t\b', r'\bdoesn\'?t\b', r'\bdidn\'?t\b',
        r'\bwon\'?t\b', r'\bwouldn\'?t\b', r'\bcan\'?t\b', r'\bcannot\b',
        r'\brejected\b', r'\bdisabled\b', r'\bremoved\b', r'\bstopped\b',
        r'\bno longer\b', r'\binstead of\b', r'\breplaced\b', r'\bchanged from\b',
        r'\bunlike\b', r'\bopposite\b', r'\bcontrary to\b', r'\brather than\b',
        r'\bnever\b', r'\bneither\b', r'\bnor\b', r'\babandoned\b', r'\bdiscarded\b'
    ]

    # Only check if semantic search is available
    if not has_vec_support():
        return None

    content_lower = content.lower()
    has_negation = any(re.search(pattern, content_lower) for pattern in CONTRADICTION_PATTERNS)

    # If no negation patterns, no contradiction likely
    if not has_negation:
        return None

    # Do semantic search for similar memories (>80% similarity)
    conn = get_db()
    try:
        query_embedding = embed_text(content)
        if query_embedding is None:
            conn.close()
            return None

        # Search for highly similar memories using cosine distance
        query = """
            SELECT m.id, m.content, (1 - distance) as similarity
            FROM memory_vec v
            JOIN memories m ON m.id = v.rowid
            WHERE m.active = 1
            AND v.embedding MATCH ?
            AND k = 10
            ORDER BY distance
        """

        results = conn.execute(query, (query_embedding,)).fetchall()

        # Filter by similarity threshold and category/project if provided
        high_similarity = []
        for r in results:
            if r['similarity'] > 0.80:
                if category and r['content']:
                    # Check if same general topic
                    mem = conn.execute("SELECT category, project FROM memories WHERE id = ?", (r['id'],)).fetchone()
                    if mem and mem['category'] == category:
                        high_similarity.append(r)
                else:
                    high_similarity.append(r)

        conn.close()

        if high_similarity:
            # Return warning with the most similar memory
            most_similar = high_similarity[0]
            preview = most_similar['content'][:80] + "..." if len(most_similar['content']) > 80 else most_similar['content']

            # Integration with belief system: weaken confidence of contradicting memory slightly
            try:
                from .beliefs import weaken_confidence
                weaken_confidence(
                    conn, most_similar['id'], 0.05,
                    f"Contradiction detected with new memory: {content[:60]}..."
                )
                logger.debug(f"Weakened confidence of memory #{most_similar['id']} due to contradiction")
            except Exception:
                pass  # beliefs module might not be available or column doesn't exist yet

            return f"⚠️  Potential contradiction with memory #{most_similar['id']} ({most_similar['similarity']:.0%} similar): {preview}"

        return None

    except Exception as e:
        conn.close()
        return None


def smart_ingest(category: str, content: str, tags: str = "", project: Optional[str] = None,
                 priority: int = 0, related_to: Optional[int] = None,
                 expires_at: Optional[str] = None, source: str = "manual",
                 topic_key: Optional[str] = None, derived_from: Optional[str] = None,
                 citations: Optional[str] = None, reasoning: Optional[str] = None,
                 wing: Optional[str] = None, room: Optional[str] = None, tier: Optional[str] = None,
                 agent_id: Optional[str] = None, disclosure: Optional[str] = None,
                 is_pinned: bool = False) -> Optional[int]:
    """
    Smart ingestion with 4-tier similarity handling:
    - SKIP: >85% (duplicate blocked)
    - UPDATE: 70-85% same category/project (auto-update existing)
    - SUPERSEDE: 50-70% same project (insert new, mark old superseded)
    - CREATE: <50% (normal insert)
    """
    tags = auto_tag(content, tags)

    # Determine memory tier if not explicitly set
    if tier is None:
        from .tiers import classify_tier
        # Build memory_row dict for classification
        memory_row = {
            'category': category,
            'priority': priority,
            'tags': tags,
            'proof_count': 1,
            'expires_at': expires_at,
            'access_count': 0
        }
        tier = classify_tier(memory_row)

    # Content-hash dedup check: prevent exact duplicates within 30 seconds (claude-mem pattern)
    # Use SHA256 hash of category:content (case-insensitive, whitespace-normalized)
    content_hash = hashlib.sha256(f"{category}:{content.strip().lower()}".encode()).hexdigest()
    conn = get_db()
    recent_dupe = conn.execute("""
        SELECT id, created_at FROM memories
        WHERE content_hash = ? AND active = 1
        AND datetime(created_at) > datetime('now', '-30 seconds')
        ORDER BY created_at DESC LIMIT 1
    """, (content_hash,)).fetchone()
    conn.close()

    if recent_dupe:
        # Skip duplicate and notify user
        logger.info(f"⚡ DEDUP: Blocked duplicate within 30s window (matches #{recent_dupe['id']})")
        return None

    # Check for contradictions using semantic search (if available)
    contradiction_warning = check_contradictions(content, category, project)

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
            logger.info(f"Updated memory #{existing['id']} (revision {new_revision}) key:{topic_key}")

            # Print contradiction warning if detected
            if contradiction_warning:
                logger.warning(contradiction_warning)

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
            logger.warning(f"DUPLICATE BLOCKED (score={score:.0%}): similar to #{best_id}")
            logger.warning(f"  Existing: {best_content}")
            logger.warning(f"  Use 'memory-tool update {best_id} \"{content}\"' to update instead.")
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
            logger.info(f"AUTO-UPDATED memory #{best_id} ({score:.0%} match, revision {new_revision})")

            # Print contradiction warning if detected
            if contradiction_warning:
                logger.warning(contradiction_warning)

            return best_id

        # SUPERSEDE: 50-70% same project
        elif score > 0.50 and project == best_proj:
            # Insert new, mark old as superseded
            conn = get_db()

            # Compute dispatch_priority
            from .dispatch import classify_dispatch_priority
            dispatch_priority = classify_dispatch_priority({
                'category': category,
                'proof_count': 1,
                'access_count': 0,
                'tags': tags,
                'created_at': datetime.now().isoformat()
            })

            cur = conn.execute(
                """INSERT INTO memories (category, content, tags, project, priority, accessed_at, expires_at, source, topic_key, derived_from, citations, reasoning, content_hash, wing, room, tier, agent_id, disclosure_condition, is_pinned, dispatch_priority)
                   VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (category, content, tags, project, priority, expires_at, source, topic_key, derived_from, citations, reasoning, content_hash, wing, room, tier, agent_id, disclosure, 1 if is_pinned else 0, dispatch_priority)
            )
            new_id = cur.lastrowid

            # Deactivate old
            old_row = conn.execute("SELECT content, category, project FROM memories WHERE id = ?", (best_id,)).fetchone()
            conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (best_id,))

            # Audit the supersession
            from .gdpr import audit
            if old_row:
                audit(
                    event_type='supersede',
                    memory_id=best_id,
                    content=old_row['content'],
                    category=old_row['category'],
                    project=old_row['project'],
                    actor='system',
                    reason='topic_upsert_supersede',
                    conn=conn
                )

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
            logger.info(f"Added memory #{new_id}, supersedes #{best_id} ({score:.0%} overlap, different content)")

            # Print contradiction warning if detected
            if contradiction_warning:
                logger.warning(contradiction_warning)

            return new_id

        # CREATE with warning: <50% or different category/project
        else:
            logger.info(f"Similar memory exists (score={score:.0%}): #{best_id}: {best_content}")
            if contradiction_warning:
                logger.warning(contradiction_warning)

    # CREATE: Normal insert
    conn = get_db()

    # Compute dispatch_priority
    from .dispatch import classify_dispatch_priority
    dispatch_priority = classify_dispatch_priority({
        'category': category,
        'proof_count': 1,
        'access_count': 0,
        'tags': tags,
        'created_at': datetime.now().isoformat()
    })

    cur = conn.execute(
        """INSERT INTO memories (category, content, tags, project, priority, accessed_at, expires_at, source, topic_key, derived_from, citations, reasoning, content_hash, wing, room, tier, agent_id, disclosure_condition, is_pinned, dispatch_priority)
           VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (category, content, tags, project, priority, expires_at, source, topic_key, derived_from, citations, reasoning, content_hash, wing, room, tier, agent_id, disclosure, 1 if is_pinned else 0, dispatch_priority)
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
    logger.info(f"Added memory #{mem_id} [{category}]{key_str}{' tags:' + tags if tags else ''}")

    # Print contradiction warning if detected
    if contradiction_warning:
        logger.warning(contradiction_warning)

    # Circus write-through hook
    try:
        from .circus_sync import init_circus_sync
        from .config import DB_PATH
        sync = init_circus_sync(str(DB_PATH))
        sync.auto_publish_on_add(mem_id, content, category, tags, confidence=0.9)
    except Exception:
        pass  # Silent fail - don't break local operations

    return mem_id




def add_memory(category: str, content: str, tags: str = "", project: Optional[str] = None,
               priority: int = 0, related_to: Optional[int] = None,
               expires_at: Optional[str] = None, source: str = "manual",
               topic_key: Optional[str] = None, skip_dedup: bool = False,
               derived_from: Optional[str] = None, citations: Optional[str] = None,
               reasoning: Optional[str] = None, wing: Optional[str] = None,
               room: Optional[str] = None, tier: Optional[str] = None,
               agent_id: Optional[str] = None, disclosure: Optional[str] = None,
               is_pinned: bool = False) -> Optional[int]:
    """Legacy add_memory wrapper for backward compatibility."""
    if skip_dedup:
        # Old behavior: skip dedup entirely
        tags = auto_tag(content, tags)

        # Check for contradictions even if skipping dedup
        contradiction_warning = check_contradictions(content, category, project)

        # Compute content hash (truncate to 16 chars to match dedup logic)
        content_hash = hashlib.sha256(f"{category}:{content.strip().lower()}".encode()).hexdigest()

        conn = get_db()

        # Determine tier for skip_dedup path
        mem_tier = tier
        if mem_tier is None:
            from .tiers import classify_tier
            memory_row = {
                'category': category,
                'priority': priority,
                'tags': tags,
                'proof_count': 1,
                'expires_at': expires_at,
                'access_count': 0
            }
            mem_tier = classify_tier(memory_row)

        # Compute dispatch_priority
        from .dispatch import classify_dispatch_priority
        dispatch_priority = classify_dispatch_priority({
            'category': category,
            'proof_count': 1,
            'access_count': 0,
            'tags': tags,
            'created_at': datetime.now().isoformat()
        })

        cur = conn.execute(
            """INSERT INTO memories (category, content, tags, project, priority, accessed_at, expires_at, source, topic_key, derived_from, citations, reasoning, content_hash, wing, room, tier, agent_id, disclosure_condition, is_pinned, dispatch_priority)
               VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (category, content, tags, project, priority, expires_at, source, topic_key, derived_from, citations, reasoning, content_hash, wing, room, mem_tier, agent_id, disclosure, 1 if is_pinned else 0, dispatch_priority)
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
        logger.info(f"Added memory #{mem_id} [{category}]{' tags:' + tags if tags else ''}")

        # Print contradiction warning if detected
        if contradiction_warning:
            logger.warning(contradiction_warning)

        # Circus write-through hook
        try:
            from .circus_sync import init_circus_sync
            from .config import DB_PATH
            sync = init_circus_sync(str(DB_PATH))
            sync.auto_publish_on_add(mem_id, content, category, tags, confidence=0.9)
        except Exception:
            pass  # Silent fail - don't break local operations

        return mem_id
    else:
        return smart_ingest(category, content, tags, project, priority, related_to, expires_at, source, topic_key, derived_from, citations, reasoning, wing, room, tier, agent_id, disclosure, is_pinned)




def search_memories(query: str, mode: str = "hybrid", since: Optional[str] = None, until: Optional[str] = None, apply_recency_boost: bool = True, reasoning_boost: bool = True, project: Optional[str] = None, tags: Optional[str] = None, wing: Optional[str] = None, room: Optional[str] = None, passport_credential: Optional[Dict] = None) -> Tuple[List[sqlite3.Row], int, Optional[Tuple[datetime, datetime]]]:
    """
    Search memories with multiple modes:
    - hybrid: Combine FTS and vector search with RRF (default)
    - keyword: FTS only
    - semantic: Vector only

    Args:
        query: Search query string
        mode: Search mode (hybrid/keyword/semantic)
        since: ISO date string for filtering memories created/updated after this date
        until: ISO date string for filtering memories created/updated before this date
        apply_recency_boost: Apply recency boost to search scores (default: True)
        reasoning_boost: Apply ReasoningBank boost based on prediction outcomes (default: True)
        project: Filter by project name (applied before search)
        tags: Filter by tags (applied before search)
        wing: Filter by wing namespace (applied before search)
        room: Filter by room namespace (applied before search)
        passport_credential: Passport credential for access control filtering

    Returns:
        Tuple of (rows, search_id, temporal_range) where:
        - rows: List of memory rows
        - search_id: ID for feedback logging
        - temporal_range: (start_date, end_date) if temporal filtering applied, None otherwise
    """
    import time
    start_time = time.time()

    # Extract quoted phrases for boosting
    quoted_phrases = re.findall(r'"([^"]+)"', query)

    # Detect person names (capitalized words, 2+ chars, not at start of sentence)
    words = query.split()
    person_names = []
    for i, word in enumerate(words):
        # Skip first word (might be capitalized for sentence start)
        if i > 0 and word[0].isupper() and len(word) >= 2:
            # Basic heuristic: not common words
            if word.lower() not in ['i', 'a', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
                person_names.append(word)

    # Extract temporal constraints from query if not explicitly provided
    temporal_range = None
    if not since and not until:
        try:
            from .temporal import extract_temporal_constraint, strip_temporal_expressions
            temporal_range = extract_temporal_constraint(query)
            if temporal_range:
                since = temporal_range[0].isoformat()
                until = temporal_range[1].isoformat()
                # Clean query for better matching
                query = strip_temporal_expressions(query)
        except ImportError:
            pass  # temporal module not available
        except Exception:
            pass  # Date parsing failed, continue with original query

    conn = get_db()
    fts_results = []
    vec_results = []

    # Build metadata pre-filter (project/tags/wing/room) - applied BEFORE search
    metadata_filter_fts = ""
    metadata_filter_plain = ""
    metadata_params = []
    if project:
        metadata_filter_fts += " AND m.project = ?"
        metadata_filter_plain += " AND project = ?"
        metadata_params.append(project)
    if tags:
        metadata_filter_fts += " AND m.tags LIKE ?"
        metadata_filter_plain += " AND tags LIKE ?"
        metadata_params.append(f"%{tags}%")
    if wing:
        metadata_filter_fts += " AND m.wing = ?"
        metadata_filter_plain += " AND wing = ?"
        metadata_params.append(wing)
    if room:
        metadata_filter_fts += " AND m.room = ?"
        metadata_filter_plain += " AND room = ?"
        metadata_params.append(room)

    # Build date filter clause if temporal constraints provided
    date_filter_fts = ""
    date_filter_plain = ""
    date_params = []
    if since or until:
        conditions_fts = []
        conditions_plain = []
        if since:
            conditions_fts.append("m.created_at >= ?")
            conditions_plain.append("created_at >= ?")
            date_params.append(since)
        if until:
            conditions_fts.append("m.created_at <= ?")
            conditions_plain.append("created_at <= ?")
            date_params.append(until)
        if conditions_fts:
            date_filter_fts = " AND " + " AND ".join(conditions_fts)
            date_filter_plain = " AND " + " AND ".join(conditions_plain)

    # Combine all filters
    all_params_fts = metadata_params + date_params
    all_params_plain = metadata_params + date_params
    combined_filter_fts = metadata_filter_fts + date_filter_fts
    combined_filter_plain = metadata_filter_plain + date_filter_plain

    # 1. FTS keyword search
    if mode in ("hybrid", "keyword"):
        try:
            fts_query = f"""
                SELECT m.id FROM memories m
                JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ? AND m.active = 1{combined_filter_fts}
                ORDER BY rank LIMIT 15
            """
            rows = conn.execute(fts_query, [query] + all_params_fts).fetchall()
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
                    AND k = 100
                    ORDER BY distance
                """, (query_vec,)).fetchall()

                # Adaptive k: drop results where cosine distance jumps >40% above the best distance
                if rows:
                    best_dist = rows[0]['distance']
                    cutoff = best_dist + max(0.15, best_dist * 0.4)  # absolute floor of 0.15
                    rows = [r for r in rows if r['distance'] <= cutoff]

                # Filter to active only and apply metadata + date filters BEFORE semantic search
                if combined_filter_plain:
                    active_query = f"SELECT id FROM memories WHERE active = 1{combined_filter_plain}"
                    active_ids = set(r['id'] for r in conn.execute(active_query, all_params_plain).fetchall())
                else:
                    active_ids = set(r['id'] for r in conn.execute(
                        "SELECT id FROM memories WHERE active = 1"
                    ).fetchall())
                vec_results = [(r['id'], i) for i, r in enumerate(rows) if r['id'] in active_ids]
            except Exception:
                # Silently fail if vec table doesn't exist yet
                pass

    # 3. Causal graph routing — for "why/caused/led to" queries
    graph_results: List[Tuple[int, int]] = []
    if _is_causal_query(query) and mode in ("hybrid", "keyword"):
        try:
            graph_results = _graph_search(query, conn)
        except Exception:
            pass  # graph search is best-effort

    # 4. Reciprocal Rank Fusion (combine scores) with recency boost
    if mode == "hybrid" and (fts_results or vec_results or graph_results):
        scores = {}
        for mem_id, rank in fts_results:
            scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (RRF_K + rank + 1)
        for mem_id, rank in vec_results:
            scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (RRF_K + rank + 1)
        # Add graph results as third signal (causal queries only) with 0.7 weight
        for mem_id, rank in graph_results:
            scores[mem_id] = scores.get(mem_id, 0) + 0.7 / (RRF_K + rank + 1)

        # Apply recency boost to final scores (if enabled)
        if scores and apply_recency_boost:
            # Fetch created_at, proof_count, tier, and content for all candidates
            mem_ids = list(scores.keys())
            placeholders = ','.join('?' * len(mem_ids))
            date_rows = conn.execute(
                f"SELECT id, created_at, proof_count, tier, content FROM memories WHERE id IN ({placeholders})",
                mem_ids
            ).fetchall()

            for row in date_rows:
                mem_id = row['id']
                created_at = row['created_at']
                content = row['content'] or ''
                content_lower = content.lower()
                tier = row['tier'] if 'tier' in row.keys() else 'episodic'

                if created_at:
                    try:
                        created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00')).replace(tzinfo=None)
                        boost = recency_boost(created_dt)
                        scores[mem_id] *= boost
                    except (ValueError, AttributeError):
                        pass  # Invalid date, skip boost

                # Apply tier boost: semantic > episodic > working
                # Semantic memories are proven long-term knowledge, rank highest
                tier_boosts = {
                    'semantic': 1.2,
                    'episodic': 1.0,
                    'working': 0.8
                }
                tier_boost = tier_boosts.get(tier, 1.0)
                scores[mem_id] *= tier_boost

                # Apply proof boost: memories confirmed by multiple sources rank higher
                proof_count = row['proof_count'] or 1
                if proof_count > 1:
                    # Boost by up to 50% for well-confirmed memories (capped at 5 sources)
                    proof_boost = 1.0 + 0.1 * min(proof_count, 5)
                    scores[mem_id] *= proof_boost

                # Apply quoted phrase boost: exact matches get 1.5x boost
                if quoted_phrases:
                    for phrase in quoted_phrases:
                        if phrase.lower() in content_lower:
                            scores[mem_id] *= 1.5
                            break  # Only boost once per memory

                # Apply person name boost: 1.3x for mentions of detected names
                if person_names:
                    for name in person_names:
                        if name.lower() in content_lower or name in content:
                            scores[mem_id] *= 1.3
                            break  # Only boost once per memory

        # Apply reasoning boost: memories linked to confirmed predictions rank higher
        from .reasoning import apply_reasoning_boost_to_scores
        apply_reasoning_boost_to_scores(scores, conn)

        # Sort by combined RRF score with all boosts applied
        ranked_ids = sorted(scores.keys(), key=lambda x: -scores[x])[:20]

        # Fetch full rows
        if ranked_ids:
            placeholders = ','.join('?' * len(ranked_ids))
            rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", ranked_ids).fetchall()
            # Re-sort by RRF score
            id_to_row = {r['id']: r for r in rows}
            rows = [id_to_row[mid] for mid in ranked_ids if mid in id_to_row]

            # Apply PPR boost if we have enough results and the graph isn't too large
            if len(rows) >= 3:
                # Check memory count to avoid PPR performance issues
                mem_count = conn.execute("SELECT COUNT(*) FROM memories WHERE active=1").fetchone()[0]
                if mem_count <= 10000:
                    # Normalize scores to 0-1 range for PPR
                    max_score = max(scores.values()) if scores else 1.0
                    # Convert rows to format expected by PPR (list of dicts with 'id' and 'score')
                    ppr_input = [{'id': r['id'], 'score': scores[r['id']] / max_score} for r in rows]
                    # Apply PPR boost (25% weight)
                    ppr_results = ppr_boost_search_results(ppr_input, ppr_weight=0.25)
                    # Re-sort rows based on PPR-boosted scores
                    ppr_id_order = [r['id'] for r in ppr_results]
                    rows = [id_to_row[mid] for mid in ppr_id_order if mid in id_to_row]
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
        fallback_query = f"""
            SELECT * FROM memories
            WHERE active = 1 AND (content LIKE ? OR tags LIKE ? OR project LIKE ?){combined_filter_plain}
            ORDER BY updated_at DESC LIMIT 20
        """
        fallback_params = [f"%{query}%", f"%{query}%", f"%{query}%"] + all_params_plain
        rows = conn.execute(fallback_query, fallback_params).fetchall()

    # Apply access control filtering if passport provided
    if passport_credential:
        from .access_control import filter_memories_by_access
        before_count = len(rows)
        rows = filter_memories_by_access(rows, passport_credential)
        filtered_count = before_count - len(rows)
        if filtered_count > 0:
            logger.debug(f"Access control filtered {filtered_count} memories")

    # Apply dispatch filtering based on query urgency
    from .dispatch import classify_query_urgency, apply_dispatch_filter, cap_render_results
    query_urgency = classify_query_urgency(query)
    before_dispatch = len(rows)
    rows = apply_dispatch_filter(rows, query_urgency)
    rows = cap_render_results(rows, max_tier3=3)
    after_dispatch = len(rows)
    if before_dispatch != after_dispatch:
        logger.debug(f"Dispatch filter ({query_urgency}): {before_dispatch} -> {after_dispatch} results")

    # Calculate latency
    latency_ms = int((time.time() - start_time) * 1000)

    # Log the search
    result_ids = ','.join(str(r['id']) for r in rows)
    cur = conn.execute("""
        INSERT INTO search_log (query, search_type, result_ids, result_count, latency_ms)
        VALUES (?, ?, ?, ?, ?)
    """, (query, mode, result_ids, len(rows), latency_ms))
    search_id = cur.lastrowid

    # Touch accessed memories
    for r in rows:
        touch_memory(conn, r["id"])
        auto_adjust_priority(conn, r["id"])
    conn.commit()
    conn.close()

    # Return temporal_range if it was applied
    return rows, search_id, temporal_range




def get_memory(mem_id: int) -> Optional[sqlite3.Row]:
    """Get full detail for a single memory."""
    conn = get_db()
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    conn.close()
    return row




def list_memories(category: Optional[str] = None, project: Optional[str] = None,
                  tag: Optional[str] = None, stale_only: bool = False,
                  expired_only: bool = False, sort_by_proof: bool = False,
                  wing: Optional[str] = None, room: Optional[str] = None) -> List[sqlite3.Row]:
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
    if wing:
        query += " AND wing = ?"
        params.append(wing)
    if room:
        query += " AND room = ?"
        params.append(room)
    if stale_only:
        query += " AND stale = 1"
    if expired_only:
        query += " AND expires_at IS NOT NULL AND expires_at < datetime('now')"

    if sort_by_proof:
        query += " ORDER BY proof_count DESC, updated_at DESC"
    else:
        query += " ORDER BY priority DESC, updated_at DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows




def update_memory(mem_id: int, content: str) -> None:
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
    logger.info(f"Updated memory #{mem_id} (revision {new_revision})")




def delete_memory(mem_id: int) -> None:
    from .gdpr import audit
    conn = get_db()

    # Get memory details for audit log
    row = conn.execute("SELECT content, category, project FROM memories WHERE id = ?", (mem_id,)).fetchone()

    conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (mem_id,))
    conn.commit()

    # Audit the deletion
    if row:
        audit(
            event_type='delete',
            memory_id=mem_id,
            content=row['content'],
            category=row['category'],
            project=row['project'],
            actor='user',
            reason='user_initiated_delete',
            conn=conn
        )

    conn.close()
    _get_export_memory_md()()
    logger.info(f"Deactivated memory #{mem_id}")




def tag_memory(mem_id: int, tags: str) -> None:
    conn = get_db()
    existing = conn.execute("SELECT tags FROM memories WHERE id = ?", (mem_id,)).fetchone()
    if existing:
        current = set(filter(None, existing["tags"].split(",")))
        new_tags = set(filter(None, tags.split(",")))
        merged = ",".join(sorted(current | new_tags))
        conn.execute("UPDATE memories SET tags = ?, updated_at = datetime('now') WHERE id = ?", (merged, mem_id))
        conn.commit()
        logger.info(f"Tagged memory #{mem_id}: {merged}")
    conn.close()
    _get_export_memory_md()()


# ── Relationships ──



def show_importance_ranking() -> None:
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

    # User-facing output - display stays as print()
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


def add_reflection(task_summary: str, outcome: str, worked: str, failed: str,
                   next_time: str, task_type: Optional[str] = None,
                   project: Optional[str] = None) -> Optional[int]:
    """Add a Reflexion-style self-improvement reflection.

    Args:
        task_summary: Brief description of the task
        outcome: success, partial, or failure
        worked: What worked well
        failed: What didn't work or failed
        next_time: What to do differently next time
        task_type: Optional task type (auto-detected from summary if not provided)
        project: Optional project name

    Returns:
        Memory ID of the stored reflection
    """
    # Auto-detect task type from summary if not provided
    if not task_type:
        summary_lower = task_summary.lower()
        if any(word in summary_lower for word in ['nginx', 'apache', 'config', 'server']):
            task_type = 'configuration'
        elif any(word in summary_lower for word in ['deploy', 'release', 'push']):
            task_type = 'deployment'
        elif any(word in summary_lower for word in ['bug', 'fix', 'error', 'issue']):
            task_type = 'debugging'
        elif any(word in summary_lower for word in ['test', 'pytest', 'jest']):
            task_type = 'testing'
        elif any(word in summary_lower for word in ['database', 'sql', 'migration']):
            task_type = 'database'
        elif any(word in summary_lower for word in ['api', 'endpoint', 'route']):
            task_type = 'api'
        else:
            task_type = 'general'

    # Build structured reflection content
    content = f"""Task: {task_summary}
Outcome: {outcome}

What worked:
{worked}

What failed:
{failed}

What to do differently:
{next_time}"""

    # Store as learning memory with reflection tag
    tags = f"reflection,{outcome},{task_type}"

    mem_id = add_memory(
        category='learning',
        content=content,
        tags=tags,
        project=project,
        priority=1 if outcome == 'failure' else 0,  # Higher priority for failures
        wing='reflections',
        room=task_type,
        source='reflexion'
    )

    logger.info(f"Added reflection #{mem_id} for task: {task_summary[:50]}... (outcome: {outcome})")
    return mem_id


def load_reflections(task_description: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Load relevant past reflections before starting a task.

    Args:
        task_description: Description of the upcoming task
        limit: Maximum number of reflections to return

    Returns:
        List of relevant reflection memories
    """
    conn = get_db()

    # Search for relevant reflections in the reflections wing
    # Use hybrid search if available, fall back to FTS
    from .embedding import has_vec_support

    if has_vec_support():
        # Use semantic search for better matching
        results, _, _ = search_memories(
            query=task_description,
            mode='hybrid',
            wing='reflections'
        )
    else:
        # Fall back to FTS search
        results = conn.execute("""
            SELECT m.*, rank
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.rowid
            WHERE memories_fts MATCH ?
            AND m.active = 1
            AND m.wing = 'reflections'
            ORDER BY rank
            LIMIT ?
        """, (task_description, limit)).fetchall()

    conn.close()

    # Parse reflection structure
    reflections = []
    for row in results[:limit]:
        content = row['content']

        # Extract structured fields
        outcome_match = re.search(r'Outcome:\s*(\w+)', content)
        worked_match = re.search(r'What worked:\s*(.+?)(?=\n\nWhat failed:|$)', content, re.DOTALL)
        failed_match = re.search(r'What failed:\s*(.+?)(?=\n\nWhat to do differently:|$)', content, re.DOTALL)
        next_match = re.search(r'What to do differently:\s*(.+?)$', content, re.DOTALL)

        reflections.append({
            'id': row['id'],
            'task': re.search(r'Task:\s*(.+)', content).group(1) if re.search(r'Task:\s*(.+)', content) else 'Unknown',
            'outcome': outcome_match.group(1) if outcome_match else 'unknown',
            'worked': worked_match.group(1).strip() if worked_match else '',
            'failed': failed_match.group(1).strip() if failed_match else '',
            'next_time': next_match.group(1).strip() if next_match else '',
            'tags': row['tags'],
            'created_at': row['created_at']
        })

    return reflections


def list_reflections_by_task() -> Dict[str, List[Dict[str, Any]]]:
    """Show all stored reflections grouped by task type (room).

    Returns:
        Dictionary mapping task types to lists of reflections
    """
    conn = get_db()

    rows = conn.execute("""
        SELECT id, content, tags, room, created_at, accessed_at, access_count
        FROM memories
        WHERE active = 1
        AND wing = 'reflections'
        ORDER BY room, created_at DESC
    """).fetchall()

    conn.close()

    # Group by room (task type)
    grouped = {}
    for row in rows:
        room = row['room'] or 'general'
        if room not in grouped:
            grouped[room] = []

        # Extract outcome from tags
        outcome = 'unknown'
        if 'success' in row['tags']:
            outcome = 'success'
        elif 'partial' in row['tags']:
            outcome = 'partial'
        elif 'failure' in row['tags']:
            outcome = 'failure'

        # Extract task summary
        task_match = re.search(r'Task:\s*(.+)', row['content'])
        task = task_match.group(1) if task_match else row['content'][:60]

        grouped[room].append({
            'id': row['id'],
            'task': task,
            'outcome': outcome,
            'created_at': row['created_at'],
            'access_count': row['access_count'],
            'full_content': row['content']
        })

    return grouped


def passive_verify(dry_run: bool = False) -> dict:
    """
    Promote memories that repeatedly appear in search results (revealed preference).
    Memories in top search results on 3+ distinct days = auto-promote.
    Returns: {promoted: [ids], already_verified: [ids], checked: int}
    """
    conn = get_db()

    # Parse search_log result_ids (comma-separated memory IDs per search)
    # Group by date (created_at[:10]), count distinct dates per memory_id
    rows = conn.execute("""
        SELECT result_ids, created_at FROM search_log
        WHERE result_ids IS NOT NULL AND result_ids != ''
        ORDER BY created_at DESC LIMIT 1000
    """).fetchall()

    # Count distinct dates each memory appeared in search results
    from collections import defaultdict
    memory_dates = defaultdict(set)
    for row in rows:
        date = row['created_at'][:10]
        for mid in row['result_ids'].split(','):
            mid = mid.strip()
            if mid.isdigit():
                memory_dates[int(mid)].add(date)

    total_distinct_days = len(set(row['created_at'][:10] for row in rows))
    # Require appearance on >60% of tracked days (genuine signal, not noise)
    # With 23 days of data, threshold = ~14 days. Catches top ~1% of memories.
    min_days = max(5, int(total_distinct_days * 0.60))
    candidates = {mid: dates for mid, dates in memory_dates.items() if len(dates) >= min_days}

    promoted = []
    already_verified = []

    for mem_id, dates in candidates.items():
        mem = conn.execute(
            "SELECT id, priority, tags, active, access_count FROM memories WHERE id = ? AND active = 1",
            (mem_id,)
        ).fetchone()
        if not mem:
            continue

        tags = (mem['tags'] or '').split(',')
        tags = [t.strip() for t in tags if t.strip()]

        if 'search-verified' in tags:
            already_verified.append(mem_id)
            continue

        if not dry_run:
            # Boost priority (cap at 9)
            new_priority = min(9, (mem['priority'] or 5) + 2)
            # Add search-verified tag
            new_tags = ','.join(tags + ['search-verified'])
            # Boost access_count to >= 5 for decay immunity
            new_access = max(5, mem['access_count'] or 0)
            conn.execute(
                "UPDATE memories SET priority = ?, tags = ?, access_count = ?, updated_at = datetime('now') WHERE id = ?",
                (new_priority, new_tags, new_access, mem_id)
            )

        promoted.append(mem_id)

    if not dry_run:
        conn.commit()
    conn.close()

    return {
        'promoted': promoted,
        'already_verified': already_verified,
        'checked': len(candidates),
        'dry_run': dry_run
    }


