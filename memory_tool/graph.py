"""Graph intelligence operations (Phase 3 feature)."""

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


def graph_add_entity(name, entity_type, summary="", importance=3):
    """Add or update an entity. Returns entity id."""
    conn = get_db()
    try:
        cursor = conn.execute(
            """INSERT INTO graph_entities (name, type, summary, importance)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   type = excluded.type,
                   summary = excluded.summary,
                   importance = excluded.importance,
                   updated_at = datetime('now')
               RETURNING id""",
            (name, entity_type, summary, importance)
        )
        entity_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return entity_id
    except sqlite3.IntegrityError as e:
        print(f"Error adding entity: {e}")
        conn.close()
        return None




def graph_get_or_create_entity(name, entity_type="concept", summary=""):
    """Get entity by name, create if doesn't exist. Case-insensitive lookup."""
    conn = get_db()
    # Case-insensitive lookup
    row = conn.execute(
        "SELECT id FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (name,)
    ).fetchone()

    if row:
        entity_id = row[0]
        conn.close()
        return entity_id

    conn.close()
    return graph_add_entity(name, entity_type, summary)




def graph_add_relationship(from_name, to_name, relation_type, note=""):
    """Add a relationship between two entities (by name). Creates entities if they don't exist."""
    from_id = graph_get_or_create_entity(from_name)
    to_id = graph_get_or_create_entity(to_name)

    if not from_id or not to_id:
        return False

    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO graph_relationships (from_entity_id, to_entity_id, relation_type, note)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(from_entity_id, to_entity_id, relation_type) DO UPDATE SET
                   note = excluded.note""",
            (from_id, to_id, relation_type, note)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError as e:
        print(f"Error adding relationship: {e}")
        conn.close()
        return False




def graph_set_fact(entity_name, key, value, confidence=1.0, source=""):
    """Set a fact on an entity. If key exists, update it and log history."""
    entity_id = graph_get_or_create_entity(entity_name)
    if not entity_id:
        return False

    conn = get_db()

    # Check if fact exists
    existing = conn.execute(
        "SELECT value FROM graph_facts WHERE entity_id = ? AND key = ?",
        (entity_id, key)
    ).fetchone()

    if existing and existing[0] != value:
        # Log to history
        conn.execute(
            """INSERT INTO graph_fact_history (entity_id, key, old_value, new_value)
               VALUES (?, ?, ?, ?)""",
            (entity_id, key, existing[0], value)
        )

    # Insert or update fact
    conn.execute(
        """INSERT INTO graph_facts (entity_id, key, value, confidence, source)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(entity_id, key) DO UPDATE SET
               value = excluded.value,
               confidence = excluded.confidence,
               source = excluded.source,
               updated_at = datetime('now')""",
        (entity_id, key, value, confidence, source)
    )
    conn.commit()
    conn.close()
    return True




def graph_get_entity(name):
    """Get entity with all its facts and relationships."""
    conn = get_db()

    # Get entity
    entity = conn.execute(
        "SELECT * FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (name,)
    ).fetchone()

    if not entity:
        conn.close()
        return None

    entity_id = entity['id']

    # Get facts
    facts = conn.execute(
        "SELECT * FROM graph_facts WHERE entity_id = ? ORDER BY key",
        (entity_id,)
    ).fetchall()

    # Get outgoing relationships
    out_rels = conn.execute(
        """SELECT r.*, e.name as to_name, e.type as to_type
           FROM graph_relationships r
           JOIN graph_entities e ON r.to_entity_id = e.id
           WHERE r.from_entity_id = ?
           ORDER BY r.relation_type, e.name""",
        (entity_id,)
    ).fetchall()

    # Get incoming relationships
    in_rels = conn.execute(
        """SELECT r.*, e.name as from_name, e.type as from_type
           FROM graph_relationships r
           JOIN graph_entities e ON r.from_entity_id = e.id
           WHERE r.to_entity_id = ?
           ORDER BY r.relation_type, e.name""",
        (entity_id,)
    ).fetchall()

    # Get linked memories
    linked_memories = conn.execute(
        """SELECT m.* FROM memories m
           JOIN memory_entity_links l ON m.id = l.memory_id
           WHERE l.entity_id = ? AND m.active = 1
           ORDER BY m.updated_at DESC LIMIT 5""",
        (entity_id,)
    ).fetchall()

    conn.close()

    return {
        'entity': dict(entity),
        'facts': [dict(f) for f in facts],
        'outgoing': [dict(r) for r in out_rels],
        'incoming': [dict(r) for r in in_rels],
        'memories': [dict(m) for m in linked_memories]
    }




def graph_list_entities(entity_type=None):
    """List all entities, optionally filtered by type."""
    conn = get_db()
    if entity_type:
        rows = conn.execute(
            "SELECT * FROM graph_entities WHERE type = ? ORDER BY importance DESC, name",
            (entity_type,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM graph_entities ORDER BY type, importance DESC, name"
        ).fetchall()
    conn.close()
    return rows




def graph_delete_entity(name):
    """Delete an entity and its relationships/facts."""
    conn = get_db()
    result = conn.execute(
        "DELETE FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (name,)
    )
    deleted = result.rowcount
    conn.commit()
    conn.close()
    return deleted > 0




def graph_remove_relationship(from_name, to_name, relation_type=None):
    """Remove a relationship."""
    conn = get_db()

    # Get entity IDs
    from_id = conn.execute(
        "SELECT id FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (from_name,)
    ).fetchone()
    to_id = conn.execute(
        "SELECT id FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (to_name,)
    ).fetchone()

    if not from_id or not to_id:
        conn.close()
        return False

    if relation_type:
        result = conn.execute(
            "DELETE FROM graph_relationships WHERE from_entity_id = ? AND to_entity_id = ? AND relation_type = ?",
            (from_id[0], to_id[0], relation_type)
        )
    else:
        result = conn.execute(
            "DELETE FROM graph_relationships WHERE from_entity_id = ? AND to_entity_id = ?",
            (from_id[0], to_id[0])
        )

    deleted = result.rowcount
    conn.commit()
    conn.close()
    return deleted > 0




def graph_remove_fact(entity_name, key):
    """Remove a fact from an entity."""
    conn = get_db()

    entity = conn.execute(
        "SELECT id FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (entity_name,)
    ).fetchone()

    if not entity:
        conn.close()
        return False

    result = conn.execute(
        "DELETE FROM graph_facts WHERE entity_id = ? AND key = ?",
        (entity[0], key)
    )
    deleted = result.rowcount
    conn.commit()
    conn.close()
    return deleted > 0




def graph_spread(start_entity_name, depth=2):
    """
    Spreading activation: starting from an entity, find connected entities
    up to `depth` hops. Returns entities with activation scores (closer = higher).

    depth=1: direct connections only
    depth=2: friends-of-friends (default)
    depth=3: 3 hops

    Activation decays by 0.5 per hop.
    """
    conn = get_db()

    # Get start entity
    start = conn.execute(
        "SELECT id, name, type FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (start_entity_name,)
    ).fetchone()

    if not start:
        conn.close()
        return []

    start_id = start[0]

    # BFS to find connected entities
    visited = {start_id: 1.0}  # entity_id -> activation score
    queue = [(start_id, 0)]  # (entity_id, current_depth)

    while queue:
        entity_id, current_depth = queue.pop(0)

        if current_depth >= depth:
            continue

        # Get neighbors (both directions)
        neighbors = conn.execute(
            """SELECT DISTINCT
                   CASE
                       WHEN from_entity_id = ? THEN to_entity_id
                       ELSE from_entity_id
                   END as neighbor_id
               FROM graph_relationships
               WHERE from_entity_id = ? OR to_entity_id = ?""",
            (entity_id, entity_id, entity_id)
        ).fetchall()

        decay = 0.5 ** (current_depth + 1)

        for neighbor in neighbors:
            neighbor_id = neighbor[0]
            if neighbor_id not in visited:
                visited[neighbor_id] = decay
                queue.append((neighbor_id, current_depth + 1))
            else:
                # Update activation if this path gives higher score
                visited[neighbor_id] = max(visited[neighbor_id], decay)

    # Get entity details for all visited (except start)
    visited.pop(start_id, None)

    if not visited:
        conn.close()
        return []

    placeholders = ','.join('?' * len(visited))
    entities = conn.execute(
        f"SELECT * FROM graph_entities WHERE id IN ({placeholders})",
        list(visited.keys())
    ).fetchall()

    conn.close()

    # Build result with activation scores
    result = []
    for e in entities:
        entity_dict = dict(e)
        entity_dict['activation'] = visited[e['id']]
        result.append(entity_dict)

    # Sort by activation score (highest first)
    result.sort(key=lambda x: -x['activation'])

    return result




def link_memory_to_entity(memory_id, entity_name):
    """Link a memory to a graph entity."""
    entity_id = graph_get_or_create_entity(entity_name)
    if not entity_id:
        return False

    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO memory_entity_links (memory_id, entity_id) VALUES (?, ?)",
            (memory_id, entity_id)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error linking memory to entity: {e}")
        conn.close()
        return False




def auto_link_memory(memory_id, content):
    """Auto-detect entity mentions in content and create links."""
    conn = get_db()

    # Get all entity names
    entities = conn.execute("SELECT id, name FROM graph_entities").fetchall()

    content_lower = content.lower()
    linked_count = 0

    for entity in entities:
        entity_id, entity_name = entity[0], entity[1]

        # Simple substring match (case-insensitive)
        if entity_name.lower() in content_lower:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_entity_links (memory_id, entity_id) VALUES (?, ?)",
                    (memory_id, entity_id)
                )
                linked_count += 1
            except sqlite3.Error:
                pass

    conn.commit()
    conn.close()
    return linked_count




def graph_auto_link_all():
    """Auto-link all existing memories to entities."""
    conn = get_db()
    memories = conn.execute("SELECT id, content FROM memories WHERE active = 1").fetchall()
    conn.close()

    total_links = 0
    for mem in memories:
        links = auto_link_memory(mem[0], mem[1])
        total_links += links

    return total_links, len(memories)




def graph_import_openclaw():
    """Import entities, relationships, and facts from OpenClaw's graph DB."""
    from .config import OPENCLAW_GRAPH_DB
    openclaw_path = OPENCLAW_GRAPH_DB

    if not openclaw_path.exists():
        print(f"OpenClaw graph DB not found at {openclaw_path}")
        return

    try:
        source = sqlite3.connect(str(openclaw_path))
        source.row_factory = sqlite3.Row

        # Import entities
        entities = source.execute("SELECT * FROM entities").fetchall()
        entity_map = {}  # old_id -> new_id

        for e in entities:
            new_id = graph_add_entity(
                e['name'],
                e['type'],
                e['summary'] or "",
                e['importance'] or 3
            )
            entity_map[e['id']] = new_id

        print(f"Imported {len(entities)} entities")

        # Import relationships
        relationships = source.execute("SELECT * FROM relationships").fetchall()
        for r in relationships:
            from_id = entity_map.get(r['from_entity_id'])
            to_id = entity_map.get(r['to_entity_id'])

            if from_id and to_id:
                conn = get_db()
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO graph_relationships
                           (from_entity_id, to_entity_id, relation_type, note)
                           VALUES (?, ?, ?, ?)""",
                        (from_id, to_id, r['relation_type'], r['note'] or "")
                    )
                    conn.commit()
                    conn.close()
                except sqlite3.Error:
                    conn.close()

        print(f"Imported {len(relationships)} relationships")

        # Import facts
        facts = source.execute("SELECT * FROM facts").fetchall()
        imported_facts = 0
        for f in facts:
            entity_id = entity_map.get(f['entity_id'])
            if entity_id:
                conn = get_db()
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO graph_facts
                           (entity_id, key, value, confidence, source)
                           VALUES (?, ?, ?, ?, ?)""",
                        (entity_id, f['key'], f['value'], f['confidence'] or 1.0, f['source'] or "")
                    )
                    conn.commit()
                    conn.close()
                    imported_facts += 1
                except sqlite3.Error:
                    conn.close()

        print(f"Imported {imported_facts} facts")

        source.close()

    except sqlite3.Error as e:
        print(f"Error importing from OpenClaw: {e}")




def graph_stats():
    """Get graph statistics."""
    conn = get_db()

    entities = conn.execute("SELECT COUNT(*) FROM graph_entities").fetchone()[0]
    relationships = conn.execute("SELECT COUNT(*) FROM graph_relationships").fetchone()[0]
    facts = conn.execute("SELECT COUNT(*) FROM graph_facts").fetchone()[0]
    links = conn.execute("SELECT COUNT(*) FROM memory_entity_links").fetchone()[0]

    # Entity breakdown by type
    by_type = conn.execute(
        "SELECT type, COUNT(*) as count FROM graph_entities GROUP BY type ORDER BY count DESC"
    ).fetchall()

    conn.close()

    return {
        'entities': entities,
        'relationships': relationships,
        'facts': facts,
        'memory_links': links,
        'by_type': [dict(t) for t in by_type]
    }


# ── Smart MEMORY.md Export ──


