"""OpenClaw bridge bidirectional sync (Phase 4 feature)."""

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


def load_sync_state():
    """Load sync state (checksums) from JSON."""
    if SYNC_STATE_FILE.exists():
        try:
            return json.loads(SYNC_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}  # Invalid JSON or read error
    return {}




def save_sync_state(state):
    """Save sync state to JSON."""
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))




def file_checksum(content):
    """Generate MD5 checksum for content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()




def sync_to_openclaw():
    """Export memories and graph data to OpenClaw's workspace format."""
    if not OPENCLAW_MEMORY_DIR.exists():
        print(f"OpenClaw memory directory not found: {OPENCLAW_MEMORY_DIR}")
        return

    conn = get_db()
    state = load_sync_state()
    files_written = []

    # 1. Export claude-code-bridge.md (session handoff)
    bridge_path = OPENCLAW_MEMORY_DIR / "claude-code-bridge.md"

    # Get recent changes
    recent_mems = conn.execute("""
        SELECT * FROM memories
        WHERE active = 1 AND updated_at >= datetime('now', '-24 hours')
        ORDER BY updated_at DESC LIMIT 20
    """).fetchall()

    recent_snaps = conn.execute("""
        SELECT * FROM session_snapshots
        ORDER BY created_at DESC LIMIT 3
    """).fetchall()

    # Get pending items
    pending = conn.execute("""
        SELECT * FROM memories
        WHERE active = 1 AND category = 'pending'
        ORDER BY priority DESC, created_at DESC LIMIT 15
    """).fetchall()

    # Get recent decisions
    decisions = conn.execute("""
        SELECT * FROM memories
        WHERE active = 1 AND category = 'decision'
        ORDER BY updated_at DESC LIMIT 10
    """).fetchall()

    # Build bridge content
    bridge_lines = []
    bridge_lines.append("# Claude Code Memory Bridge")
    bridge_lines.append(f"_Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    bridge_lines.append("")

    # Active context from recent snapshots
    if recent_snaps:
        bridge_lines.append("## Active Context")
        for snap in recent_snaps[:2]:
            bridge_lines.append(f"- **{snap['created_at'][:16]}**: {snap['summary']}")
            if snap['project']:
                bridge_lines.append(f"  - Project: {snap['project']}")
        bridge_lines.append("")

    # Recent changes
    if recent_mems:
        bridge_lines.append("## Recent Changes (Last 24h)")
        for mem in recent_mems[:10]:
            cat = mem['category']
            proj = f" [{mem['project']}]" if mem['project'] else ""
            bridge_lines.append(f"- [{cat}] {mem['content'][:120]}{proj}")
        bridge_lines.append("")

    # Key decisions
    if decisions:
        bridge_lines.append("## Key Decisions")
        for dec in decisions:
            proj = f" [{dec['project']}]" if dec['project'] else ""
            bridge_lines.append(f"- {dec['content']}{proj}")
        bridge_lines.append("")

    # Pending items
    if pending:
        bridge_lines.append("## Pending Items")
        for p in pending:
            proj = f" [{p['project']}]" if p['project'] else ""
            exp = ""
            if p['expires_at']:
                exp_date = p['expires_at'][:10]
                if p['expires_at'] < datetime.now().isoformat():
                    exp = " (EXPIRED)"
                else:
                    exp = f" (due {exp_date})"
            bridge_lines.append(f"- [ ] {p['content']}{proj}{exp}")
        bridge_lines.append("")

    # Graph summary
    g_stats = graph_stats()
    bridge_lines.append("## Shared Graph")
    bridge_lines.append(f"- Entities: {g_stats['entities']} | Relationships: {g_stats['relationships']} | Facts: {g_stats['facts']}")
    if g_stats['by_type']:
        bridge_lines.append(f"- Entity types: " + ", ".join([f"{t['type']}({t['count']})" for t in g_stats['by_type']]))
    bridge_lines.append("")

    # Memory stats
    total = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1").fetchone()['c']
    stale = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1 AND stale = 1").fetchone()['c']
    bridge_lines.append("## Memory Stats")
    bridge_lines.append(f"- Total active: {total} | Stale: {stale}")
    bridge_lines.append(f"- Source: `{DB_PATH}`")
    bridge_lines.append("")

    bridge_content = "\n".join(bridge_lines)

    # Write if changed
    old_checksum = state.get('bridge_checksum', '')
    new_checksum = file_checksum(bridge_content)
    if old_checksum != new_checksum:
        bridge_path.write_text(bridge_content)
        state['bridge_checksum'] = new_checksum
        state['last_sync'] = datetime.now().isoformat()
        files_written.append(str(bridge_path))

    # 2. Export graph-sync.md (graph data in human-readable format)
    graph_path = OPENCLAW_MEMORY_DIR / "graph-sync.md"

    entities = conn.execute("""
        SELECT * FROM graph_entities
        ORDER BY importance DESC, updated_at DESC
    """).fetchall()

    graph_lines = []
    graph_lines.append("# Claude Code Graph Export")
    graph_lines.append(f"_Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    graph_lines.append("")
    graph_lines.append(f"Total: {g_stats['entities']} entities, {g_stats['relationships']} relationships, {g_stats['facts']} facts")
    graph_lines.append("")

    # Group entities by type
    for entity_type in ['project', 'person', 'org', 'feature', 'tool', 'service', 'concept']:
        type_entities = [e for e in entities if e['type'] == entity_type]
        if type_entities:
            graph_lines.append(f"## {entity_type.title()}s")
            for e in type_entities:
                graph_lines.append(f"### {e['name']}")
                if e['summary']:
                    graph_lines.append(f"{e['summary']}")
                graph_lines.append(f"_Importance: {e['importance']}, Updated: {e['updated_at'][:16]}_")

                # Facts
                facts = conn.execute(
                    "SELECT * FROM graph_facts WHERE entity_id = ? ORDER BY key",
                    (e['id'],)
                ).fetchall()
                if facts:
                    graph_lines.append("**Facts:**")
                    for f in facts:
                        conf = f" (confidence: {f['confidence']})" if f['confidence'] < 1.0 else ""
                        graph_lines.append(f"- {f['key']}: {f['value']}{conf}")

                # Relationships
                rels_out = conn.execute("""
                    SELECT r.relation_type, r.note, e2.name, e2.type
                    FROM graph_relationships r
                    JOIN graph_entities e2 ON r.to_entity_id = e2.id
                    WHERE r.from_entity_id = ?
                """, (e['id'],)).fetchall()

                rels_in = conn.execute("""
                    SELECT r.relation_type, r.note, e1.name, e1.type
                    FROM graph_relationships r
                    JOIN graph_entities e1 ON r.from_entity_id = e1.id
                    WHERE r.to_entity_id = ?
                """, (e['id'],)).fetchall()

                if rels_out:
                    graph_lines.append("**Relationships (outgoing):**")
                    for r in rels_out:
                        note = f" - {r['note']}" if r['note'] else ""
                        graph_lines.append(f"- --{r['relation_type']}--> {r['name']} ({r['type']}){note}")

                if rels_in:
                    graph_lines.append("**Relationships (incoming):**")
                    for r in rels_in:
                        note = f" - {r['note']}" if r['note'] else ""
                        graph_lines.append(f"- <--{r['relation_type']}-- {r['name']} ({r['type']}){note}")

                graph_lines.append("")

    graph_content = "\n".join(graph_lines)

    # Write if changed
    old_checksum = state.get('graph_checksum', '')
    new_checksum = file_checksum(graph_content)
    if old_checksum != new_checksum:
        graph_path.write_text(graph_content)
        state['graph_checksum'] = new_checksum
        files_written.append(str(graph_path))

    # 3. Sync graph DB to OpenClaw's graph DB
    if OPENCLAW_GRAPH_DB.exists():
        try:
            graph_sync_to_openclaw_db()
            files_written.append(str(OPENCLAW_GRAPH_DB))
        except Exception as e:
            print(f"Warning: Graph DB sync failed: {e}")

    # Save state
    save_sync_state(state)
    conn.close()

    if files_written:
        print(f"Synced {len(files_written)} files to OpenClaw:")
        for f in files_written:
            print(f"  - {f}")
    else:
        print("No changes to sync (all files up to date)")




def graph_sync_to_openclaw_db():
    """Sync graph entities/relationships/facts to OpenClaw's graph DB."""
    if not OPENCLAW_GRAPH_DB.exists():
        print("OpenClaw graph DB not found, skipping DB sync")
        return

    source_conn = get_db()
    target_conn = sqlite3.connect(str(OPENCLAW_GRAPH_DB))
    target_conn.row_factory = sqlite3.Row

    # Ensure OpenClaw DB has the same schema
    target_conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL,
            summary TEXT DEFAULT '',
            importance INTEGER DEFAULT 3,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_entity_id INTEGER NOT NULL,
            to_entity_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            note TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(from_entity_id, to_entity_id, relation_type)
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(entity_id, key)
        );

        CREATE TABLE IF NOT EXISTS fact_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            old_value TEXT NOT NULL,
            new_value TEXT NOT NULL,
            changed_at TEXT DEFAULT (datetime('now'))
        );
    """)
    target_conn.commit()

    # Map entity names to IDs in both DBs
    entity_map = {}  # source_id -> target_id

    # Sync entities
    source_entities = source_conn.execute("SELECT * FROM graph_entities").fetchall()
    for e in source_entities:
        # Map entity types to OpenClaw's supported types
        entity_type = e['type']
        if entity_type not in ('person', 'project', 'org', 'feature', 'concept'):
            # Map 'tool' and 'service' to 'concept'
            entity_type = 'concept'

        # Check if entity exists in target
        existing = target_conn.execute(
            "SELECT id FROM entities WHERE name = ?", (e['name'],)
        ).fetchone()

        if existing:
            # Update existing
            target_conn.execute("""
                UPDATE entities SET type = ?, summary = ?, importance = ?, updated_at = ?
                WHERE name = ?
            """, (entity_type, e['summary'], e['importance'], e['updated_at'], e['name']))
            entity_map[e['id']] = existing['id']
        else:
            # Insert new
            cursor = target_conn.execute("""
                INSERT INTO entities (name, type, summary, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (e['name'], entity_type, e['summary'], e['importance'], e['created_at'], e['updated_at']))
            entity_map[e['id']] = cursor.lastrowid

    target_conn.commit()

    # Sync relationships
    source_rels = source_conn.execute("SELECT * FROM graph_relationships").fetchall()
    for r in source_rels:
        from_id = entity_map.get(r['from_entity_id'])
        to_id = entity_map.get(r['to_entity_id'])

        if from_id and to_id:
            try:
                target_conn.execute("""
                    INSERT OR REPLACE INTO relationships
                    (from_entity_id, to_entity_id, relation_type, note, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (from_id, to_id, r['relation_type'], r['note'], r['created_at']))
            except sqlite3.Error:
                pass

    target_conn.commit()

    # Sync facts
    source_facts = source_conn.execute("SELECT * FROM graph_facts").fetchall()
    for f in source_facts:
        entity_id = entity_map.get(f['entity_id'])
        if entity_id:
            # Check if fact exists
            existing = target_conn.execute(
                "SELECT value FROM facts WHERE entity_id = ? AND key = ?",
                (entity_id, f['key'])
            ).fetchone()

            if existing and existing['value'] != f['value']:
                # Record history
                target_conn.execute("""
                    INSERT INTO fact_history (entity_id, key, old_value, new_value)
                    VALUES (?, ?, ?, ?)
                """, (entity_id, f['key'], existing['value'], f['value']))

            # Upsert fact
            target_conn.execute("""
                INSERT OR REPLACE INTO facts
                (entity_id, key, value, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (entity_id, f['key'], f['value'], f['confidence'], f['source'],
                  f['created_at'], f['updated_at']))

    target_conn.commit()
    target_conn.close()
    source_conn.close()




def sync_from_openclaw():
    """Import new memories from OpenClaw's daily notes and topic files."""
    if not OPENCLAW_MEMORY_DIR.exists():
        print(f"OpenClaw memory directory not found: {OPENCLAW_MEMORY_DIR}")
        return

    imported_count = 0

    # Read today's daily note
    today = datetime.now().strftime("%Y-%m-%d")
    daily_path = OPENCLAW_MEMORY_DIR / f"{today}.md"

    if daily_path.exists():
        content = daily_path.read_text()

        # Simple pattern matching for structured entries
        # Look for patterns like "## [Category]" or "**Decision:**" etc.

        # Example: extract decision-like statements
        for line in content.split('\n'):
            line = line.strip()

            # Look for decision markers
            if line.startswith('**Decision:**') or line.startswith('**Recommendation:**'):
                decision = line.split(':', 1)[1].strip()
                if decision and len(decision) > 20:
                    # Check if similar memory exists
                    similar = find_similar(decision, category='decision', threshold=0.75)
                    if not similar:
                        add_memory('decision', decision, project='OpenClaw', source='openclaw-import')
                        imported_count += 1

            # Look for todo items
            elif line.startswith('- [ ]'):
                todo = line[5:].strip()
                if todo and len(todo) > 15:
                    similar = find_similar(todo, category='pending', threshold=0.75)
                    if not similar:
                        add_memory('pending', todo, project='OpenClaw', source='openclaw-import')
                        imported_count += 1

    # Import from OpenClaw's graph DB
    if OPENCLAW_GRAPH_DB.exists():
        try:
            graph_import_openclaw()
        except Exception as e:
            print(f"Warning: Failed to import from OpenClaw graph DB: {e}")

    if imported_count > 0:
        print(f"Imported {imported_count} new memories from OpenClaw")
    else:
        print("No new memories to import from OpenClaw")




def sync_bidirectional():
    """Run both sync-to and sync-from."""
    print("=== Syncing to OpenClaw ===")
    sync_to_openclaw()
    print("\n=== Syncing from OpenClaw ===")
    sync_from_openclaw()
    print("\n=== Sync complete ===")



