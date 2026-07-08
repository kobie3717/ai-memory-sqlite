"""Contradiction journal for deferred active-forgetting mechanism."""

import sqlite3
import hashlib
import os
from typing import Dict, List, Optional
from datetime import datetime
from .database import get_db
from .config import get_logger

logger = get_logger(__name__)


def ensure_contradiction_table(conn: sqlite3.Connection) -> None:
    """Create contradiction_journal table if not exists."""
    conn.execute('''CREATE TABLE IF NOT EXISTS contradiction_journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER NOT NULL,
        expected TEXT,
        found TEXT,
        context_hash TEXT NOT NULL,
        session_id TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        reconciled INTEGER DEFAULT 0
    )''')
    conn.execute('''CREATE INDEX IF NOT EXISTS idx_contradiction_memory
                    ON contradiction_journal(memory_id)''')
    conn.execute('''CREATE INDEX IF NOT EXISTS idx_contradiction_reconciled
                    ON contradiction_journal(reconciled)''')
    conn.commit()


def _get_context_hash() -> str:
    """Generate context hash from environment or session identifiers.

    Uses CLAUDE_SESSION_ID or falls back to hostname+timestamp-based hash.
    """
    # Try session ID from environment
    session_id = os.getenv('CLAUDE_SESSION_ID') or os.getenv('TERM_SESSION_ID')
    if session_id:
        return hashlib.sha256(session_id.encode()).hexdigest()[:16]

    # Fallback: hostname + current hour (groups within same hour)
    import socket
    hostname = socket.gethostname()
    hour_marker = datetime.now().strftime('%Y%m%d%H')
    combined = f"{hostname}:{hour_marker}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def log_contradiction(
    memory_id: int,
    expected: Optional[str] = None,
    found: Optional[str] = None,
    context_hash: Optional[str] = None,
    session_id: Optional[str] = None
) -> int:
    """Append contradiction without deleting. Safe to call mid-execution.

    Args:
        memory_id: ID of the memory being contradicted
        expected: What the memory claimed
        found: What was actually observed
        context_hash: Unique identifier for this execution context (auto-generated if None)
        session_id: Optional session identifier for tracking

    Returns:
        ID of the created contradiction journal entry
    """
    conn = get_db()
    ensure_contradiction_table(conn)

    if context_hash is None:
        context_hash = _get_context_hash()

    if session_id is None:
        session_id = os.getenv('CLAUDE_SESSION_ID') or os.getenv('TERM_SESSION_ID')

    cursor = conn.execute('''
        INSERT INTO contradiction_journal
        (memory_id, expected, found, context_hash, session_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (memory_id, expected, found, context_hash, session_id))

    contradiction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    logger.info(f"Logged contradiction for memory #{memory_id} (context: {context_hash[:8]}...)")
    return contradiction_id


def reconcile(dry_run: bool = False) -> Dict[str, any]:
    """Run at task boundary to reconcile contradictions.

    Logic:
    - Memories with 2+ distinct context_hashes -> soft-delete (set active=0)
    - Memories with 1 contradiction -> flag for review (add 'contradiction-flagged' tag)
    - Mark processed entries as reconciled=1

    Args:
        dry_run: If True, show what would happen without acting

    Returns:
        Dict with keys: deleted (list of IDs), flagged (list of IDs), skipped (int)
    """
    conn = get_db()
    ensure_contradiction_table(conn)

    # Find unreconciled contradictions grouped by memory_id
    rows = conn.execute('''
        SELECT memory_id,
               COUNT(DISTINCT context_hash) as unique_contexts,
               GROUP_CONCAT(DISTINCT context_hash) as context_hashes,
               COUNT(*) as total_count
        FROM contradiction_journal
        WHERE reconciled = 0
        GROUP BY memory_id
    ''').fetchall()

    deleted = []
    flagged = []
    skipped = 0

    for row in rows:
        memory_id = row['memory_id']
        unique_contexts = row['unique_contexts']
        context_hashes = row['context_hashes']
        total_count = row['total_count']

        # Check if memory still exists and is active
        memory = conn.execute(
            'SELECT id, active, tags FROM memories WHERE id = ?',
            (memory_id,)
        ).fetchone()

        if not memory:
            skipped += 1
            logger.debug(f"Memory #{memory_id} no longer exists, skipping")
            # Mark as reconciled anyway to clean up
            if not dry_run:
                conn.execute(
                    'UPDATE contradiction_journal SET reconciled = 1 WHERE memory_id = ?',
                    (memory_id,)
                )
            continue

        if memory['active'] == 0:
            skipped += 1
            logger.debug(f"Memory #{memory_id} already inactive, skipping")
            # Mark as reconciled
            if not dry_run:
                conn.execute(
                    'UPDATE contradiction_journal SET reconciled = 1 WHERE memory_id = ?',
                    (memory_id,)
                )
            continue

        # Decision logic
        if unique_contexts >= 2:
            # Multiple independent contradictions -> soft delete
            deleted.append(memory_id)
            if dry_run:
                logger.info(f"[DRY RUN] Would delete #{memory_id} ({unique_contexts} contradictions from different contexts)")
            else:
                conn.execute(
                    'UPDATE memories SET active = 0 WHERE id = ?',
                    (memory_id,)
                )
                conn.execute(
                    'UPDATE contradiction_journal SET reconciled = 1 WHERE memory_id = ?',
                    (memory_id,)
                )
                logger.info(f"Deleted #{memory_id} ({unique_contexts} contradictions)")

        elif unique_contexts == 1:
            # Single context contradiction -> flag for review
            flagged.append(memory_id)
            if dry_run:
                logger.info(f"[DRY RUN] Would flag #{memory_id} for review (1 contradiction)")
            else:
                # Add contradiction-flagged tag if not already present
                current_tags = memory['tags'] or ''
                tags_list = [t.strip() for t in current_tags.split(',') if t.strip()]
                if 'contradiction-flagged' not in tags_list:
                    tags_list.append('contradiction-flagged')
                    new_tags = ','.join(tags_list)
                    conn.execute(
                        'UPDATE memories SET tags = ? WHERE id = ?',
                        (new_tags, memory_id)
                    )

                conn.execute(
                    'UPDATE contradiction_journal SET reconciled = 1 WHERE memory_id = ?',
                    (memory_id,)
                )
                logger.info(f"Flagged #{memory_id} for review (1 contradiction)")

    if not dry_run:
        conn.commit()

    conn.close()

    result = {
        'deleted': deleted,
        'flagged': flagged,
        'skipped': skipped
    }

    return result


def show_contradiction_journal(limit: int = 20) -> None:
    """Show unreconciled contradictions.

    Args:
        limit: Maximum number of entries to show
    """
    conn = get_db()
    ensure_contradiction_table(conn)

    rows = conn.execute('''
        SELECT cj.id, cj.memory_id, cj.expected, cj.found,
               cj.context_hash, cj.session_id, cj.created_at,
               m.content as memory_content, m.active
        FROM contradiction_journal cj
        LEFT JOIN memories m ON cj.memory_id = m.id
        WHERE cj.reconciled = 0
        ORDER BY cj.created_at DESC
        LIMIT ?
    ''', (limit,)).fetchall()

    conn.close()

    if not rows:
        print("No unreconciled contradictions.")
        return

    print(f"\n{'='*80}")
    print(f"CONTRADICTION JOURNAL (showing {len(rows)} unreconciled entries)")
    print(f"{'='*80}\n")

    for row in rows:
        print(f"Contradiction #{row['id']} | Memory #{row['memory_id']} | {row['created_at']}")
        print(f"  Context: {row['context_hash'][:16]}...")
        if row['session_id']:
            print(f"  Session: {row['session_id'][:32]}...")

        if row['memory_content']:
            content_preview = row['memory_content'][:80]
            active_status = "ACTIVE" if row['active'] else "INACTIVE"
            print(f"  Memory: [{active_status}] {content_preview}...")
        else:
            print(f"  Memory: [DELETED]")

        if row['expected']:
            print(f"  Expected: {row['expected'][:80]}")
        if row['found']:
            print(f"  Found: {row['found'][:80]}")
        print()

    # Show summary statistics
    conn = get_db()
    stats = conn.execute('''
        SELECT memory_id,
               COUNT(*) as contradiction_count,
               COUNT(DISTINCT context_hash) as unique_contexts
        FROM contradiction_journal
        WHERE reconciled = 0
        GROUP BY memory_id
        ORDER BY unique_contexts DESC, contradiction_count DESC
    ''').fetchall()
    conn.close()

    if stats:
        print(f"\n{'='*80}")
        print("SUMMARY BY MEMORY")
        print(f"{'='*80}\n")
        for stat in stats[:10]:  # Top 10 most contradicted
            print(f"  Memory #{stat['memory_id']}: {stat['contradiction_count']} contradictions from {stat['unique_contexts']} context(s)")
