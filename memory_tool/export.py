"""Export, maintenance, and housekeeping operations."""
from typing import Optional, List, Dict, Any, Tuple, Callable

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
from .embedding import embed_and_store, embed_text, embed_texts_batch, semantic_search
from .phase_context import detect_phase, counterfactual_score, survivor_score

logger = get_logger(__name__)

# Lazy imports for optional dependencies
try:
    import numpy as np
    import sqlite_vec
except ImportError:
    pass


# Lazy imports to avoid circular dependency
def _get_relations_functions() -> Tuple[Any, Any]:
    """Lazy import of relations functions to avoid circular dependency."""
    from .relations import find_conflicts
    from .snapshots import get_snapshots
    return find_conflicts, get_snapshots


def export_memory_md_phased(focus_project: Optional[str] = None) -> None:
    """Phase-aware memory export with counterfactual scoring.

    Splits the 5KB budget into:
    - ~2KB survivor layer (highest counterfactual score across ALL phases)
    - ~3KB phase-specific slice (ranked by counterfactual score FOR current phase)
    """
    conn = get_db()

    # Detect current phase
    phase = detect_phase()

    # Header sections (session, projects, pending) - keep these from original
    header_lines = []
    header_lines.append("# Persistent Memory (Auto-Generated)")
    header_lines.append(f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                       f"Phase: {phase} | v5: phase-aware counterfactual scoring_")
    header_lines.append("")

    # Latest session snapshot
    snaps = conn.execute("SELECT * FROM session_snapshots ORDER BY created_at DESC LIMIT 3").fetchall()
    if snaps:
        header_lines.append(f"## Last Session ({snaps[0]['created_at'][:16]})")
        header_lines.append(f"{snaps[0]['summary']}")
        if snaps[0]["project"]:
            header_lines.append(f"Project: {snaps[0]['project']}")
        header_lines.append("")

    # Projects (minimal - just list names, details in phase-specific section)
    projects = conn.execute(
        "SELECT DISTINCT project FROM memories WHERE active = 1 AND project IS NOT NULL ORDER BY project"
    ).fetchall()
    if projects:
        header_lines.append("## Active Projects")
        proj_names = [p["project"] for p in projects]
        # Show first 5 projects, rest as count
        if len(proj_names) <= 5:
            header_lines.append(", ".join(proj_names))
        else:
            header_lines.append(", ".join(proj_names[:5]) + f" (+{len(proj_names)-5} more)")
        header_lines.append("")

    # Pending count only - details in phase-specific section
    pending_count = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE active = 1 AND category = 'pending'"
    ).fetchone()["c"]
    if pending_count:
        header_lines.append(f"## Pending: {pending_count} items")
        header_lines.append("_(Details in phase-specific section below)_")
        header_lines.append("")

    header_text = "\n".join(header_lines)
    header_bytes = len(header_text.encode('utf-8'))

    # Calculate budgets - reserve more for content sections
    total_budget = MAX_MEMORY_MD_BYTES
    footer_reserve = 300  # Reserve for footer
    header_cap = 800  # Cap header at 800 bytes to ensure room for content

    # If header is too large, truncate it
    if header_bytes > header_cap:
        logger.warning(f"Header too large ({header_bytes}B), capping at {header_cap}B")
        header_text = header_text[:header_cap]
        header_bytes = header_cap

    available_budget = total_budget - header_bytes - footer_reserve
    survivor_budget = int(available_budget * 0.4)  # 40% for survivors
    phase_budget = available_budget - survivor_budget  # 60% for phase-specific

    # Fetch ALL active non-stale memories
    all_memories = conn.execute("""
        SELECT * FROM memories
        WHERE active = 1 AND stale = 0
        ORDER BY priority DESC, updated_at DESC
    """).fetchall()

    # Convert to list of dicts for easier processing
    memories = [dict(m) for m in all_memories]

    # Score each memory
    for m in memories:
        m['_survivor_score'] = survivor_score(m)
        m['_phase_score'] = counterfactual_score(m, phase)

    # Build survivor layer - highest survivor scores
    survivors_sorted = sorted(memories, key=lambda m: m['_survivor_score'], reverse=True)
    survivor_memories = []
    survivor_ids = set()
    survivor_bytes = 0

    for m in survivors_sorted:
        # Format memory line
        proj = f" [{m['project']}]" if m.get("project") else ""
        tag = f" ({m['tags']})" if m.get("tags") else ""
        acc = f" [x{m['access_count']}]" if m.get('access_count', 0) > 2 else ""
        line = f"- [{m['category']}] {m['content']}{proj}{tag}{acc}\n"
        line_bytes = len(line.encode('utf-8'))

        if survivor_bytes + line_bytes > survivor_budget:
            break

        survivor_memories.append((m, line))
        survivor_ids.add(m['id'])
        survivor_bytes += line_bytes

    # Build phase-specific slice - highest phase scores, excluding survivors
    phase_candidates = [m for m in memories if m['id'] not in survivor_ids]
    phase_sorted = sorted(phase_candidates, key=lambda m: m['_phase_score'], reverse=True)
    phase_memories = []
    phase_bytes = 0

    for m in phase_sorted:
        proj = f" [{m['project']}]" if m.get("project") else ""
        tag = f" ({m['tags']})" if m.get("tags") else ""
        acc = f" [x{m['access_count']}]" if m.get('access_count', 0) > 2 else ""
        line = f"- [{m['category']}] {m['content']}{proj}{tag}{acc}\n"
        line_bytes = len(line.encode('utf-8'))

        if phase_bytes + line_bytes > phase_budget:
            break

        phase_memories.append((m, line))
        phase_bytes += line_bytes

    # Assemble final output
    output_lines = [header_text]

    # Phase-specific section
    if phase_memories:
        output_lines.append(f"## Core Context (Phase: {phase})")
        output_lines.append(f"_Top memories for {phase} work (score: phase-specific counterfactual)_")
        output_lines.append("")
        for m, line in phase_memories:
            output_lines.append(line.rstrip())
        output_lines.append("")

    # Survivor layer
    if survivor_memories:
        output_lines.append("## Always-Load (Critical)")
        output_lines.append("_High-stakes memories that persist across all phases_")
        output_lines.append("")
        for m, line in survivor_memories:
            output_lines.append(line.rstrip())
        output_lines.append("")

    # Footer with stats
    total = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1").fetchone()["c"]
    stale_count = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1 AND stale = 1").fetchone()["c"]

    output_lines.append("---")
    output_lines.append(f"_Phase: {phase} | Survivors: {len(survivor_memories)} | Phase-specific: {len(phase_memories)} | Total DB: {total} memories_")
    if stale_count:
        output_lines.append(f"_{stale_count} stale memories hidden. Run `memory-tool stale` to review._")
    output_lines.append(f"_Manage: `memory-tool help`_")

    content = "\n".join(output_lines)

    # Final safety check
    if len(content.encode('utf-8')) > MAX_MEMORY_MD_BYTES:
        # Truncate with warning
        max_chars = MAX_MEMORY_MD_BYTES - 100
        content = content[:max_chars] + "\n\n_[Over budget — run `memory-tool topics` for full view]_"

    conn.close()
    MEMORY_MD_PATH.write_text(content + "\n")


def export_memory_md(focus_project: Optional[str] = None) -> None:
    conn = get_db()
    lines = []
    lines.append("# Persistent Memory (Auto-Generated)")
    lines.append(f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                 f"v4: progressive disclosure, topic upserts, conflicts, smart ingest, topics export, budget cap_")
    lines.append("")

    # Latest session snapshot (limit to 3 recent, auto-prune older)
    snaps = conn.execute("SELECT * FROM session_snapshots ORDER BY created_at DESC LIMIT 3").fetchall()
    if snaps:
        lines.append(f"## Last Session ({snaps[0]['created_at'][:16]})")
        lines.append(f"{snaps[0]['summary']}")
        if snaps[0]["project"]:
            lines.append(f"Project: {snaps[0]['project']}")
        lines.append("")

    # Projects
    projects = conn.execute(
        "SELECT DISTINCT project FROM memories WHERE active = 1 AND project IS NOT NULL ORDER BY project"
    ).fetchall()
    if projects:
        lines.append("## Active Projects")
        for proj in projects:
            p = proj["project"]
            is_focus = (focus_project and p == focus_project)
            details = conn.execute(
                "SELECT * FROM memories WHERE active = 1 AND project = ? AND category = 'project' ORDER BY priority DESC, updated_at DESC",
                (p,)
            ).fetchall()
            marker = " (ACTIVE)" if is_focus else ""
            lines.append(f"### {p}{marker}")
            for d in details:
                lines.append(f"- {d['content']}")
            lines.append("")

    # Pending
    pending = conn.execute(
        "SELECT * FROM memories WHERE active = 1 AND category = 'pending' ORDER BY priority DESC, created_at DESC LIMIT 10"
    ).fetchall()
    if pending:
        lines.append("## Pending / TODO")
        for p in pending:
            proj = f" [{p['project']}]" if p["project"] else ""
            stale_mark = " (STALE)" if p["stale"] else ""
            exp = ""
            if p["expires_at"]:
                exp_date = p["expires_at"][:10]
                if p["expires_at"] < datetime.now().isoformat():
                    exp = " (EXPIRED)"
                else:
                    exp = f" (due {exp_date})"
            lines.append(f"- [ ] {p['content']}{proj}{stale_mark}{exp}")
        lines.append("")

    # Decisions (prioritize focus project)
    decisions = conn.execute("""
        SELECT * FROM memories WHERE active = 1 AND category = 'decision'
        ORDER BY CASE WHEN project = ? THEN 0 ELSE 1 END, priority DESC, updated_at DESC LIMIT 10
    """, (focus_project,)).fetchall()
    if decisions:
        lines.append("## Key Decisions")
        for d in decisions:
            proj = f" [{d['project']}]" if d["project"] else ""
            lines.append(f"- {d['content']}{proj}")
        lines.append("")

    # Preferences
    prefs = conn.execute(
        "SELECT * FROM memories WHERE active = 1 AND category = 'preference' ORDER BY priority DESC"
    ).fetchall()
    if prefs:
        lines.append("## User Preferences")
        for p in prefs:
            lines.append(f"- {p['content']}")
        lines.append("")

    # Errors & Learnings (limit to 5 if over budget)
    errors_limit = 10
    errors = conn.execute("""
        SELECT * FROM memories WHERE active = 1 AND stale = 0 AND category IN ('error', 'learning')
        ORDER BY CASE WHEN project = ? THEN 0 ELSE 1 END, access_count DESC, updated_at DESC LIMIT ?
    """, (focus_project, errors_limit)).fetchall()
    if errors:
        lines.append("## Errors & Learnings")
        for e in errors:
            proj = f" [{e['project']}]" if e["project"] else ""
            tag = f" ({e['tags']})" if e["tags"] else ""
            acc = f" [x{e['access_count']}]" if e["access_count"] > 2 else ""
            src = " [auto]" if e["source"] == "auto-hook" else ""
            lines.append(f"- [{e['category']}] {e['content']}{proj}{tag}{acc}{src}")
        lines.append("")

    # Architecture (limit to 4 if over budget)
    arch_limit = 8
    arch = conn.execute("""
        SELECT * FROM memories WHERE active = 1 AND category = 'architecture'
        ORDER BY CASE WHEN project = ? THEN 0 ELSE 1 END, priority DESC LIMIT ?
    """, (focus_project, arch_limit)).fetchall()
    if arch:
        lines.append("## Architecture")
        for a in arch:
            proj = f" [{a['project']}]" if a["project"] else ""
            lines.append(f"- {a['content']}{proj}")
        lines.append("")

    # Workflow (limit to 3 if over budget)
    workflow_limit = 6
    workflow = conn.execute("""
        SELECT * FROM memories WHERE active = 1 AND category = 'workflow'
        ORDER BY CASE WHEN project = ? THEN 0 ELSE 1 END, priority DESC LIMIT ?
    """, (focus_project, workflow_limit)).fetchall()
    if workflow:
        lines.append("## Workflow")
        for w in workflow:
            proj = f" [{w['project']}]" if w["project"] else ""
            lines.append(f"- {w['content']}{proj}")
        lines.append("")

    # Stale/expired counts
    stale_count = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1 AND stale = 1").fetchone()["c"]
    expired_count = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE active = 1 AND expires_at IS NOT NULL AND expires_at < datetime('now')"
    ).fetchone()["c"]
    notes = []
    if stale_count:
        notes.append(f"{stale_count} stale")
    if expired_count:
        notes.append(f"{expired_count} expired")
    if notes:
        lines.append(f"_{', '.join(notes)} memories hidden. Run `memory-tool stale` to review._")
        lines.append("")

    # Footer
    total = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1").fetchone()["c"]
    lines.append("---")
    lines.append(f"_Total: {total} memories | Manage: `memory-tool help`_")

    # Budget cap check (v4 Feature #6) - BEFORE closing connection
    content = "\n".join(lines)
    if len(content.encode('utf-8')) > MAX_MEMORY_MD_BYTES:
        # Simplified approach: just truncate with warning
        lines_text = "\n".join(lines)
        max_chars = MAX_MEMORY_MD_BYTES - 100  # Leave room for warning
        if len(lines_text) > max_chars:
            lines_text = lines_text[:max_chars]
            lines_text += "\n\n_[Over budget — run `memory-tool topics` for full view]_"
        content = lines_text

    conn.close()
    MEMORY_MD_PATH.write_text(content + "\n")


# ── Garbage Collection ──



def export_topics() -> None:
    """Generate topic .md files per project."""
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_db()

    # Get all projects
    projects = conn.execute(
        "SELECT DISTINCT project FROM memories WHERE active = 1 AND project IS NOT NULL ORDER BY project"
    ).fetchall()

    # Generate per-project topic files
    for proj_row in projects:
        project = proj_row["project"]
        filename = TOPICS_DIR / f"{project.lower().replace(' ', '_')}.md"

        lines = [f"# {project} (Auto-generated from memory DB)", ""]

        # Group by category
        categories = conn.execute(
            "SELECT DISTINCT category FROM memories WHERE active = 1 AND project = ? ORDER BY category",
            (project,)
        ).fetchall()

        for cat_row in categories:
            category = cat_row["category"]
            lines.append(f"## {category.title()}")

            mems = conn.execute(
                "SELECT * FROM memories WHERE active = 1 AND project = ? AND category = ? ORDER BY priority DESC, updated_at DESC",
                (project, category)
            ).fetchall()

            for m in mems:
                lines.append(f"- #{m['id']} {m['content']}")

            lines.append("")

        filename.write_text("\n".join(lines))
        logger.debug(f"Exported {filename}")

    # Special: people.md (all contacts)
    contacts = conn.execute(
        "SELECT * FROM memories WHERE active = 1 AND category = 'contact' ORDER BY project, priority DESC"
    ).fetchall()
    if contacts:
        lines = ["# People (Auto-generated from memory DB)", ""]
        for c in contacts:
            proj = f" [{c['project']}]" if c["project"] else ""
            lines.append(f"- #{c['id']} {c['content']}{proj}")
        (TOPICS_DIR / "people.md").write_text("\n".join(lines))
        logger.debug(f"Exported {TOPICS_DIR / 'people.md'}")

    # Special: todo.md (all pending)
    pending = conn.execute(
        "SELECT * FROM memories WHERE active = 1 AND category = 'pending' ORDER BY project, priority DESC"
    ).fetchall()
    if pending:
        lines = ["# TODO (Auto-generated from memory DB)", ""]
        for p in pending:
            proj = f" [{p['project']}]" if p["project"] else ""
            lines.append(f"- [ ] #{p['id']} {p['content']}{proj}")
        (TOPICS_DIR / "todo.md").write_text("\n".join(lines))
        logger.debug(f"Exported {TOPICS_DIR / 'todo.md'}")

    conn.close()
    logger.info(f"Topic files generated in {TOPICS_DIR}")


# ── Decay & Expiry (Upgrade #6: expiry added) ──



def run_decay() -> Dict[str, int]:
    """Run decay process using FSRS retention model with tier-based rules."""
    conn = get_db()
    now = datetime.now()
    changes = {"stale": 0, "deprioritized": 0, "expired": 0}

    # Expire items past their expiry date (keep existing logic, but skip pinned)
    cur = conn.execute("""
        UPDATE memories SET active = 0, stale = 0
        WHERE active = 1 AND expires_at IS NOT NULL AND expires_at < datetime('now')
        AND is_pinned = 0
    """)
    changes["expired"] = cur.rowcount

    # FSRS-based stale detection with tier-aware decay times
    # Semantic tier: immune to decay
    # Episodic tier: decay after 30 days (existing behavior)
    # Working tier: decay after 1 day
    # Pinned memories: immune to decay
    cur = conn.execute("""
        SELECT id, fsrs_stability, last_accessed_at, updated_at, category, tier, created_at, is_pinned
        FROM memories
        WHERE active = 1 AND stale = 0
        AND category NOT IN ('preference', 'project')
        AND tier != 'semantic'
        AND is_pinned = 0
    """)

    for row in cur.fetchall():
        # Grace period: skip memories created < 30 days ago (let them prove value)
        try:
            created_dt = datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')).replace(tzinfo=None)
            age_days = (now - created_dt).total_seconds() / 86400
            if age_days < 30:
                continue  # Skip young memories
        except (ValueError, AttributeError):
            pass  # If can't parse created_at, proceed with normal decay

        tier = row["tier"] if row["tier"] else "episodic"
        stability = row["fsrs_stability"] or 1.0
        last_acc = row["last_accessed_at"] or row["updated_at"]
        try:
            last_dt = datetime.fromisoformat(last_acc.replace('Z', '+00:00')).replace(tzinfo=None)
            elapsed = (now - last_dt).total_seconds() / 86400
        except (ValueError, AttributeError):
            elapsed = 90  # Fallback for invalid date format

        # Tier-based decay thresholds
        if tier == 'working':
            # Working memory: decay after 1 day
            if elapsed > 1:
                conn.execute("UPDATE memories SET stale = 1 WHERE id = ?", (row["id"],))
                changes["stale"] += 1
        else:
            # Episodic: use FSRS retention model (existing behavior)
            retention = fsrs_retention(stability, elapsed)
            if retention < 0.5:
                conn.execute("UPDATE memories SET stale = 1 WHERE id = ?", (row["id"],))
                changes["stale"] += 1

    # Also deprioritize memories with very low retention (< 0.3)
    cur2 = conn.execute("""
        SELECT id, fsrs_stability, last_accessed_at, updated_at, priority, category, created_at
        FROM memories
        WHERE active = 1 AND priority > 1
        AND category NOT IN ('preference', 'project')
    """)
    for row in cur2.fetchall():
        # Grace period: skip memories created < 30 days ago
        try:
            created_dt = datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')).replace(tzinfo=None)
            age_days = (now - created_dt).total_seconds() / 86400
            if age_days < 30:
                continue  # Skip young memories
        except (ValueError, AttributeError):
            pass  # If can't parse created_at, proceed with normal decay

        stability = row["fsrs_stability"] or 1.0
        last_acc = row["last_accessed_at"] or row["updated_at"]
        try:
            last_dt = datetime.fromisoformat(last_acc.replace('Z', '+00:00')).replace(tzinfo=None)
            elapsed = (now - last_dt).total_seconds() / 86400
        except (ValueError, AttributeError):
            elapsed = 90  # Fallback for invalid date format

        retention = fsrs_retention(stability, elapsed)
        if retention < 0.3 and row["priority"] > 1:
            conn.execute("UPDATE memories SET priority = MAX(1, priority - 1) WHERE id = ?", (row["id"],))
            changes["deprioritized"] += 1

    conn.commit()
    conn.close()
    logger.info(f"Decay (FSRS): {changes['stale']} stale (R<0.5), {changes['deprioritized']} deprioritized (R<0.3), {changes['expired']} expired")
    return changes




def get_stale() -> List[sqlite3.Row]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM memories WHERE active = 1 AND stale = 1 ORDER BY category, updated_at ASC").fetchall()
    conn.close()
    return rows


# ── Session Snapshots ──



def garbage_collect(days: int = 180) -> None:
    conn = get_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    cur = conn.execute("DELETE FROM memories WHERE active = 0 AND updated_at < ? AND is_pinned = 0", (cutoff,))
    count = cur.rowcount
    conn.execute("""
        DELETE FROM memory_relations
        WHERE source_id NOT IN (SELECT id FROM memories) OR target_id NOT IN (SELECT id FROM memories)
    """)

    # Prune old session snapshots (keep max 30)
    cur2 = conn.execute("""
        DELETE FROM session_snapshots
        WHERE id NOT IN (
            SELECT id FROM session_snapshots ORDER BY created_at DESC LIMIT 30
        )
    """)
    snapshot_count = cur2.rowcount

    # Clean up orphaned vector embeddings
    vec_count = 0
    if has_vec_support():
        try:
            cur3 = conn.execute("""
                DELETE FROM memory_vec
                WHERE rowid NOT IN (SELECT id FROM memories WHERE active = 1)
            """)
            vec_count = cur3.rowcount
        except sqlite3.OperationalError:
            pass  # Vec table doesn't exist

    conn.commit()
    conn.close()
    if vec_count > 0:
        logger.info(f"GC: purged {count} inactive memories older than {days} days, {snapshot_count} old snapshots, {vec_count} orphaned embeddings")
    else:
        logger.info(f"GC: purged {count} inactive memories older than {days} days, {snapshot_count} old snapshots")




def backup_db() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"memories_{timestamp}.db"

    # Use SQLite backup API for consistency
    source = sqlite3.connect(str(DB_PATH))
    dest = sqlite3.connect(str(backup_path))
    source.backup(dest)
    dest.close()
    source.close()

    # Keep only last 7 backups
    backups = sorted(BACKUP_DIR.glob("memories_*.db"), key=lambda p: p.stat().st_mtime)
    for old in backups[:-7]:
        old.unlink()

    size_kb = backup_path.stat().st_size / 1024
    logger.info(f"Backup saved: {backup_path} ({size_kb:.1f} KB)")
    logger.debug(f"Keeping last {min(7, len(backups))} backups")
    return backup_path




def restore_db(backup_file: str) -> bool:
    path = Path(backup_file)
    if not path.exists():
        logger.error(f"Backup not found: {backup_file}")
        return False

    # Verify it's a valid SQLite DB
    try:
        test = sqlite3.connect(str(path))
        test.execute("SELECT COUNT(*) FROM memories")
        test.close()
    except sqlite3.Error as e:
        logger.error(f"Invalid backup file: {e}")
        return False

    # Backup current before restoring
    if DB_PATH.exists():
        emergency = DB_PATH.with_suffix(".db.pre-restore")
        shutil.copy2(str(DB_PATH), str(emergency))
        logger.info(f"Current DB backed up to {emergency}")

    shutil.copy2(str(path), str(DB_PATH))
    logger.info(f"Restored from {backup_file}")
    export_memory_md()
    return True


# ── Phase 3: Graph Intelligence ──



def suggest_next() -> None:
    """Suggest next actions based on current memory state."""
    conn = get_db()
    suggestions = []

    # 1. Expiring soon (within 7 days)
    expiring = conn.execute("""
        SELECT COUNT(*) as c FROM memories
        WHERE active = 1 AND expires_at IS NOT NULL
        AND expires_at > datetime('now') AND expires_at < datetime('now', '+7 days')
    """).fetchone()["c"]
    if expiring:
        suggestions.append(f"⏰ {expiring} memories expiring within 7 days — run: memory-tool list --expired")

    # 2. Stale memories
    stale = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1 AND stale = 1").fetchone()["c"]
    if stale:
        suggestions.append(f"🕸️ {stale} stale memories need review — run: memory-tool stale")

    # 3. Pending items
    pending = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1 AND category = 'pending'").fetchone()["c"]
    if pending:
        suggestions.append(f"📋 {pending} pending items to complete or clean up — run: memory-tool pending")

    # 3.5. Pending corrections
    try:
        pending_corrections = conn.execute("SELECT COUNT(*) as c FROM corrections WHERE status = 'pending'").fetchone()["c"]
        if pending_corrections:
            suggestions.append(f"📝 {pending_corrections} pending corrections — run: memory-tool corrections")
    except sqlite3.OperationalError:
        pass  # corrections table might not exist in older versions

    # 4. Conflicts (close connection first to avoid interference with find_conflicts)
    conn.close()
    find_conflicts, get_snapshots = _get_relations_functions()
    conflicts = find_conflicts()
    if conflicts:
        suggestions.append(f"⚠️ {len(conflicts)} potential duplicate memories — run: memory-tool conflicts")

    # Re-open connection for remaining checks
    conn = get_db()

    # 5. Unembedded memories (vector index gaps)
    if has_vec_support():
        try:
            active = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1").fetchone()["c"]
            embedded = conn.execute("SELECT COUNT(*) as c FROM memory_vec").fetchone()["c"]
            gap = active - embedded
            if gap > 5:
                suggestions.append(f"🔍 {gap} memories not indexed for semantic search — run: memory-tool reindex")
        except sqlite3.OperationalError:
            pass  # Vec table doesn't exist yet

    # 6. Orphan memories (no tags, no project, no relations)
    orphans = conn.execute("""
        SELECT COUNT(*) as c FROM memories m
        WHERE m.active = 1 AND (m.tags = '' OR m.tags IS NULL)
        AND m.project IS NULL
        AND m.id NOT IN (SELECT source_id FROM memory_relations UNION SELECT target_id FROM memory_relations)
    """).fetchone()["c"]
    if orphans:
        suggestions.append(f"🏷️ {orphans} orphan memories (no tags/project/relations) — consider tagging")

    # 7. Running runs that might be stale
    try:
        stale_runs = conn.execute("""
            SELECT COUNT(*) as c FROM runs
            WHERE status = 'running' AND started_at < datetime('now', '-24 hours')
        """).fetchone()["c"]
        if stale_runs:
            suggestions.append(f"🏃 {stale_runs} runs still 'running' for 24h+ — run: memory-tool run list --status running")
    except sqlite3.OperationalError:
        pass  # runs table might not exist in older versions

    # 8. Backup age
    if BACKUP_DIR.exists():
        backups = sorted(BACKUP_DIR.glob("memories_*.db"))
        if backups:
            newest = backups[-1].stat().st_mtime
            days_ago = (datetime.now().timestamp() - newest) / 86400
            if days_ago > 7:
                suggestions.append(f"💾 Last backup was {int(days_ago)} days ago — run: memory-tool backup")
        else:
            suggestions.append("💾 No backups found — run: memory-tool backup")
    else:
        suggestions.append("💾 No backups found — run: memory-tool backup")

    # 9. Unlinked graph entities
    try:
        unlinked = conn.execute("""
            SELECT COUNT(*) as c FROM graph_entities ge
            WHERE ge.id NOT IN (SELECT entity_id FROM memory_entity_links)
        """).fetchone()["c"]
        if unlinked:
            suggestions.append(f"🔗 {unlinked} graph entities not linked to any memories — run: memory-tool graph auto-link")
    except sqlite3.OperationalError:
        pass  # graph tables might not exist

    conn.close()

    if suggestions:
        print("Next actions suggested:\n")  # User-facing output
        for s in suggestions:
            print(f"  {s}")
        print(f"\n({len(suggestions)} suggestions)")
    else:
        print("✅ Everything looks good! No actions needed.")  # User-facing output


# ── CLI ──



def reindex_embeddings() -> None:
    """Bulk-embed all active memories that don't have embeddings yet."""
    if not has_vec_support():
        logger.error("Vector search not available (missing dependencies or model files)")
        return

    conn = get_db()

    # Get all active memories
    all_memories = conn.execute(
        "SELECT id, content FROM memories WHERE active = 1 ORDER BY id"
    ).fetchall()

    if not all_memories:
        logger.info("No active memories to index")
        conn.close()
        return

    # Check which ones already have embeddings
    try:
        existing_ids = set(
            r['rowid'] for r in conn.execute("SELECT rowid FROM memory_vec").fetchall()
        )
    except sqlite3.OperationalError:
        # Vec table doesn't exist yet
        existing_ids = set()

    # Find missing embeddings
    to_embed = [(r['id'], r['content']) for r in all_memories if r['id'] not in existing_ids]

    if not to_embed:
        logger.info(f"All {len(all_memories)} active memories already have embeddings")
        conn.close()
        return

    logger.info(f"Indexing {len(to_embed)} memories (batch size 32)...")

    # Batch process
    BATCH_SIZE = 32
    indexed = 0

    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i:i+BATCH_SIZE]
        batch_ids = [mem_id for mem_id, _ in batch]
        batch_texts = [content for _, content in batch]

        # Generate embeddings
        embeddings = embed_texts_batch(batch_texts)

        # Store embeddings
        for mem_id, embedding in zip(batch_ids, embeddings):
            if embedding is not None:
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO memory_vec(rowid, embedding) VALUES (?, ?)",
                        (mem_id, embedding)
                    )
                    indexed += 1
                except Exception as e:
                    logger.warning(f"Failed to store embedding for #{mem_id}: {e}")

        # Progress update
        if (i + BATCH_SIZE) % 128 == 0:
            logger.debug(f"  Indexed {min(i + BATCH_SIZE, len(to_embed))}/{len(to_embed)}...")

    conn.commit()
    conn.close()
    logger.info(f"Reindex complete: {indexed}/{len(to_embed)} embeddings stored")


# ── Display ──


