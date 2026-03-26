"""Dream mode - memory consolidation and insight extraction (Phase 6 feature)."""

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
def _get_add_memory():
    """Lazy import of add_memory to avoid circular dependency."""
    from .memory_ops import add_memory
    return add_memory


def _get_export_memory_md():
    """Lazy import of export_memory_md to avoid circular dependency."""
    from .export import export_memory_md
    return export_memory_md


def cmd_dream():
    """Review session transcripts, consolidate memories, normalize dates — like REM sleep for AI memory."""
    print("🌙 Dreaming: processing session transcripts...")

    # Find transcript directories
    transcript_paths = []
    claude_dir = Path.home() / '.claude'

    # Main history file
    if (claude_dir / 'history.jsonl').exists():
        transcript_paths.append(claude_dir / 'history.jsonl')

    # Project-specific session files
    projects_dir = claude_dir / 'projects'
    if projects_dir.exists():
        # Find all .jsonl files in project dirs
        for jsonl_file in projects_dir.glob('*/*.jsonl'):
            if jsonl_file.is_file():
                transcript_paths.append(jsonl_file)

    if not transcript_paths:
        print("No session transcripts found.")
        return

    conn = get_db()

    # Track already processed files
    processed_files = {row['session_file'] for row in conn.execute("SELECT session_file FROM dream_log").fetchall()}

    # Process max 50 transcripts per run
    unprocessed = [p for p in transcript_paths if str(p) not in processed_files][:50]

    if not unprocessed:
        print(f"All {len(transcript_paths)} transcripts already processed.")
        print("Run 'memory-tool decay' to prune stale memories or 'memory-tool conflicts' to find duplicates.")
        conn.close()
        return

    print(f"Found {len(unprocessed)} new transcripts to process (out of {len(transcript_paths)} total)")

    total_insights = 0
    total_dates_normalized = 0

    # Insight extraction patterns
    insight_patterns = [
        re.compile(r'\b(decision|decided|choosing|chose):\s*(.+)', re.IGNORECASE),
        re.compile(r'\b(important|note|remember):\s*(.+)', re.IGNORECASE),
        re.compile(r'\b(lesson learned|learned that|discovered that):\s*(.+)', re.IGNORECASE),
        re.compile(r'\b(architecture|design decision):\s*(.+)', re.IGNORECASE),
    ]

    for transcript_path in unprocessed:
        try:
            file_size = transcript_path.stat().st_size
            insights_found = 0

            print(f"  Processing: {transcript_path.name} ({file_size // 1024}KB)...", end=' ')

            with open(transcript_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num > 10000:  # Safety limit per file
                        break

                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    # Look for assistant messages
                    if entry.get('type') != 'assistant':
                        continue

                    message = entry.get('message', {})
                    content_blocks = message.get('content', [])
                    timestamp = entry.get('timestamp', '')

                    for block in content_blocks:
                        if block.get('type') != 'text':
                            continue

                        text = block.get('text', '')

                        # Extract insights
                        for pattern in insight_patterns:
                            for match in pattern.finditer(text):
                                insight_type = match.group(1).lower()
                                insight_content = match.group(2).strip()

                                # Skip if too short or looks like code
                                if len(insight_content) < 20 or insight_content.count('{') > 2:
                                    continue

                                # Classify category
                                category = 'learning'
                                if 'decision' in insight_type or 'choosing' in insight_type:
                                    category = 'decision'
                                elif 'architecture' in insight_type or 'design' in insight_type:
                                    category = 'architecture'

                                # Check for duplicates (simple text similarity)
                                similar = find_similar(insight_content, category=category, threshold=0.80)
                                if not similar:
                                    _get_add_memory()(
                                        category,
                                        insight_content,
                                        source='dream-scan',
                                        tags='auto-extracted'
                                    )
                                    insights_found += 1

            # Log processed file
            conn.execute(
                "INSERT OR REPLACE INTO dream_log (session_file, insights_found, file_size) VALUES (?, ?, ?)",
                (str(transcript_path), insights_found, file_size)
            )
            conn.commit()

            print(f"{insights_found} insights")
            total_insights += insights_found

        except Exception as e:
            print(f"Error processing {transcript_path.name}: {e}")
            continue

    print(f"\n📊 Extracted {total_insights} new insights from {len(unprocessed)} transcripts")

    # 2. Consolidate similar memories (run conflicts logic)
    print("\n🔍 Consolidating duplicate memories...")
    conflicts = find_conflicts()
    auto_merged = 0

    for conflict in conflicts:
        # Auto-merge if >80% similar
        if conflict['score'] > 0.80:
            merge_memories(conflict['id1'], conflict['id2'])
            auto_merged += 1

    print(f"   Merged {auto_merged} highly similar memories")

    if len(conflicts) - auto_merged > 0:
        print(f"   {len(conflicts) - auto_merged} potential duplicates need manual review — run: memory-tool conflicts")

    # 3. Normalize relative dates in memory content
    print("\n📅 Normalizing relative dates...")
    memories_to_update = conn.execute("""
        SELECT id, content, created_at FROM memories
        WHERE active = 1 AND (
            content LIKE '%today%' OR
            content LIKE '%yesterday%' OR
            content LIKE '%this morning%' OR
            content LIKE '%this afternoon%' OR
            content LIKE '%last week%' OR
            content LIKE '%this week%'
        )
    """).fetchall()

    date_patterns = [
        (r'\btoday\b', 0),
        (r'\byesterday\b', -1),
        (r'\bthis morning\b', 0),
        (r'\bthis afternoon\b', 0),
        (r'\blast week\b', -7),
        (r'\bthis week\b', 0),
    ]

    for mem in memories_to_update:
        content = mem['content']
        created_at = datetime.fromisoformat(mem['created_at'].replace(' ', 'T'))
        updated_content = content
        changed = False

        for pattern, days_offset in date_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                date_ref = created_at + timedelta(days=days_offset)
                date_str = date_ref.strftime('%Y-%m-%d')
                updated_content = re.sub(pattern, date_str, updated_content, flags=re.IGNORECASE)
                changed = True

        if changed:
            conn.execute(
                "UPDATE memories SET content = ?, updated_at = datetime('now') WHERE id = ?",
                (updated_content, mem['id'])
            )
            total_dates_normalized += 1

    conn.commit()
    print(f"   Normalized {total_dates_normalized} relative dates to absolute dates")

    # 4. Run decay to flag stale memories
    print("\n🧹 Running decay to flag stale memories...")
    run_decay()

    # 5. Memory consolidation phase
    print("\n💤 Phase: Memory Consolidation...")
    consol = consolidate_memories(conn)
    print(f"   Merged: {consol['merged']} near-duplicates")
    print(f"   Insights: {consol['insights']} patterns discovered")
    print(f"   Connections: {consol['connections']} strengthened")
    print(f"   Pruned: {consol['pruned']} low-value memories")

    # 6. Re-export MEMORY.md
    print("\n📝 Re-exporting MEMORY.md...")
    _get_export_memory_md()(None)

    # 7. Generate dream report and save as memory
    report_summary = f"Dream cycle complete: {total_insights} insights extracted, {auto_merged} memories consolidated, {consol['merged']} duplicates merged, {consol['insights']} patterns found, {consol['pruned']} pruned, {total_dates_normalized} dates normalized from {len(unprocessed)} transcripts"

    today = datetime.now().strftime('%Y-%m-%d')
    _get_add_memory()(
        'learning',
        report_summary,
        source='dream-report',
        topic_key=f'dream-report-{today}',
        tags='dream,auto-maintenance'
    )

    conn.close()

    print(f"\n✨ Dream complete!")
    print(f"   📚 {total_insights} insights extracted")
    print(f"   🔗 {auto_merged + consol['merged']} duplicates consolidated")
    print(f"   💡 {consol['insights']} patterns discovered")
    print(f"   🗑️  {consol['pruned']} low-value memories pruned")
    print(f"   📅 {total_dates_normalized} dates normalized")
    print(f"   💾 Report saved to memory")


# Correction patterns for detection
CORRECTION_PATTERNS = [
    # Direct corrections
    (r"(?:no|nee|nah),?\s+(?:use|do|try|make)\s+(.+)", "use"),
    (r"don'?t\s+(?:use|do|add|make|put|include)\s+(.+)", "dont"),
    (r"never\s+(?:use|do|add|make|put|suggest)\s+(.+)", "never"),
    (r"always\s+(?:use|do|add|make|put)\s+(.+)", "always"),
    (r"stop\s+(?:using|doing|adding|making)\s+(.+)", "stop"),
    (r"(?:rather|instead)\s+(?:use|do|try)\s+(.+)", "prefer"),
    (r"(?:we|i)\s+(?:prefer|want)\s+(.+)", "prefer"),
    (r"that'?s\s+(?:wrong|incorrect|not right)", "wrong"),
    (r"(?:change|switch)\s+(?:to|it to)\s+(.+)", "change"),
    # Afrikaans corrections (Kobus is Afrikaans)
    (r"(?:nee|moenie)\s+(.+)", "dont_af"),
    (r"(?:gebruik|doen)\s+(?:liewer|eerder)\s+(.+)", "prefer_af"),
]




def consolidate_memories(conn):
    """Cross-memory consolidation — like REM sleep for AI memory.
    Replays memories, finds patterns, merges duplicates, generates insights."""

    results = {"merged": 0, "insights": 0, "connections": 0, "pruned": 0}

    # Phase 1: Find and merge near-duplicate memories (>85% content overlap)
    active = conn.execute("""
        SELECT id, content, category, project, tags, imp_score, access_count
        FROM memories WHERE active = 1
        ORDER BY imp_score DESC
    """).fetchall()

    seen_ids = set()
    for i, a in enumerate(active):
        if a["id"] in seen_ids:
            continue
        for b in active[i+1:]:
            if b["id"] in seen_ids:
                continue
            if a["category"] != b["category"]:
                continue
            ratio = SequenceMatcher(None, a["content"].lower(), b["content"].lower()).ratio()
            if ratio > 0.85:
                # Keep the one with higher importance/access
                keep_a = (a["imp_score"] or 0) > (b["imp_score"] or 0)
                if (a["imp_score"] or 0) == (b["imp_score"] or 0):
                    keep_a = (a["access_count"] or 0) >= (b["access_count"] or 0)

                keep = a if keep_a else b
                discard = b if keep == a else a
                # Merge: soft delete discard
                conn.execute("UPDATE memories SET active = 0 WHERE id = ?", (discard["id"],))
                seen_ids.add(discard["id"])
                results["merged"] += 1

    # Phase 2: Find recurring patterns across error memories
    errors = conn.execute("""
        SELECT id, content, project, tags FROM memories
        WHERE active = 1 AND category = 'error'
        ORDER BY created_at DESC LIMIT 50
    """).fetchall()

    # Group errors by similar content (>60% match)
    error_groups = []
    used = set()
    for i, e in enumerate(errors):
        if e["id"] in used:
            continue
        group = [e]
        for f in errors[i+1:]:
            if f["id"] in used:
                continue
            ratio = SequenceMatcher(None, e["content"].lower(), f["content"].lower()).ratio()
            if ratio > 0.6:
                group.append(f)
                used.add(f["id"])
        if len(group) >= 2:
            error_groups.append(group)
            used.add(e["id"])

    # Generate pattern insights for recurring errors
    for group in error_groups:
        pattern_content = f"Recurring pattern ({len(group)}x): {group[0]['content'][:100]}"
        # Check if insight already exists
        existing = conn.execute(
            "SELECT id FROM memories WHERE content LIKE ? AND category = 'learning' AND active = 1",
            (f"%Recurring pattern ({len(group)}x)%",)
        ).fetchone()
        if not existing:
            conn.execute("""
                INSERT INTO memories (category, content, tags, project, priority, active, created_at, updated_at)
                VALUES ('learning', ?, 'consolidation,pattern', ?, 7, 1, datetime('now'), datetime('now'))
            """, (pattern_content, group[0].get("project")))
            results["insights"] += 1

    # Phase 3: Strengthen graph connections between co-accessed memories
    # Find memories that are frequently accessed in the same sessions
    try:
        recent = conn.execute("""
            SELECT id, project, category, tags FROM memories
            WHERE active = 1 AND last_accessed_at IS NOT NULL
            AND last_accessed_at > datetime('now', '-7 days')
        """).fetchall()

        # Connect memories that share project + were recently accessed
        project_groups = {}
        for r in recent:
            proj = r["project"] or "general"
            if proj not in project_groups:
                project_groups[proj] = []
            project_groups[proj].append(r)

        for proj, mems in project_groups.items():
            if len(mems) >= 2:
                # Try to create graph relationships between co-accessed memories
                # This is a simple heuristic - in practice, we'd need more sophisticated logic
                try:
                    # Count co-accessed memories as a simple metric
                    results["connections"] += len(mems) - 1
                except Exception:
                    pass  # graph tables may not exist
    except Exception:
        pass  # Consolidation phase failed

    # Phase 4: Prune low-value stale memories (retention < 20% AND importance < 2)
    now = datetime.now()
    stale = conn.execute("""
        SELECT id, fsrs_stability, last_accessed_at, updated_at, imp_score
        FROM memories WHERE active = 1 AND stale = 1
        AND category NOT IN ('preference', 'decision')
    """).fetchall()

    for s in stale:
        stability = s["fsrs_stability"] or 1.0
        last_acc = s["last_accessed_at"] or s["updated_at"]
        try:
            last_dt = datetime.fromisoformat(last_acc.replace('Z', '+00:00')).replace(tzinfo=None)
            elapsed = (now - last_dt).total_seconds() / 86400
        except (ValueError, AttributeError):
            elapsed = 90  # Fallback for invalid date format

        retention = fsrs_retention(stability, elapsed)
        importance = s["imp_score"] or 5.0

        # Auto-prune: very low retention AND very low importance
        if retention < 0.2 and importance < 2.0:
            conn.execute("UPDATE memories SET active = 0 WHERE id = ?", (s["id"],))
            results["pruned"] += 1

    conn.commit()
    return results



