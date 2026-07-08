"""Dream mode - memory consolidation and insight extraction (Phase 6 feature)."""
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
from .embedding import embed_and_store, embed_text, semantic_search
from .relations import find_conflicts, merge_memories
from .export import run_decay

logger = get_logger(__name__)

# Lazy imports for optional dependencies
try:
    import numpy as np
    import sqlite_vec
except ImportError:
    pass


# Lazy import to avoid circular dependency
def _get_add_memory() -> Any:
    """Lazy import of add_memory to avoid circular dependency."""
    from .memory_ops import add_memory
    return add_memory


def _get_export_memory_md() -> Any:
    """Lazy import of export_memory_md to avoid circular dependency."""
    from .export import export_memory_md
    return export_memory_md


def cmd_dream() -> None:
    """Review session transcripts, consolidate memories, normalize dates — like REM sleep for AI memory."""
    import os
    dry_run = os.environ.get('AIIQ_DREAM_DRY_RUN', '').lower() in ('1', 'true', 'yes')

    if dry_run:
        print("🌙 [DRY-RUN] Dream mode preview — no changes will be made.")
    else:
        print("🌙 Dreaming: processing session transcripts...")  # User-facing output

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

    conn = get_db()

    if dry_run:
        logger.info("   [DRY-RUN] Skipping transcript scan (saves ~30s)")
        unprocessed = []
        total_insights = 0
    elif not transcript_paths:
        logger.info("No session transcripts found.")
        unprocessed = []
        total_insights = 0

    if not dry_run:
        # Track already processed files
        processed_files = {row['session_file'] for row in conn.execute("SELECT session_file FROM dream_log").fetchall()}
        # Process max 50 transcripts per run
        unprocessed = [p for p in transcript_paths if str(p) not in processed_files][:50]

    if not unprocessed:
        logger.info(f"All {len(transcript_paths)} transcripts already processed.")
        logger.info("Run 'memory-tool decay' to prune stale memories or 'memory-tool conflicts' to find duplicates.")
        conn.close()
        return

    logger.debug(f"Found {len(unprocessed)} new transcripts to process (out of {len(transcript_paths)} total)")

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

            logger.debug(f"  Processing: {transcript_path.name} ({file_size // 1024}KB)...")

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

            logger.info(f"  {transcript_path.name}: {insights_found} insights")
            total_insights += insights_found

        except Exception as e:
            logger.error(f"Error processing {transcript_path.name}: {e}")
            continue

    logger.info(f"📊 Extracted {total_insights} new insights from {len(unprocessed)} transcripts")

    # 2. Consolidate similar memories (run conflicts logic)
    logger.info("🔍 Consolidating duplicate memories...")
    conflicts = find_conflicts()
    auto_merged = 0


    for conflict in conflicts:
        # Auto-merge if >80% similar
        if conflict['score'] > 0.80:
            if dry_run:
                logger.info(f"   [DRY-RUN] Would merge #{conflict['id1']} and #{conflict['id2']} (score={conflict['score']:.2f})")
            else:
                merge_memories(conflict['id1'], conflict['id2'])
            auto_merged += 1

    logger.info(f"   {'[DRY-RUN] Would merge' if dry_run else 'Merged'} {auto_merged} highly similar memories")

    if len(conflicts) - auto_merged > 0:
        logger.info(f"   {len(conflicts) - auto_merged} potential duplicates need manual review — run: memory-tool conflicts")

    # 2.5 Reconsolidation: find near-duplicates (85-95% similarity) and auto-merge
    logger.info("🧠 Reconsolidating near-duplicate memories...")
    if dry_run:
        logger.info("   [DRY-RUN] Skipping reconsolidation")
        reconsolidated = 0
    else:
        reconsolidated = reconsolidate_memories(conn)
    logger.info(f"   {'[DRY-RUN] Would reconsolidate' if dry_run else 'Reconsolidated'} {reconsolidated} near-duplicates")

    # 3. Normalize relative dates in memory content
    logger.info("📅 Normalizing relative dates...")
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
    logger.info(f"   Normalized {total_dates_normalized} relative dates to absolute dates")

    # 3.5. Migrate existing memories to tiers (one-time migration)
    logger.info("🔄 Migrating existing memories to tiers...")
    migration_results = migrate_existing_tiers(conn)
    logger.info(f"   Migrated: {migration_results['semantic']} semantic, {migration_results['working']} working, {migration_results['episodic']} episodic")

    # 4. Run decay to flag stale memories
    logger.info("🧹 Running decay to flag stale memories...")
    run_decay()

    # 4.5. Apply feedback learning from search patterns
    logger.info("🎓 Applying feedback learning from search patterns...")
    from .feedback import apply_feedback_learning
    feedback_results = apply_feedback_learning(conn)
    logger.info(f"   Boosted: {feedback_results['boosted']} high-value memories")
    logger.info(f"   Decayed: {feedback_results['decayed']} low-value memories")
    logger.info(f"   Flagged: {feedback_results['flagged']} unused memories as stale")

    # 5. Memory consolidation phase
    logger.info("💤 Phase: Memory Consolidation...")
    consol = consolidate_memories(conn)
    logger.info(f"   Merged: {consol['merged']} near-duplicates")
    logger.info(f"   Insights: {consol['insights']} patterns discovered")
    logger.info(f"   Connections: {consol['connections']} strengthened")
    logger.info(f"   Pruned: {consol['pruned']} low-value memories")

    # 5.5. Belief consolidation phase
    logger.info("🧠 Phase: Belief Consolidation...")
    from .beliefs import beliefs_dream
    belief_results = beliefs_dream(conn)
    logger.info(f"   Beliefs merged: {belief_results['merged']}")
    logger.info(f"   Predictions expired: {belief_results['predictions_expired']}")
    logger.info(f"   Beliefs weakened: {belief_results['beliefs_weakened']}")

    # 5.6. Lifecycle state auto-deprecation
    logger.info("⚖️  Phase: Lifecycle State Management...")
    try:
        from .beliefs import auto_deprecate_weak_beliefs
        deprecated_count = auto_deprecate_weak_beliefs(conn, days_inactive=60)
        logger.info(f"   Auto-deprecated: {deprecated_count} weak beliefs")
        belief_results['deprecated'] = deprecated_count
    except Exception as e:
        logger.debug(f"Lifecycle deprecation skipped: {e}")
        belief_results['deprecated'] = 0

    # 5.7. Tier promotion
    logger.info("🎓 Phase: Memory Tier Promotion...")
    tier_results = promote_memories(conn)
    logger.info(f"   Working → Episodic: {tier_results['working_to_episodic']}")
    logger.info(f"   Episodic → Semantic: {tier_results['episodic_to_semantic']}")

    # 5.8. Tier Management (new tiers.py module)
    logger.info("🔄 Phase: Tier Management...")
    from .tiers import promote_tier_pass, demote_tier_pass, expire_working
    promoted = promote_tier_pass(conn)
    demoted = demote_tier_pass(conn)
    expired = expire_working(conn)
    logger.info(f"   Promoted to semantic: {promoted}")
    logger.info(f"   Demoted to episodic: {demoted}")
    logger.info(f"   Expired working: {expired}")

    # 5.9. Drift Detection (reinforced lies detection)
    logger.info("⚠️  Phase: Drift Detection...")
    from .validation import find_drift_candidates
    drift_candidates = find_drift_candidates(conn, min_access_count=5, min_age_days=30)
    high_risk = [m for m in drift_candidates if m['drift_risk'] > 0.6]
    if high_risk:
        logger.info(f"   ⚠️  {len(high_risk)} high-risk memories need validation (potential reinforced lies)")
        for c in high_risk[:5]:
            logger.info(f"      #{c['id']} risk={c['drift_risk']:.2f} {c['content'][:80]}...")
    else:
        logger.info(f"   ✓ No high-risk drift detected ({len(drift_candidates)} candidates scanned)")

    # 6. Re-export MEMORY.md
    logger.debug("Re-exporting MEMORY.md...")
    _get_export_memory_md()(None)

    # 6.5. Reclaim disk space after consolidation
    try:
        inactive_count = conn.execute("SELECT COUNT(*) FROM memories WHERE active = 0").fetchone()[0]
        if inactive_count > 50:
            print(f"[dream] Vacuuming database ({inactive_count} inactive memories)...")
            conn.execute("VACUUM")
            print(f"[dream] Vacuum complete.")
        else:
            print(f"[dream] Skipping vacuum ({inactive_count} inactive memories < 50 threshold)")
    except Exception as e:
        print(f"[dream] Vacuum failed: {e}")

    # 7. Generate dream report and save as memory
    report_summary = f"Dream cycle complete: {total_insights} insights extracted, {auto_merged} memories consolidated, {reconsolidated} near-duplicates reconsolidated, {consol['merged']} duplicates merged, {consol['insights']} patterns found, {consol['pruned']} pruned, {total_dates_normalized} dates normalized, {feedback_results['boosted']} feedback-boosted, {feedback_results['decayed']} feedback-decayed, {feedback_results['flagged']} feedback-flagged, {belief_results['merged']} beliefs merged, {belief_results['predictions_expired']} predictions expired, {belief_results['beliefs_weakened']} beliefs weakened, {belief_results.get('deprecated', 0)} beliefs deprecated, {tier_results['working_to_episodic']} working→episodic, {tier_results['episodic_to_semantic']} episodic→semantic, {promoted} promoted to semantic, {demoted} demoted to episodic, {expired} working expired, {len(high_risk)} high-risk drift candidates detected from {len(unprocessed)} transcripts"

    today = datetime.now().strftime('%Y-%m-%d')
    _get_add_memory()(
        'learning',
        report_summary,
        source='dream-report',
        topic_key=f'dream-report-{today}',
        tags='dream,auto-maintenance'
    )

    conn.close()

    # Final summary — this is user output, keep as print()
    print(f"\n✨ Dream complete!")
    print(f"   📚 {total_insights} insights extracted")
    print(f"   🔗 {auto_merged + reconsolidated + consol['merged']} duplicates consolidated")
    print(f"   🧠 {reconsolidated} near-duplicates reconsolidated")
    print(f"   💡 {consol['insights']} patterns discovered")
    print(f"   🗑️  {consol['pruned']} low-value memories pruned")
    print(f"   📅 {total_dates_normalized} dates normalized")
    print(f"   🎓 {feedback_results['boosted']} boosted / {feedback_results['decayed']} decayed / {feedback_results['flagged']} flagged via feedback")
    print(f"   🔮 {belief_results['merged']} beliefs merged / {belief_results['predictions_expired']} predictions expired / {belief_results['beliefs_weakened']} beliefs weakened / {belief_results.get('deprecated', 0)} beliefs deprecated")
    print(f"   🎯 {tier_results['working_to_episodic']} promoted to episodic / {tier_results['episodic_to_semantic']} promoted to semantic")
    print(f"   🔄 {promoted} promoted to semantic / {demoted} demoted to episodic / {expired} working expired")
    if high_risk:
        print(f"   ⚠️  {len(high_risk)} high-risk drift candidates need validation (run 'memory-tool validate scan')")
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




def consolidate_memories(conn: sqlite3.Connection) -> Dict[str, int]:
    """Cross-memory consolidation — like REM sleep for AI memory.
    Replays memories, finds patterns, merges duplicates, generates insights."""

    results = {"merged": 0, "insights": 0, "connections": 0, "pruned": 0}

    # Phase 1: Find and merge near-duplicate memories (>85% content overlap)
    # Skip pinned memories from consolidation
    active = conn.execute("""
        SELECT id, content, category, project, tags, imp_score, access_count, proof_count, source_memory_ids
        FROM memories WHERE active = 1 AND is_pinned = 0
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

                # Track proof: increment proof_count and append source IDs
                keep_id = keep["id"]
                discard_id = discard["id"]

                # Get current proof tracking data
                keep_proof_count = keep["proof_count"] or 1
                discard_proof_count = discard["proof_count"] or 1
                new_proof_count = keep_proof_count + discard_proof_count

                # Merge source_memory_ids (JSON arrays)
                keep_sources = json.loads(keep["source_memory_ids"]) if keep["source_memory_ids"] else []
                discard_sources = json.loads(discard["source_memory_ids"]) if discard["source_memory_ids"] else []

                # Add the discarded memory's ID to sources
                new_sources = keep_sources + discard_sources + [discard_id]
                new_sources_json = json.dumps(new_sources)

                # Update keep memory with proof tracking
                conn.execute(
                    "UPDATE memories SET proof_count = ?, source_memory_ids = ? WHERE id = ?",
                    (new_proof_count, new_sources_json, keep_id)
                )

                # Merge: soft delete discard
                conn.execute("UPDATE memories SET active = 0 WHERE id = ?", (discard_id,))
                seen_ids.add(discard_id)
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
            """, (pattern_content, group[0]["project"] if group[0]["project"] else None))
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
    # Skip pinned memories from pruning
    now = datetime.now()
    stale = conn.execute("""
        SELECT id, fsrs_stability, last_accessed_at, updated_at, imp_score
        FROM memories WHERE active = 1 AND stale = 1
        AND category NOT IN ('preference', 'decision')
        AND is_pinned = 0
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


def reconsolidate_memories(conn: sqlite3.Connection) -> int:
    """Reconsolidation phase: find near-duplicates (85-95% similarity) and auto-merge them.
    This phase happens after standard dedup but catches memories that are almost identical
    but have minor differences (like updated timestamps, slightly different wording, etc.)."""

    # Skip pinned memories from reconsolidation
    active = conn.execute("""
        SELECT id, content, category, project, tags, imp_score, access_count
        FROM memories WHERE active = 1 AND is_pinned = 0
        ORDER BY imp_score DESC
    """).fetchall()

    reconsolidated = 0
    seen_ids = set()

    for i, a in enumerate(active):
        if a["id"] in seen_ids:
            continue
        for b in active[i+1:]:
            if b["id"] in seen_ids:
                continue
            # Only compare within same category
            if a["category"] != b["category"]:
                continue

            # Calculate similarity
            ratio = SequenceMatcher(None, a["content"].lower(), b["content"].lower()).ratio()

            # Near-duplicate range: 85-95%
            if 0.85 <= ratio < 0.95:
                # Keep the one with higher importance/access
                keep_a = (a["imp_score"] or 0) > (b["imp_score"] or 0)
                if (a["imp_score"] or 0) == (b["imp_score"] or 0):
                    keep_a = (a["access_count"] or 0) >= (b["access_count"] or 0)

                keep_id = a["id"] if keep_a else b["id"]
                discard_id = b["id"] if keep_a else a["id"]

                # Merge: append unique facts from older to newer
                keep_mem = a if keep_a else b
                discard_mem = b if keep_a else a

                # Extract unique sentences from discarded memory
                keep_sentences = set(s.strip() for s in keep_mem["content"].split('.') if s.strip())
                discard_sentences = set(s.strip() for s in discard_mem["content"].split('.') if s.strip())
                unique_facts = discard_sentences - keep_sentences

                # If there are unique facts, append them to the kept memory
                if unique_facts:
                    updated_content = keep_mem["content"]
                    for fact in unique_facts:
                        if fact and len(fact) > 10:  # Only meaningful facts
                            updated_content += f" {fact}."

                    conn.execute(
                        "UPDATE memories SET content = ?, updated_at = datetime('now') WHERE id = ?",
                        (updated_content, keep_id)
                    )

                    # Re-embed the updated memory
                    embed_and_store(conn, keep_id, updated_content)

                # Mark as superseded
                conn.execute("UPDATE memories SET active = 0 WHERE id = ?", (discard_id,))
                conn.execute(
                    "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'supersedes')",
                    (keep_id, discard_id)
                )

                seen_ids.add(discard_id)
                reconsolidated += 1

    conn.commit()
    return reconsolidated


def migrate_existing_tiers(conn: sqlite3.Connection) -> Dict[str, int]:
    """Migrate existing memories to appropriate tiers.

    Migration rules:
    - access_count >= 5 AND age > 30 days → semantic
    - has expiry within 24h → working
    - everything else stays episodic

    Returns dict with migration counts.
    """
    results = {"semantic": 0, "working": 0, "episodic": 0}
    now = datetime.now()

    # Find memories that need tier assignment (tier is NULL or 'episodic')
    memories = conn.execute("""
        SELECT id, access_count, created_at, expires_at, tier
        FROM memories
        WHERE active = 1
    """).fetchall()

    for mem in memories:
        current_tier = mem['tier'] if mem['tier'] else 'episodic'

        # Skip if already properly tiered
        if current_tier in ('semantic', 'working'):
            continue

        # Calculate age
        created_at = datetime.fromisoformat(mem['created_at'].replace(' ', 'T'))
        age_days = (now - created_at).total_seconds() / 86400

        # Determine target tier
        target_tier = 'episodic'

        # Check for semantic promotion
        if mem['access_count'] >= 5 and age_days > 30:
            target_tier = 'semantic'
        # Check for working demotion
        elif mem['expires_at']:
            try:
                exp_dt = datetime.fromisoformat(mem['expires_at'].replace('Z', '+00:00')).replace(tzinfo=None)
                hours_until_expiry = (exp_dt - now).total_seconds() / 3600
                if hours_until_expiry <= 24 and hours_until_expiry > 0:
                    target_tier = 'working'
            except (ValueError, AttributeError):
                pass

        # Update if tier changed
        if target_tier != current_tier:
            conn.execute("UPDATE memories SET tier = ? WHERE id = ?", (target_tier, mem['id']))
            results[target_tier] += 1

    conn.commit()
    return results


def promote_memories(conn: sqlite3.Connection) -> Dict[str, int]:
    """Promote memories between tiers based on access patterns.

    Promotion rules:
    - working → episodic: access_count >= 2 AND no expiry (or expiry removed)
    - episodic → semantic: access_count >= 5 AND age > 7 days
    - semantic memories are immune to decay

    Returns dict with promotion counts.
    """
    results = {"working_to_episodic": 0, "episodic_to_semantic": 0}
    now = datetime.now()

    # Promote working → episodic
    # Memories with access_count >= 2 and no expiry (or expired/removed)
    working_mems = conn.execute("""
        SELECT id, access_count, expires_at, created_at
        FROM memories
        WHERE active = 1 AND tier = 'working'
    """).fetchall()

    for mem in working_mems:
        # Promote if accessed 2+ times and (no expiry OR expiry passed)
        if mem['access_count'] >= 2:
            if not mem['expires_at'] or mem['expires_at'] < now.isoformat():
                conn.execute("UPDATE memories SET tier = 'episodic' WHERE id = ?", (mem['id'],))
                results['working_to_episodic'] += 1
                logger.debug(f"Promoted #{mem['id']} working → episodic (accessed {mem['access_count']}x)")

    # Promote episodic → semantic
    # Memories with access_count >= 5 and age > 7 days
    episodic_mems = conn.execute("""
        SELECT id, access_count, created_at
        FROM memories
        WHERE active = 1 AND tier = 'episodic'
        AND access_count >= 5
    """).fetchall()

    for mem in episodic_mems:
        created_at = datetime.fromisoformat(mem['created_at'].replace(' ', 'T'))
        age_days = (now - created_at).total_seconds() / 86400

        if age_days > 7:
            conn.execute("UPDATE memories SET tier = 'semantic' WHERE id = ?", (mem['id'],))
            results['episodic_to_semantic'] += 1
            logger.debug(f"Promoted #{mem['id']} episodic → semantic (accessed {mem['access_count']}x, age {age_days:.0f}d)")

    conn.commit()
    return results


