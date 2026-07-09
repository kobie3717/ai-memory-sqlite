"""Triple-Gate Retrieval system for AI-IQ memory.

Implements 3-stage retrieval pipeline:
1. Classify at ingestion - add dispatch_priority (0-3) to memories
2. Filter at query time - query intent gates which tiers surface
3. Cap at render - max 3 Tier-3 per query, global Tier-3 cap 200 with FSRS-driven replacement
"""

import re
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .config import get_logger

logger = get_logger(__name__)


def classify_dispatch_priority(memory_dict: Dict[str, Any]) -> int:
    """
    Classify memory into dispatch tier (0-3).

    Tier 3 = cardiac arrest (never cull, always surface when category matches) - target ~10-20%
        - proof_count >= 8 AND access_count >= 10 OR
        - category IN ('architecture','decision') + proof_count >= 8 AND access_count >= 5 OR
        - access_count >= 100 OR
        - 'critical' OR 'security' in tags (OR, not AND) + proof_count >= 5
    Tier 2 = recurring - target ~30-40%
        - proof_count >= 5 AND access_count >= 3 OR
        - access_count >= 20
    Tier 1 = routine (default)
    Tier 0 = noise
        - proof_count == 1 AND access_count == 0 AND created_at < 7 days ago OR
        - category IN ('pending','error') AND access_count == 0

    Args:
        memory_dict: Dictionary with keys: category, proof_count, access_count, tags, created_at

    Returns:
        int: 0-3 dispatch priority tier
    """
    category = memory_dict.get('category', '')
    proof_count = memory_dict.get('proof_count', 1)
    access_count = memory_dict.get('access_count', 0)
    created_at = memory_dict.get('created_at', '')
    tags = memory_dict.get('tags', '')

    # Tier 3: Cardiac arrest - critical memories (strictest criteria - target ~10-20%)
    # Must have both proof AND access (raised thresholds)
    if proof_count >= 8 and access_count >= 10:
        return 3

    # Architecture/decision needs both high proof and access
    if category in ('architecture', 'decision') and proof_count >= 8 and access_count >= 5:
        return 3

    # NOTE: access_count is now citation-count (post 2026-07).
    # Pre-2026-07 entries may have inflated counts from surface-semantics.
    # Guard against auto-captured noise reaching T3 via legacy count inflation.
    # access≥100 is T3 ONLY if not auto-captured noise
    # Auto-captured session logs reach this via surface-count era, not citation merit
    if access_count >= 100:
        is_noise_category = category in ('learning', 'pending', 'error')
        is_auto_captured = tags and 'auto-captured' in tags.lower()
        if not (is_noise_category and is_auto_captured):
            return 3
        # Fall through to T2 check for auto-captured learning entries

    # Critical/security tags with high proof
    # NOTE: OR conjunction is intentional — either tag alone qualifies.
    # Docstring previously said AND; code has always been OR. OR governs.
    if ('critical' in tags.lower() or 'security' in tags.lower()) and proof_count >= 5:
        return 3

    # Tier 2: Recurring - frequently accessed or well-confirmed
    # Must have both proof AND access
    if proof_count >= 5 and access_count >= 3:
        return 2

    # Access-alone path — threshold calibrated for CITATION semantics.
    # Under surface-semantics (pre-2026-07), ≥20 was correct (surfaced ~10x/wk).
    # Under citation-semantics, access_count only increments when Claude cites
    # the memory inline ([mem:N]). Expected rate: 2-5 citations/week per active memory.
    # ≥7 citation-counts ≈ ~1-3 weeks of active use — reasonable T2 signal.
    if access_count >= 7:
        return 2

    # Tier 0: Noise - low value memories
    # Check if old and never accessed
    if proof_count == 1 and access_count == 0 and created_at:
        try:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00')).replace(tzinfo=None)
            age_days = (datetime.now() - created_dt).days
            if age_days > 7:
                return 0
        except (ValueError, AttributeError):
            pass  # Invalid date, skip this check

    # Pending/error categories with no access are noise
    if category in ('pending', 'error') and access_count == 0:
        return 0

    # Default: Tier 1 (routine)
    return 1


def classify_query_urgency(query: str) -> str:
    """
    Classify query urgency to determine which dispatch tiers to surface.

    Returns:
        'urgent' - debugging, errors, critical issues (surface all tiers)
        'architectural' - design decisions, why/how questions (surface 2+3)
        'preference' - user preferences, settings (surface 2+3)
        'routine' - normal queries (surface 1+2+3)
    """
    query_lower = query.lower()

    # Urgent: debugging, errors, failures
    urgent_patterns = [
        r'\b(bug|error|fail|crash|break|wrong|issue)\b',
        r'\b(why.*not|doesn\'t work|not working)\b',
        r'\b(urgent|critical|emergency)\b',
    ]
    for pattern in urgent_patterns:
        if re.search(pattern, query_lower):
            return 'urgent'

    # Architectural: design decisions, rationale
    architectural_patterns = [
        r'\b(why did (we|i)|why (is|are|do))\b',
        r'\b(architecture|design|decision|rationale)\b',
        r'\b(how (does|do|did) (we|i))\b',
        r'\b(pattern|approach|strategy)\b',
    ]
    for pattern in architectural_patterns:
        if re.search(pattern, query_lower):
            return 'architectural'

    # Preference: user settings, preferences
    preference_patterns = [
        r'\b(prefer|like|want|should)\b',
        r'\b(setting|config|preference)\b',
        r'\b(always|never|usually)\b',
    ]
    for pattern in preference_patterns:
        if re.search(pattern, query_lower):
            return 'preference'

    # Default: routine
    return 'routine'


def apply_dispatch_filter(results: List[Any], query_urgency: str) -> List[Any]:
    """
    Filter and boost results based on query urgency and dispatch priority.

    Args:
        results: List of memory rows (must have 'dispatch_priority' field)
        query_urgency: Output from classify_query_urgency()

    Returns:
        Filtered and boosted list of memory rows
    """
    if not results:
        return []

    # Count T3 memories for logging
    count_t3_in_pool = sum(1 for r in results if r.get('dispatch_priority') == 3)
    logger.debug(f"[Gate1] Query urgency classified as: {query_urgency} (T3 memories: {count_t3_in_pool})")

    # Gate 0: Always exclude T0 (noise) from search results.
    # T0 is not search-eligible — it's pending expiry. Gate 1 softening removed
    # the only previous floor, so T0 needs an explicit exclusion here.
    results = [r for r in results if r.get('dispatch_priority', 1) != 0]

    # Gate 1: Soft logging only - no hard exclusion
    # All T1/T2/T3 memories pass through, Gate 3 will cap T3 at 3
    # Old behavior: Gate 1 hard-excluded T3 from routine queries
    # New behavior: Gate 1 logs only, Gate 3 becomes sole T3 gating mechanism

    # Log for audit: which memories would have been excluded under old Gate 1
    if query_urgency == 'routine':
        for row in results:
            try:
                dispatch_priority = row['dispatch_priority'] if 'dispatch_priority' in row.keys() else 1
            except (KeyError, TypeError):
                dispatch_priority = 1

            if dispatch_priority == 3:
                mem_id = row.get('id', 'unknown')
                logger.info(f"[Gate1] Urgency={query_urgency}, query classified as routine — T3 memory {mem_id} allowed through (Gate 3 will cap). Old behavior: would have excluded.")

    # All memories pass through - no filtering at Gate 1
    filtered = results

    # Boost Tier 3 memories to top
    # Sort: Tier 3 first, then original order
    # Create a list of (index, row) tuples to preserve original order
    indexed_rows = list(enumerate(filtered))

    def sort_key(indexed_row):
        idx, row = indexed_row
        try:
            priority = row['dispatch_priority'] if 'dispatch_priority' in row.keys() else 1
        except (KeyError, TypeError):
            priority = 1

        # Tier 3 gets negative index (sorts first), others keep original position
        if priority == 3:
            return (0, idx)  # Tier 3 at top
        else:
            return (1, idx)  # Others keep order

    indexed_rows.sort(key=sort_key)
    filtered = [row for _, row in indexed_rows]

    return filtered


def enforce_tier3_cap(conn: sqlite3.Connection) -> int:
    """
    Enforce global cap of 200 Tier-3 memories.
    If >200 exist, demote lowest-stability ones to Tier 2.

    Args:
        conn: Database connection

    Returns:
        Number of memories demoted
    """
    # Count Tier-3 memories
    count = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE dispatch_priority = 3 AND active = 1"
    ).fetchone()['c']

    if count <= 200:
        return 0

    # Find lowest-stability Tier-3 memories to demote
    excess = count - 200

    # Select candidates: lowest fsrs_stability, excluding pinned
    candidates = conn.execute("""
        SELECT id, fsrs_stability, content
        FROM memories
        WHERE dispatch_priority = 3 AND active = 1 AND is_pinned = 0
        ORDER BY fsrs_stability ASC, access_count ASC
        LIMIT ?
    """, (excess,)).fetchall()

    if not candidates:
        logger.warning(f"Tier-3 cap exceeded ({count}/200) but all are pinned. Cannot demote.")
        return 0

    # Demote to Tier 2
    demoted_ids = [c['id'] for c in candidates]
    placeholders = ','.join('?' * len(demoted_ids))
    conn.execute(
        f"UPDATE memories SET dispatch_priority = 2 WHERE id IN ({placeholders})",
        demoted_ids
    )
    conn.commit()

    logger.info(f"Enforced Tier-3 cap: demoted {len(demoted_ids)} memories from Tier 3 to Tier 2")

    return len(demoted_ids)


def cap_render_results(results: List[Any], max_tier3: int = 3) -> List[Any]:
    """
    Cap number of Tier-3 memories in result list.
    Deduplicates by content_hash.

    Args:
        results: List of memory rows
        max_tier3: Maximum number of Tier-3 memories to include (default: 3)

    Returns:
        Capped list of memory rows
    """
    if not results:
        return []

    # Deduplicate by content_hash first
    seen_hashes = set()
    deduped = []

    for row in results:
        # Handle sqlite3.Row objects
        try:
            content_hash = row['content_hash'] if 'content_hash' in row.keys() else None
        except (KeyError, TypeError):
            content_hash = None

        if content_hash and content_hash in seen_hashes:
            continue  # Skip duplicate

        if content_hash:
            seen_hashes.add(content_hash)

        deduped.append(row)

    # Cap Tier-3 memories
    tier3_count = 0
    capped = []

    for row in deduped:
        # Handle sqlite3.Row objects
        try:
            priority = row['dispatch_priority'] if 'dispatch_priority' in row.keys() else 1
        except (KeyError, TypeError):
            priority = 1

        if priority == 3:
            if tier3_count < max_tier3:
                capped.append(row)
                tier3_count += 1
            # else: skip this Tier-3 memory (cap exceeded)
        else:
            capped.append(row)

    return capped
