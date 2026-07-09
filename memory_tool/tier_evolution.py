"""Auto-Tier Promotion with cooldown for AI-IQ memory system.

Implements behavioral signal accumulation and tier promotion/demotion:
- 3 promotion signals required to jump a tier (not 1)
- 14-day demotion cooldown after promotion
- FSRS stability gates for higher tiers
- Tier 3 hard cap with strict requirements
- Tier 0 auto-expire after 90 days of no access
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from .config import get_logger

logger = get_logger(__name__)


def can_promote(memory: dict) -> Tuple[bool, str]:
    """Check if memory can be promoted to next tier.

    Gates:
    - promotion_signals >= 3
    - fsrs_stability > 5.0 (for Tier 2+)
    - not tier_locked_until (cooldown expired)
    - Tier 3 gate: stability > 10.0 AND access_count >= 5 AND proof_count >= 3

    Args:
        memory: Dictionary with keys: dispatch_priority, promotion_signals,
                fsrs_stability, tier_locked_until, access_count, proof_count

    Returns:
        (can_promote, reason)
    """
    priority = memory.get('dispatch_priority', 1)
    signals = memory.get('promotion_signals', 0)
    stability = memory.get('fsrs_stability', 1.0)
    locked_until = memory.get('tier_locked_until')
    access_count = memory.get('access_count', 0)
    proof_count = memory.get('proof_count', 1)

    # Check if already at max tier
    if priority >= 3:
        return False, "Already at Tier 3 (max)"

    # Check if enough signals accumulated
    if signals < 3:
        return False, f"Only {signals}/3 promotion signals"

    # Check cooldown
    if locked_until:
        try:
            locked_dt = datetime.fromisoformat(locked_until.replace('Z', '+00:00')).replace(tzinfo=None)
            if datetime.now() < locked_dt:
                return False, f"Locked until {locked_until[:10]} (cooldown)"
        except (ValueError, AttributeError):
            pass  # Invalid date, ignore lock

    # Stability gate for Tier 2+
    if priority >= 1 and stability < 5.0:
        return False, f"Stability too low ({stability:.1f} < 5.0)"

    # Tier 3 hard requirements
    if priority == 2:  # Promoting to Tier 3
        if stability < 10.0:
            return False, f"Stability too low for Tier 3 ({stability:.1f} < 10.0)"
        if access_count < 5:
            return False, f"Access count too low for Tier 3 ({access_count} < 5)"
        if proof_count < 3:
            return False, f"Proof count too low for Tier 3 ({proof_count} < 3)"

    return True, "OK"


def can_demote(memory: dict) -> Tuple[bool, str]:
    """Check if memory can be demoted to lower tier.

    Gates:
    - last_promoted_at is None OR > 14 days ago
    - Tier 3 CANNOT be demoted by behavioral signals (only enforce_tier3_cap)

    Args:
        memory: Dictionary with keys: dispatch_priority, last_promoted_at

    Returns:
        (can_demote, reason)
    """
    priority = memory.get('dispatch_priority', 1)
    last_promoted = memory.get('last_promoted_at')

    # Tier 3 is protected from behavioral demotion
    if priority >= 3:
        return False, "Tier 3 cannot be demoted by behavioral signals"

    # Check if already at min tier
    if priority <= 0:
        return False, "Already at Tier 0 (min)"

    # Check cooldown - must be 14+ days since last promotion
    if last_promoted:
        try:
            promoted_dt = datetime.fromisoformat(last_promoted.replace('Z', '+00:00')).replace(tzinfo=None)
            age_days = (datetime.now() - promoted_dt).days
            if age_days < 14:
                return False, f"Promoted {age_days} days ago (cooldown: 14 days)"
        except (ValueError, AttributeError):
            pass  # Invalid date, allow demotion

    return True, "OK"


def record_promotion_signal(conn: sqlite3.Connection, memory_id: int, signal_type: str) -> None:
    """Record a promotion signal and promote if threshold reached.

    Increment promotion_signals by 1. If signals >= 3 AND can_promote():
    - Promote dispatch_priority by 1
    - Reset promotion_signals to 0
    - Set last_promoted_at = now
    - Set tier_locked_until = now + 14 days

    Args:
        conn: Database connection
        memory_id: Memory ID
        signal_type: 'echo' | 'flow_boost' | 'proof_added'
    """
    # Get current state
    row = conn.execute("""
        SELECT dispatch_priority, promotion_signals, fsrs_stability,
               tier_locked_until, access_count, proof_count, last_promoted_at
        FROM memories
        WHERE id = ?
    """, (memory_id,)).fetchone()

    if not row:
        logger.warning(f"Memory {memory_id} not found for promotion signal")
        return

    mem_dict = dict(row)
    current_priority = mem_dict['dispatch_priority'] or 1
    current_signals = mem_dict['promotion_signals'] or 0

    # Increment signal counter
    new_signals = current_signals + 1

    logger.info(f"[Tier+] Memory {memory_id}: promotion signal '{signal_type}' ({new_signals}/3)")

    # Check if we can promote
    if new_signals >= 3:
        mem_dict['promotion_signals'] = new_signals
        can_promo, reason = can_promote(mem_dict)

        if can_promo:
            # Promote!
            new_priority = min(3, current_priority + 1)
            now = datetime.now().isoformat()
            locked_until = (datetime.now() + timedelta(days=14)).isoformat()

            conn.execute("""
                UPDATE memories
                SET dispatch_priority = ?,
                    promotion_signals = 0,
                    last_promoted_at = ?,
                    tier_locked_until = ?
                WHERE id = ?
            """, (new_priority, now, locked_until, memory_id))

            logger.info(f"[Tier+] Memory {memory_id}: PROMOTED {current_priority} → {new_priority} (locked until {locked_until[:10]})")
        else:
            # Can't promote yet, just update counter
            conn.execute("""
                UPDATE memories
                SET promotion_signals = ?
                WHERE id = ?
            """, (new_signals, memory_id))
            logger.info(f"[Tier+] Memory {memory_id}: cannot promote yet ({reason})")
    else:
        # Just update counter
        conn.execute("""
            UPDATE memories
            SET promotion_signals = ?
            WHERE id = ?
        """, (new_signals, memory_id))


def record_demotion_signal(conn: sqlite3.Connection, memory_id: int, signal_type: str) -> None:
    """Record a demotion signal and demote if allowed.

    If can_demote():
    - Demote dispatch_priority by 1 (min 0)
    - Reset promotion_signals to 0
    - Set last_demoted_at = now

    Args:
        conn: Database connection
        memory_id: Memory ID
        signal_type: 'redirect' | 'contradiction' | 'flow_penalize'
    """
    # Get current state
    row = conn.execute("""
        SELECT dispatch_priority, last_promoted_at
        FROM memories
        WHERE id = ?
    """, (memory_id,)).fetchone()

    if not row:
        logger.warning(f"Memory {memory_id} not found for demotion signal")
        return

    mem_dict = dict(row)
    current_priority = mem_dict['dispatch_priority'] or 1

    can_demo, reason = can_demote(mem_dict)

    if can_demo:
        # Demote!
        new_priority = max(0, current_priority - 1)
        now = datetime.now().isoformat()

        conn.execute("""
            UPDATE memories
            SET dispatch_priority = ?,
                promotion_signals = 0,
                last_demoted_at = ?
            WHERE id = ?
        """, (new_priority, now, memory_id))

        logger.info(f"[Tier-] Memory {memory_id}: DEMOTED {current_priority} → {new_priority} (signal: {signal_type})")
    else:
        logger.info(f"[Tier-] Memory {memory_id}: cannot demote ({reason})")


def run_tier_evolution(conn: sqlite3.Connection) -> Dict[str, int]:
    """Batch job for tier evolution rules.

    Runs:
    0. Reclassify stale T1 noise to T0 (write-time classification missed these)
    1. Expire Tier 0: memories with dispatch_priority=0 AND access_count=0
       AND last_accessed_at < 90 days ago → set active=0
    2. Stability gate check: memories with dispatch_priority >= 2 AND fsrs_stability < 5.0
       → demote to 1 (if not locked)
    3. Tier 3 stability gate: dispatch_priority=3 AND (fsrs_stability < 10.0 OR access_count < 5)
       → demote to 2 (if not locked)

    Args:
        conn: Database connection

    Returns:
        {reclassified_to_t0: N, expired: N, stability_demoted: N, tier3_demoted: N}
    """
    now = datetime.now()
    cutoff_90d = (now - timedelta(days=90)).isoformat()

    # Step 0: Reclassify stale T1 noise to T0 (write-time classification missed these)
    # Criteria: proof_count <= 1, access_count = 0, age > 7 days
    # Update last_accessed_at to give 90-day grace period from today
    reclassified_to_t0 = conn.execute("""
        UPDATE memories
        SET dispatch_priority = 0,
            last_accessed_at = datetime('now')
        WHERE active = 1
          AND dispatch_priority = 1
          AND proof_count <= 1
          AND access_count = 0
          AND (julianday('now') - julianday(created_at)) > 7
    """).rowcount

    # Also reclassify pending/error with no access
    reclassified_to_t0 += conn.execute("""
        UPDATE memories
        SET dispatch_priority = 0,
            last_accessed_at = datetime('now')
        WHERE active = 1
          AND dispatch_priority = 1
          AND access_count = 0
          AND category IN ('pending', 'error')
    """).rowcount

    if reclassified_to_t0 > 0:
        logger.info(f"[Tier Evolution] Reclassified {reclassified_to_t0} T1 memories to T0 (noise)")

    # 1. Expire Tier 0 memories with no access in 90 days
    expired = conn.execute("""
        UPDATE memories
        SET active = 0
        WHERE dispatch_priority = 0
          AND access_count = 0
          AND (last_accessed_at IS NULL OR last_accessed_at < ?)
          AND active = 1
    """, (cutoff_90d,))
    expired_count = expired.rowcount

    if expired_count > 0:
        logger.info(f"[Tier Evolution] Expired {expired_count} Tier 0 memories (90+ days no access)")

    # 2. Stability gate: Tier 2+ requires stability >= 5.0
    candidates = conn.execute("""
        SELECT id, dispatch_priority, fsrs_stability, tier_locked_until
        FROM memories
        WHERE dispatch_priority >= 2
          AND fsrs_stability < 5.0
          AND active = 1
    """).fetchall()

    stability_demoted_count = 0
    for row in candidates:
        mem_dict = dict(row)

        # Check if locked
        locked_until = mem_dict.get('tier_locked_until')
        if locked_until:
            try:
                locked_dt = datetime.fromisoformat(locked_until.replace('Z', '+00:00')).replace(tzinfo=None)
                if now < locked_dt:
                    continue  # Still locked
            except (ValueError, AttributeError):
                pass

        # Demote to Tier 1
        conn.execute("""
            UPDATE memories
            SET dispatch_priority = 1,
                last_demoted_at = ?
            WHERE id = ?
        """, (now.isoformat(), mem_dict['id']))

        stability_demoted_count += 1
        logger.info(f"[Tier Evolution] Memory {mem_dict['id']}: stability gate failed ({mem_dict['fsrs_stability']:.1f} < 5.0), demoted to Tier 1")

    # 3. Tier 3 hard gate: stability >= 10.0 AND access_count >= 5
    tier3_candidates = conn.execute("""
        SELECT id, fsrs_stability, access_count, tier_locked_until
        FROM memories
        WHERE dispatch_priority = 3
          AND (fsrs_stability < 10.0 OR access_count < 5)
          AND active = 1
    """).fetchall()

    tier3_demoted_count = 0
    for row in tier3_candidates:
        mem_dict = dict(row)

        # Check if locked
        locked_until = mem_dict.get('tier_locked_until')
        if locked_until:
            try:
                locked_dt = datetime.fromisoformat(locked_until.replace('Z', '+00:00')).replace(tzinfo=None)
                if now < locked_dt:
                    continue  # Still locked
            except (ValueError, AttributeError):
                pass

        # Demote to Tier 2
        conn.execute("""
            UPDATE memories
            SET dispatch_priority = 2,
                last_demoted_at = ?
            WHERE id = ?
        """, (now.isoformat(), mem_dict['id']))

        tier3_demoted_count += 1
        reason = f"stability={mem_dict['fsrs_stability']:.1f}" if mem_dict['fsrs_stability'] < 10.0 else f"access_count={mem_dict['access_count']}"
        logger.info(f"[Tier Evolution] Memory {mem_dict['id']}: Tier 3 gate failed ({reason}), demoted to Tier 2")

    conn.commit()

    return {
        'reclassified_to_t0': reclassified_to_t0,
        'expired': expired_count,
        'stability_demoted': stability_demoted_count,
        'tier3_demoted': tier3_demoted_count
    }
