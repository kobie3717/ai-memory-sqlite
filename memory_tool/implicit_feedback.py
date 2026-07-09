"""Implicit behavioral feedback loop for AI-IQ memory system.

Detects 3 types of feedback signals from conversation logs:
1. Reference Echo - user references surfaced memory content
2. Redirect Detection - user re-asks same question after memory surfaced
3. Flow Continuity - tracks turns-to-resolution after memory surface

All signals update FSRS stability/difficulty automatically. Zero user effort required.
"""

import re
import sqlite3
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from .database import get_db
from .config import get_logger

logger = get_logger(__name__)

# Stop phrases to exclude from n-gram matching
STOP_PHRASES = {
    'this is', 'we should', 'the approach', 'that is', 'it should',
    'you can', 'i think', 'let me', 'here is', 'there is',
    'we need', 'to do', 'we can', 'this will', 'that will'
}

# Contradiction markers
CONTRADICTION_MARKERS = {
    'actually no', 'that\'s wrong', 'no wait', 'not that',
    'i meant', 'forget that', 'that\'s not', 'no that',
    'wait no', 'wrong', 'incorrect', 'mistake'
}


def extract_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract n-grams from text, excluding stop-phrases.

    Args:
        text: Input text
        n: N-gram size (default: 3)

    Returns:
        Set of n-grams (lowercase)
    """
    # Normalize: lowercase, remove punctuation, collapse whitespace
    normalized = re.sub(r'[^\w\s]', ' ', text.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    words = normalized.split()
    if len(words) < n:
        return set()

    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        # Skip if it's a stop phrase
        if ngram not in STOP_PHRASES:
            ngrams.add(ngram)

    return ngrams


def reference_echo_score(user_message: str, memory_content: str) -> float:
    """Calculate reference echo score between user message and memory content.

    Returns 0.0-1.0. Score >0.3 indicates echo detected (3+ word n-gram match).
    Uses both n-gram matching and significant term overlap.

    Args:
        user_message: Current user message
        memory_content: Content from surfaced memory

    Returns:
        Float score 0.0-1.0
    """
    if not user_message or not memory_content:
        return 0.0

    scores = []

    # Approach 1: N-gram matching (strict)
    for n in [3, 4]:
        user_ngrams = extract_ngrams(user_message, n=n)
        memory_ngrams = extract_ngrams(memory_content, n=n)

        if user_ngrams and memory_ngrams:
            intersection = user_ngrams & memory_ngrams
            if intersection:
                # Exact n-gram match is strong signal
                scores.append(0.8)

    # Approach 2: Bigram matching (medium strength)
    user_bigrams = extract_ngrams(user_message, n=2)
    memory_bigrams = extract_ngrams(memory_content, n=2)

    if user_bigrams and memory_bigrams:
        intersection = user_bigrams & memory_bigrams
        if len(intersection) >= 2:  # At least 2 bigrams match
            bigram_score = (len(intersection) / len(user_bigrams)) * 0.6
            scores.append(bigram_score)

    # Approach 3: Significant word overlap (softer signal)
    # Extract words >3 chars, excluding common stop words
    stop_words = {'this', 'that', 'with', 'from', 'have', 'will', 'what', 'when', 'yeah', 'correct', 'setting', 'default'}

    user_words = set(w for w in re.sub(r'[^\w\s]', ' ', user_message.lower()).split() if len(w) > 3 and w not in stop_words)
    memory_words = set(w for w in re.sub(r'[^\w\s]', ' ', memory_content.lower()).split() if len(w) > 3 and w not in stop_words)

    if user_words and memory_words:
        intersection = user_words & memory_words
        # Need at least 2 significant words matching
        if len(intersection) >= 2:
            word_score = (len(intersection) / len(user_words)) * 0.5
            scores.append(word_score)

    return max(scores) if scores else 0.0


def cosine_similarity_simple(text1: str, text2: str) -> float:
    """Word-overlap cosine similarity. No external deps.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Float similarity 0.0-1.0
    """
    if not text1 or not text2:
        return 0.0

    # Normalize and tokenize
    words1 = set(re.sub(r'[^\w\s]', ' ', text1.lower()).split())
    words2 = set(re.sub(r'[^\w\s]', ' ', text2.lower()).split())

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity (simpler than true cosine)
    intersection = words1 & words2
    union = words1 | words2

    if not union:
        return 0.0

    return len(intersection) / len(union)


def detect_contradiction(user_message: str) -> bool:
    """Detect if user message contains contradiction markers.

    Args:
        user_message: User's message

    Returns:
        True if contradiction detected
    """
    message_lower = user_message.lower()

    for marker in CONTRADICTION_MARKERS:
        if marker in message_lower:
            return True

    return False


def apply_echo_feedback(conn: sqlite3.Connection, memory_id: int, score: float, was_cited: bool = False) -> None:
    """Apply positive feedback from reference echo.

    Gated on citation:
    - was_cited=True:  stability * 1.3, reps+1, access_count++, promotion credit
    - was_cited=False: no FSRS mutation, no promotion credit.
                       Caller should add to retrieval_miss (echoed-uncited = retrieval
                       hit, generation miss — counts toward miss actuator).

    Without this gate, echoed-but-uncited memories accrete stability on every turn
    with no raw signal log to replay/repair. Stability gates T3-cap demotions and
    promotion thresholds, so the contamination spreads.

    Args:
        conn: Database connection
        memory_id: Memory ID
        score: Echo score (informational only)
        was_cited: Whether the memory was cited in Claude's response ([mem:N] tag)
    """
    if not was_cited:
        # Echoed but not cited: retrieval hit, generation miss.
        # No FSRS mutation. Caller handles miss actuator.
        logger.debug(f"[Echo] Memory {memory_id}: echoed but not cited — no FSRS boost, no promotion credit (→ retrieval_miss)")
        return

    # Get current FSRS values
    row = conn.execute("""
        SELECT fsrs_stability, fsrs_reps
        FROM memories
        WHERE id = ?
    """, (memory_id,)).fetchone()

    if not row:
        logger.warning(f"Memory {memory_id} not found for echo feedback")
        return

    current_stability = row['fsrs_stability'] or 1.0
    current_reps = row['fsrs_reps'] or 0

    # Apply boost: stability * 1.3, reps+1
    new_stability = current_stability * 1.3
    new_reps = current_reps + 1

    conn.execute("""
        UPDATE memories
        SET fsrs_stability = ?,
            fsrs_reps = ?,
            last_accessed_at = datetime('now'),
            access_count = access_count + 1
        WHERE id = ?
    """, (new_stability, new_reps, memory_id))

    logger.info(f"[Echo +] Memory {memory_id}: stability {current_stability:.2f} → {new_stability:.2f}, reps {current_reps} → {new_reps}")

    # Record promotion signal (tier evolution)
    from .tier_evolution import record_promotion_signal
    record_promotion_signal(conn, memory_id, 'echo')


def apply_redirect_feedback(conn: sqlite3.Connection, memory_id: int) -> None:
    """Apply negative feedback from redirect detection.

    FSRS rating=2 (hard): stability * 0.7, difficulty + 0.5.

    Args:
        conn: Database connection
        memory_id: Memory ID
    """
    # Get current FSRS values
    row = conn.execute("""
        SELECT fsrs_stability, fsrs_difficulty
        FROM memories
        WHERE id = ?
    """, (memory_id,)).fetchone()

    if not row:
        logger.warning(f"Memory {memory_id} not found for redirect feedback")
        return

    current_stability = row['fsrs_stability'] or 1.0
    current_difficulty = row['fsrs_difficulty'] or 5.0

    # Apply penalty: stability * 0.7, difficulty + 0.5
    new_stability = current_stability * 0.7
    new_difficulty = min(10.0, current_difficulty + 0.5)  # Cap at 10

    conn.execute("""
        UPDATE memories
        SET fsrs_stability = ?,
            fsrs_difficulty = ?
        WHERE id = ?
    """, (new_stability, new_difficulty, memory_id))

    logger.info(f"[Redirect -] Memory {memory_id}: stability {current_stability:.2f} → {new_stability:.2f}, difficulty {current_difficulty:.2f} → {new_difficulty:.2f}")

    # Record demotion signal (tier evolution)
    from .tier_evolution import record_demotion_signal
    record_demotion_signal(conn, memory_id, 'redirect')


def apply_stale_flag(conn: sqlite3.Connection, memory_id: int) -> None:
    """Flag memory as stale and demote dispatch_priority.

    Sets stale=1, demotes dispatch_priority by 1 (min 0).

    Args:
        conn: Database connection
        memory_id: Memory ID
    """
    # Get current dispatch_priority
    row = conn.execute("""
        SELECT dispatch_priority
        FROM memories
        WHERE id = ?
    """, (memory_id,)).fetchone()

    if not row:
        logger.warning(f"Memory {memory_id} not found for stale flag")
        return

    current_priority = row['dispatch_priority'] or 1
    new_priority = max(0, current_priority - 1)

    conn.execute("""
        UPDATE memories
        SET stale = 1,
            dispatch_priority = ?
        WHERE id = ?
    """, (new_priority, memory_id))

    logger.info(f"[Stale] Memory {memory_id}: priority {current_priority} → {new_priority}, stale=1")

    # Record demotion signal (tier evolution)
    from .tier_evolution import record_demotion_signal
    record_demotion_signal(conn, memory_id, 'contradiction')


def apply_flow_feedback(conn: sqlite3.Connection, memory_ids: List[int], turns_to_resolution: int) -> None:
    """Apply flow continuity feedback based on turns-to-resolution.

    - ≤3 turns: stability * 1.2, priority +1 (quick resolution)
    - ≥8 turns: stability * 0.8, priority -1 (slow resolution)
    - 3-8 turns: no change (neutral)

    Args:
        conn: Database connection
        memory_ids: List of memory IDs that were surfaced
        turns_to_resolution: Number of conversation turns after surfacing
    """
    if not memory_ids:
        return

    # Determine adjustment based on turns
    if turns_to_resolution <= 3:
        stability_mult = 1.2
        label = "quick resolution"
        signal_type = "flow_boost"
    elif turns_to_resolution >= 8:
        stability_mult = 0.8
        label = "slow resolution"
        signal_type = "flow_penalize"
    else:
        # Neutral - no change
        return

    # Apply to all surfaced memories
    for memory_id in memory_ids:
        row = conn.execute("""
            SELECT fsrs_stability
            FROM memories
            WHERE id = ?
        """, (memory_id,)).fetchone()

        if not row:
            continue

        current_stability = row['fsrs_stability'] or 1.0
        new_stability = current_stability * stability_mult

        conn.execute("""
            UPDATE memories
            SET fsrs_stability = ?
            WHERE id = ?
        """, (new_stability, memory_id))

        logger.info(f"[Flow {label}] Memory {memory_id}: stability {current_stability:.2f} → {new_stability:.2f}")

        # Record promotion/demotion signal (tier evolution)
        from .tier_evolution import record_promotion_signal, record_demotion_signal
        if signal_type == "flow_boost":
            record_promotion_signal(conn, memory_id, signal_type)
        else:
            record_demotion_signal(conn, memory_id, signal_type)


def process_turn_feedback(
    conn: Optional[sqlite3.Connection],
    current_message: str,
    previous_message: str,
    surfaced_memory_ids: List[int],
    turns_since_surface: int,
    cited_memory_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Main entry point for implicit feedback processing.

    Call after each user turn. Runs all 3 detectors and applies appropriate FSRS adjustments.

    Args:
        conn: Database connection (None to create new)
        current_message: Current user message
        previous_message: Previous user message (for redirect detection)
        surfaced_memory_ids: Memory IDs that were surfaced before this turn
        turns_since_surface: Number of turns since memories were surfaced
        cited_memory_ids: Memory IDs that were cited in response (optional, for scoped penalties)

    Returns:
        Dict with detection results:
        {
            'echo': [memory_ids],
            'redirect': [memory_ids],
            'contradiction': [memory_ids],
            'flow': 'none' | 'boost' | 'penalize',
            'retrieval_miss': [memory_ids]
        }
    """
    if conn is None:
        conn = get_db()
        should_close = True
    else:
        should_close = False

    results = {
        'echo': [],
        'redirect': [],
        'contradiction': [],
        'flow': 'none',
        # retrieval_miss subtypes (split for census interpretability):
        #   echo_fp   — echoed (trigram match) but not cited. Primary signal for
        #               measuring echo false-positive rate. Golden set targets this.
        #   unnoticed — surfaced, not echoed, not cited. Plain retrieval noise.
        # retrieval_miss = union of both (backward compat for actuator/callers).
        'retrieval_miss': [],
        'retrieval_miss_echo_fp': [],
        'retrieval_miss_unnoticed': [],
        'retrieval_miss_demoted': []
    }

    try:
        # Compute cited set first
        cited_set = set(cited_memory_ids) if cited_memory_ids else set()

        # Signal 1: Reference Echo
        for mem_id in surfaced_memory_ids:
            # Fetch memory content
            row = conn.execute("""
                SELECT content FROM memories WHERE id = ?
            """, (mem_id,)).fetchone()

            if not row:
                logger.warning(f"Memory {mem_id} not found during echo detection — skipping")
                continue

            memory_content = row['content']
            score = reference_echo_score(current_message, memory_content)

            if score > 0.3:
                # Echo detected!
                was_cited = mem_id in cited_set
                apply_echo_feedback(conn, mem_id, score, was_cited=was_cited)
                results['echo'].append(mem_id)
                if not was_cited:
                    # Echoed but not cited = retrieval hit, generation miss.
                    # apply_echo_feedback already returned early (no FSRS boost).
                    # Subtype echo_fp: primary signal for measuring echo FP rate.
                    results['retrieval_miss'].append(mem_id)
                    results['retrieval_miss_echo_fp'].append(mem_id)

        # Signal 2: Redirect Detection
        # Scope penalty to cited memories if available, else fall back to surfaced
        if previous_message:
            similarity = cosine_similarity_simple(current_message, previous_message)

            if similarity > 0.7:
                # User is re-asking same question
                target_ids = cited_memory_ids if (cited_memory_ids and len(cited_memory_ids) > 0) else surfaced_memory_ids
                for mem_id in target_ids:
                    apply_redirect_feedback(conn, mem_id)
                    results['redirect'].append(mem_id)

        # Signal 2b: Contradiction Detection
        # SCOPED: only stale-flag memories that were echoed in preceding bot turn.
        # Unscoped contradiction ("that's wrong" about route plan) would incorrectly
        # stale-flag architectural memories that happened to be surfaced.
        if detect_contradiction(current_message) and turns_since_surface <= 2:
            # Only flag memories that also triggered an echo signal (were referenced)
            echoed_ids = set(results['echo'])
            for mem_id in surfaced_memory_ids:
                if mem_id in echoed_ids:
                    apply_stale_flag(conn, mem_id)
                    results['contradiction'].append(mem_id)
                else:
                    logger.debug(f"Contradiction detected but memory {mem_id} not echoed — skipping stale flag")

        # Signal 3: Flow Continuity
        # Only apply if this is a resolution turn (heuristic: user says "thanks", "done", "works", etc)
        # Scope penalty to cited memories if available, else fall back to surfaced
        resolution_markers = ['thanks', 'thank you', 'perfect', 'works', 'great', 'done', 'fixed', 'solved']
        is_resolution = any(marker in current_message.lower() for marker in resolution_markers)

        if is_resolution and surfaced_memory_ids:
            target_ids = cited_memory_ids if (cited_memory_ids and len(cited_memory_ids) > 0) else surfaced_memory_ids
            apply_flow_feedback(conn, target_ids, turns_since_surface)
            if turns_since_surface <= 3:
                results['flow'] = 'boost'
            elif turns_since_surface >= 8:
                results['flow'] = 'penalize'

        # Retrieval miss — unnoticed subtype:
        # Surfaced but NOT echoed AND NOT cited. Plain retrieval noise.
        # (echo_fp subtype is written inline in Signal 1 above.)
        if cited_memory_ids is not None:
            echoed_ids = set(results['echo'])
            cited_ids = set(cited_memory_ids)
            for mem_id in surfaced_memory_ids:
                if mem_id not in echoed_ids and mem_id not in cited_ids:
                    results['retrieval_miss'].append(mem_id)
                    results['retrieval_miss_unnoticed'].append(mem_id)
                    logger.debug(f"[RetrievalMiss:unnoticed] Memory {mem_id} surfaced, not echoed, not cited")

        # Signal 4: Retrieval miss actuator
        # Policy: if a memory has been retrieval_miss'd this turn, increment its miss counter.
        # When miss_count >= 5 with citations=0 in last 30 days → demotion signal.
        for mem_id in results['retrieval_miss']:
            # Check this memory's recent citation history
            row = conn.execute("""
                SELECT access_count, created_at,
                       (SELECT COUNT(*) FROM search_log
                        WHERE result_ids LIKE '%' || ? || '%'
                        AND (used_ids IS NULL OR used_ids NOT LIKE '%' || ? || '%')
                        AND created_at > datetime('now', '-30 days')) as miss_count_30d,
                       (SELECT COUNT(*) FROM search_log
                        WHERE used_ids LIKE '%' || ? || '%'
                        AND created_at > datetime('now', '-30 days')) as cite_count_30d
                FROM memories WHERE id = ?
            """, (str(mem_id), str(mem_id), str(mem_id), mem_id)).fetchone()

            if row and row['miss_count_30d'] >= 5 and row['cite_count_30d'] == 0:
                from .tier_evolution import record_demotion_signal
                record_demotion_signal(conn, mem_id, 'retrieval_miss')
                results['retrieval_miss_demoted'].append(mem_id)
                logger.info(f"[Retrieval Miss] Memory {mem_id}: {row['miss_count_30d']} misses, 0 citations in 30d → demotion signal")

        conn.commit()

    except Exception as e:
        logger.error(f"Error in process_turn_feedback: {e}")
        conn.rollback()
    finally:
        if should_close:
            conn.close()

    return results
