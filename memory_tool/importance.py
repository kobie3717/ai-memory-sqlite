"""Multi-Channel Importance Scoring System."""

import math
from datetime import datetime


def calc_novelty(created_at, updated_at):
    """Novelty decays over time. New memories score high, old ones low."""
    now = datetime.now()
    try:
        created = datetime.fromisoformat(created_at.replace('Z', '+00:00')).replace(tzinfo=None)
        age_days = (now - created).total_seconds() / 86400
    except (ValueError, AttributeError):
        age_days = 30  # Fallback for invalid date format
    # Exponential decay: starts at 10, halves every 14 days
    return max(0.0, min(10.0, 10.0 * (0.5 ** (age_days / 14))))


def calc_relevance(project, tags, active_projects=None):
    """Relevance based on matching active projects and tags."""
    if active_projects is None:
        active_projects = ["WhatsAuction", "WhatsHub", "Memzy", "WaSP", "FlashVault"]

    score = 3.0  # base
    if project and project in active_projects:
        score += 5.0
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        hot_tags = ["urgent", "critical", "bug", "production", "correction", "user-feedback"]
        for t in tag_list:
            if t in hot_tags:
                score += 1.0
    return max(0.0, min(10.0, score))


def calc_frequency(access_count, fsrs_reps):
    """Frequency based on how often memory is accessed."""
    total = (access_count or 0) + (fsrs_reps or 0)
    if total == 0:
        return 0.0
    # Logarithmic scale: 1 access = 2.0, 5 = 5.0, 20+ = 9.0
    return min(10.0, 2.0 * math.log2(total + 1))


def calc_impact(category, priority):
    """Impact based on category and manual priority."""
    category_weights = {
        "decision": 9.0,
        "error": 8.0,
        "architecture": 8.0,
        "learning": 7.0,
        "workflow": 6.0,
        "preference": 7.0,
        "project": 5.0,
        "pending": 4.0,
        "contact": 3.0,
    }
    base = category_weights.get(category, 5.0)
    # Blend with manual priority (30% manual, 70% category)
    return min(10.0, base * 0.7 + (priority or 5) * 0.3)


def calc_importance(novelty, relevance, frequency, impact):
    """Combined importance score. Weighted average."""
    # Weights: relevance matters most, then impact, frequency, novelty
    weights = {"novelty": 0.15, "relevance": 0.35, "frequency": 0.20, "impact": 0.30}
    score = (
        novelty * weights["novelty"] +
        relevance * weights["relevance"] +
        frequency * weights["frequency"] +
        impact * weights["impact"]
    )
    return round(score, 2)


def update_importance(mem_id, conn):
    """Recalculate and store importance for a single memory."""
    row = conn.execute("""
        SELECT created_at, updated_at, project, tags, access_count,
               fsrs_reps, category, priority
        FROM memories WHERE id = ?
    """, (mem_id,)).fetchone()
    if not row:
        return

    novelty = calc_novelty(row["created_at"], row["updated_at"])
    relevance = calc_relevance(row["project"], row["tags"])
    frequency = calc_frequency(row["access_count"], row["fsrs_reps"])
    impact = calc_impact(row["category"], row["priority"])
    importance = calc_importance(novelty, relevance, frequency, impact)

    conn.execute("""
        UPDATE memories SET
            imp_novelty = ?, imp_relevance = ?, imp_frequency = ?,
            imp_impact = ?, imp_score = ?
        WHERE id = ?
    """, (novelty, relevance, frequency, impact, importance, mem_id))
