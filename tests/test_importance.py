"""Tests for importance scoring system."""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_tool import importance


def test_calc_novelty_new_memory():
    """Test novelty score for brand new memory."""
    now = datetime.now()
    created = now.isoformat()
    updated = now.isoformat()

    novelty = importance.calc_novelty(created, updated)

    # New memory should have high novelty (~10.0)
    assert 9.0 <= novelty <= 10.0


def test_calc_novelty_old_memory():
    """Test novelty score for old memory."""
    now = datetime.now()
    old = (now - timedelta(days=90)).isoformat()

    novelty = importance.calc_novelty(old, old)

    # Old memory should have low novelty
    assert novelty < 5.0


def test_calc_novelty_decreases_over_time():
    """Test that novelty decreases as memory ages."""
    now = datetime.now()

    n1 = importance.calc_novelty(now.isoformat(), now.isoformat())
    n2 = importance.calc_novelty((now - timedelta(days=7)).isoformat(), (now - timedelta(days=7)).isoformat())
    n3 = importance.calc_novelty((now - timedelta(days=30)).isoformat(), (now - timedelta(days=30)).isoformat())
    n4 = importance.calc_novelty((now - timedelta(days=60)).isoformat(), (now - timedelta(days=60)).isoformat())

    # Should decrease over time
    assert n1 > n2 > n3 > n4


def test_calc_novelty_bounded():
    """Test that novelty stays within 0-10 range."""
    now = datetime.now()

    # Brand new
    n_new = importance.calc_novelty(now.isoformat(), now.isoformat())
    assert 0.0 <= n_new <= 10.0

    # Very old
    n_old = importance.calc_novelty((now - timedelta(days=365)).isoformat(), (now - timedelta(days=365)).isoformat())
    assert 0.0 <= n_old <= 10.0


def test_calc_novelty_invalid_date():
    """Test novelty calculation with invalid date format."""
    novelty = importance.calc_novelty("invalid-date", "invalid-date")

    # Should use fallback (30 days)
    assert 0.0 <= novelty <= 10.0


def test_calc_relevance_active_project():
    """Test relevance score for active project."""
    score = importance.calc_relevance(
        project="WhatsAuction",
        tags="",
        active_projects=["WhatsAuction", "FlashVault"]
    )

    # Should get bonus for active project
    assert score >= 8.0  # base 3.0 + 5.0 for active project


def test_calc_relevance_inactive_project():
    """Test relevance score for inactive project."""
    score = importance.calc_relevance(
        project="OldProject",
        tags="",
        active_projects=["WhatsAuction", "FlashVault"]
    )

    # Should get base score only
    assert score == 3.0


def test_calc_relevance_hot_tags():
    """Test relevance boost from hot tags."""
    score = importance.calc_relevance(
        project=None,
        tags="urgent,critical,bug",
        active_projects=["WhatsAuction"]
    )

    # Should get bonus for hot tags (3.0 base + 3.0 for 3 hot tags)
    assert score >= 6.0


def test_calc_relevance_bounded():
    """Test that relevance stays within 0-10 range."""
    # Maximum case
    score_max = importance.calc_relevance(
        project="WhatsAuction",
        tags="urgent,critical,bug,production",
        active_projects=["WhatsAuction"]
    )
    assert 0.0 <= score_max <= 10.0

    # Minimum case
    score_min = importance.calc_relevance(
        project=None,
        tags="",
        active_projects=[]
    )
    assert 0.0 <= score_min <= 10.0


def test_calc_frequency_no_access():
    """Test frequency score for never-accessed memory."""
    freq = importance.calc_frequency(access_count=0, fsrs_reps=0)
    assert freq == 0.0


def test_calc_frequency_single_access():
    """Test frequency score for single access."""
    freq = importance.calc_frequency(access_count=1, fsrs_reps=0)
    assert freq > 0.0
    assert freq < 5.0


def test_calc_frequency_many_accesses():
    """Test frequency score for heavily accessed memory."""
    freq = importance.calc_frequency(access_count=20, fsrs_reps=10)
    assert freq > 5.0


def test_calc_frequency_increases_logarithmically():
    """Test that frequency increases but with diminishing returns."""
    f1 = importance.calc_frequency(access_count=1, fsrs_reps=0)
    f5 = importance.calc_frequency(access_count=5, fsrs_reps=0)
    f20 = importance.calc_frequency(access_count=20, fsrs_reps=0)
    f100 = importance.calc_frequency(access_count=100, fsrs_reps=0)

    # Should increase but not linearly
    assert f5 > f1
    assert f20 > f5
    assert f100 > f20

    # Logarithmic growth means rate of increase slows down
    # The increase from f1->f5 might be less than f5->f20 due to log2(x+1)
    # But f20->f100 increase should be less than f5->f20 (diminishing returns)
    assert (f100 - f20) < (f20 - f5)


def test_calc_frequency_bounded():
    """Test that frequency stays within 0-10 range."""
    freq = importance.calc_frequency(access_count=1000, fsrs_reps=1000)
    assert 0.0 <= freq <= 10.0


def test_calc_impact_by_category():
    """Test impact scores for different categories."""
    decision = importance.calc_impact(category="decision", priority=5)
    error = importance.calc_impact(category="error", priority=5)
    learning = importance.calc_impact(category="learning", priority=5)
    pending = importance.calc_impact(category="pending", priority=5)

    # Decision should be highest
    assert decision > error
    assert error >= learning
    assert learning > pending


def test_calc_impact_high_priority():
    """Test impact with high manual priority."""
    impact_low = importance.calc_impact(category="project", priority=1)
    impact_high = importance.calc_impact(category="project", priority=10)

    # Higher priority should increase impact
    assert impact_high > impact_low


def test_calc_impact_priority_blend():
    """Test that impact blends category and priority (70/30)."""
    # Category weight for "decision" is 9.0
    # With priority=5, should be: 9.0*0.7 + 5*0.3 = 6.3 + 1.5 = 7.8
    impact = importance.calc_impact(category="decision", priority=5)
    assert 7.5 <= impact <= 8.0


def test_calc_impact_bounded():
    """Test that impact stays within 0-10 range."""
    impact = importance.calc_impact(category="decision", priority=10)
    assert 0.0 <= impact <= 10.0


def test_calc_importance_weights():
    """Test that importance calculation uses correct weights."""
    # Weights: novelty=0.15, relevance=0.35, frequency=0.20, impact=0.30
    # All components at 10.0 should give 10.0
    score = importance.calc_importance(
        novelty=10.0,
        relevance=10.0,
        frequency=10.0,
        impact=10.0
    )
    assert score == 10.0

    # All at 0.0 should give 0.0
    score = importance.calc_importance(
        novelty=0.0,
        relevance=0.0,
        frequency=0.0,
        impact=0.0
    )
    assert score == 0.0


def test_calc_importance_relevance_weighted_most():
    """Test that relevance has highest weight."""
    # Only relevance high
    score_rel = importance.calc_importance(
        novelty=0.0, relevance=10.0, frequency=0.0, impact=0.0
    )

    # Only impact high
    score_imp = importance.calc_importance(
        novelty=0.0, relevance=0.0, frequency=0.0, impact=10.0
    )

    # Relevance (0.35) should contribute more than impact (0.30)
    assert score_rel > score_imp


def test_calc_importance_rounded():
    """Test that importance is rounded to 2 decimal places."""
    score = importance.calc_importance(
        novelty=7.333,
        relevance=8.666,
        frequency=5.111,
        impact=6.999
    )

    # Should be rounded
    str_score = str(score)
    decimal_places = len(str_score.split('.')[-1]) if '.' in str_score else 0
    assert decimal_places <= 2


def test_update_importance(temp_db, sample_memory):
    """Test updating importance scores in database."""
    from memory_tool import database

    conn = database.get_db()

    # Update importance
    importance.update_importance(sample_memory, conn)
    conn.commit()

    # Verify scores were calculated and stored
    row = conn.execute("""
        SELECT imp_novelty, imp_relevance, imp_frequency, imp_impact, imp_score
        FROM memories WHERE id = ?
    """, (sample_memory,)).fetchone()

    assert row["imp_novelty"] is not None
    assert row["imp_relevance"] is not None
    assert row["imp_frequency"] is not None
    assert row["imp_impact"] is not None
    assert row["imp_score"] is not None

    # All should be within valid range
    assert 0.0 <= row["imp_novelty"] <= 10.0
    assert 0.0 <= row["imp_relevance"] <= 10.0
    assert 0.0 <= row["imp_frequency"] <= 10.0
    assert 0.0 <= row["imp_impact"] <= 10.0
    assert 0.0 <= row["imp_score"] <= 10.0

    conn.close()


def test_update_importance_nonexistent(temp_db):
    """Test updating importance for non-existent memory."""
    from memory_tool import database

    conn = database.get_db()

    # Should handle gracefully
    importance.update_importance(99999, conn)

    # Should not crash
    conn.close()
