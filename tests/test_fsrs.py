"""Tests for FSRS-6 spaced repetition functions."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_tool import fsrs


def test_retention_at_zero():
    """Test that retention at t=0 is 1.0 (100%)."""
    retention = fsrs.fsrs_retention(stability=10.0, days_elapsed=0)
    assert retention == 1.0


def test_retention_decreases_over_time():
    """Test that retention decreases as time passes."""
    stability = 10.0

    r0 = fsrs.fsrs_retention(stability, 0)
    r1 = fsrs.fsrs_retention(stability, 5)
    r2 = fsrs.fsrs_retention(stability, 10)
    r3 = fsrs.fsrs_retention(stability, 20)

    # Retention should decrease over time
    assert r0 > r1 > r2 > r3
    assert 0.0 <= r3 <= 1.0


def test_retention_with_different_stabilities():
    """Test that higher stability means slower forgetting."""
    days = 10

    r_low = fsrs.fsrs_retention(stability=5.0, days_elapsed=days)
    r_high = fsrs.fsrs_retention(stability=20.0, days_elapsed=days)

    # Higher stability should have higher retention at same time point
    assert r_high > r_low


def test_retention_with_zero_stability():
    """Test retention with zero or negative stability."""
    retention = fsrs.fsrs_retention(stability=0.0, days_elapsed=10)
    assert retention == 0.0

    retention = fsrs.fsrs_retention(stability=-1.0, days_elapsed=10)
    assert retention == 0.0


def test_new_stability_increases_on_success():
    """Test that stability increases on successful reviews."""
    old_s = 5.0
    old_d = 5.0
    elapsed = 2.0

    # Test different ratings
    s_forgotten = fsrs.fsrs_new_stability(old_s, old_d, rating=1, elapsed_days=elapsed)
    s_hard = fsrs.fsrs_new_stability(old_s, old_d, rating=2, elapsed_days=elapsed)
    s_good = fsrs.fsrs_new_stability(old_s, old_d, rating=3, elapsed_days=elapsed)
    s_easy = fsrs.fsrs_new_stability(old_s, old_d, rating=4, elapsed_days=elapsed)

    # Forgotten should decrease
    assert s_forgotten < old_s

    # Success ratings should increase stability
    assert s_hard > old_s
    assert s_good > s_hard
    assert s_easy > s_good


def test_new_stability_forgotten_rating():
    """Test stability drops significantly on forgotten rating."""
    old_s = 10.0
    new_s = fsrs.fsrs_new_stability(old_s, old_d=5.0, rating=1, elapsed_days=5.0)

    # Should drop to ~20% of original
    assert new_s < old_s * 0.25
    assert new_s >= 0.1  # Should have a minimum floor


def test_new_stability_capped_at_max():
    """Test that stability is capped at maximum value."""
    old_s = 300.0  # Already high
    new_s = fsrs.fsrs_new_stability(old_s, old_d=3.0, rating=4, elapsed_days=1.0)

    # Should be capped at 365 days
    assert new_s <= 365.0


def test_new_difficulty_adjusts_correctly():
    """Test difficulty adjustment based on ratings."""
    old_d = 5.0

    d_forgotten = fsrs.fsrs_new_difficulty(old_d, rating=1)
    d_hard = fsrs.fsrs_new_difficulty(old_d, rating=2)
    d_good = fsrs.fsrs_new_difficulty(old_d, rating=3)
    d_easy = fsrs.fsrs_new_difficulty(old_d, rating=4)

    # Forgotten and hard should increase difficulty
    assert d_forgotten > old_d
    assert d_hard > old_d

    # Good and easy should decrease difficulty
    assert d_good < old_d
    assert d_easy < d_good


def test_new_difficulty_bounded():
    """Test that difficulty stays within bounds."""
    # Test lower bound
    d = fsrs.fsrs_new_difficulty(old_d=1.0, rating=4)
    assert d >= 0.1

    # Test upper bound
    d = fsrs.fsrs_new_difficulty(old_d=9.0, rating=1)
    assert d <= 10.0


def test_new_difficulty_mean_reversion():
    """Test that difficulty slowly reverts to mean (5.0)."""
    # Very low difficulty
    d_low = fsrs.fsrs_new_difficulty(old_d=2.0, rating=3)
    # Should move slightly toward 5.0
    assert d_low > 2.0

    # Very high difficulty
    d_high = fsrs.fsrs_new_difficulty(old_d=9.0, rating=3)
    # Should move slightly toward 5.0
    assert d_high < 9.0


def test_next_interval_calculation():
    """Test interval calculation based on stability."""
    stability = 10.0

    # Default retention (0.9)
    interval = fsrs.fsrs_next_interval(stability)
    assert interval > 0

    # Lower desired retention = longer interval
    interval_80 = fsrs.fsrs_next_interval(stability, desired_retention=0.8)
    interval_90 = fsrs.fsrs_next_interval(stability, desired_retention=0.9)
    assert interval_80 > interval_90


def test_next_interval_with_zero_stability():
    """Test interval calculation with zero stability."""
    interval = fsrs.fsrs_next_interval(stability=0.0)
    assert interval > 0  # Should have minimum value
    assert interval < 1.0


def test_next_interval_scales_with_stability():
    """Test that interval scales proportionally with stability."""
    interval_1 = fsrs.fsrs_next_interval(stability=5.0)
    interval_2 = fsrs.fsrs_next_interval(stability=10.0)

    # Double stability should roughly double interval
    assert interval_2 > interval_1
    assert 1.5 < (interval_2 / interval_1) < 2.5


def test_auto_rating_first_access():
    """Test auto-rating for first access."""
    rating = fsrs.fsrs_auto_rating(category="learning", access_count=0, priority=5)
    assert rating == 3  # Should return "good" for first access


def test_auto_rating_high_priority():
    """Test auto-rating for high priority memories."""
    rating = fsrs.fsrs_auto_rating(category="decision", access_count=5, priority=9)
    assert rating == 4  # High priority = easy


def test_auto_rating_medium_priority():
    """Test auto-rating for medium priority memories."""
    rating = fsrs.fsrs_auto_rating(category="learning", access_count=3, priority=6)
    assert rating == 3  # Medium priority = good


def test_auto_rating_low_priority():
    """Test auto-rating for low priority memories."""
    rating = fsrs.fsrs_auto_rating(category="project", access_count=2, priority=2)
    assert rating == 2  # Low priority = hard


def test_fsrs_workflow_simulation():
    """Test a complete FSRS workflow simulation."""
    # Start with new memory
    stability = 1.0
    difficulty = 5.0

    # First review after 1 day - good rating
    elapsed = 1.0
    rating = 3
    stability = fsrs.fsrs_new_stability(stability, difficulty, rating, elapsed)
    difficulty = fsrs.fsrs_new_difficulty(difficulty, rating)
    interval = fsrs.fsrs_next_interval(stability)

    # Stability should increase
    assert stability > 1.0
    assert interval > 1.0

    # Second review after calculated interval - easy rating
    elapsed = interval
    rating = 4
    stability2 = fsrs.fsrs_new_stability(stability, difficulty, rating, elapsed)
    difficulty2 = fsrs.fsrs_new_difficulty(difficulty, rating)

    # Should continue improving
    assert stability2 > stability
    assert difficulty2 < difficulty  # Difficulty decreases with easy rating
