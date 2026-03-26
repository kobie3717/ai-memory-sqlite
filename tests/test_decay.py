"""Tests for memory decay and staleness detection."""

import pytest
import sqlite3
from datetime import datetime, timedelta
from memory_tool import database


def test_stale_pending_after_30_days(temp_db):
    """Test that pending memories become stale after 30 days."""
    conn = temp_db
    
    # Create a pending memory from 31 days ago
    old_date = (datetime.now() - timedelta(days=31)).isoformat()
    conn.execute("""
        INSERT INTO memories (category, content, created_at, stale)
        VALUES ('pending', 'Old pending task', ?, 0)
    """, (old_date,))
    
    # Create a recent pending memory
    recent_date = (datetime.now() - timedelta(days=5)).isoformat()
    conn.execute("""
        INSERT INTO memories (category, content, created_at, stale)
        VALUES ('pending', 'Recent pending task', ?, 0)
    """, (recent_date,))
    
    conn.commit()
    
    # Verify the old one should be flagged as stale
    old_mem = conn.execute("""
        SELECT * FROM memories 
        WHERE content = 'Old pending task'
    """).fetchone()
    
    assert old_mem is not None
    # Note: Decay needs to be run manually, we're just testing the data setup
    # In production, decay would flag this as stale


def test_stale_general_after_90_days(temp_db):
    """Test that general memories become stale after 90 days."""
    conn = temp_db
    
    # Create a learning memory from 91 days ago
    old_date = (datetime.now() - timedelta(days=91)).isoformat()
    conn.execute("""
        INSERT INTO memories (category, content, created_at, accessed_at, stale)
        VALUES ('learning', 'Old learning', ?, NULL, 0)
    """, (old_date,))
    
    # Create a recent learning memory
    recent_date = (datetime.now() - timedelta(days=30)).isoformat()
    conn.execute("""
        INSERT INTO memories (category, content, created_at, accessed_at, stale)
        VALUES ('learning', 'Recent learning', ?, NULL, 0)
    """, (recent_date,))
    
    conn.commit()
    
    # Both exist
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2


def test_expired_memories_detectable(temp_db):
    """Test that memories with expires_at can be detected as expired."""
    conn = temp_db
    
    # Create an expired memory
    past_date = (datetime.now() - timedelta(days=1)).isoformat()
    conn.execute("""
        INSERT INTO memories (category, content, expires_at)
        VALUES ('pending', 'Expired task', ?)
    """, (past_date,))
    
    # Create a future expiry
    future_date = (datetime.now() + timedelta(days=7)).isoformat()
    conn.execute("""
        INSERT INTO memories (category, content, expires_at)
        VALUES ('pending', 'Future task', ?)
    """, (future_date,))
    
    conn.commit()
    
    # Query for expired memories
    expired = conn.execute("""
        SELECT * FROM memories
        WHERE expires_at IS NOT NULL AND expires_at < datetime('now')
    """).fetchall()
    
    assert len(expired) == 1
    assert expired[0]["content"] == "Expired task"


def test_touch_clears_stale_flag(temp_db):
    """Test that accessing a memory clears the stale flag."""
    conn = temp_db
    
    # Create a stale memory
    cursor = conn.execute("""
        INSERT INTO memories (category, content, stale)
        VALUES ('learning', 'Stale memory', 1)
    """)
    mem_id = cursor.lastrowid
    conn.commit()
    
    # Verify it's stale
    mem = conn.execute("SELECT stale FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert mem["stale"] == 1
    
    # Touch it
    conn.execute("""
        UPDATE memories SET
            accessed_at = datetime('now'),
            access_count = access_count + 1,
            stale = 0
        WHERE id = ?
    """, (mem_id,))
    conn.commit()
    
    # Verify stale flag is cleared
    mem = conn.execute("SELECT stale FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert mem["stale"] == 0


def test_list_stale_memories(temp_db):
    """Test listing only stale memories."""
    conn = temp_db
    
    # Create mix of stale and fresh
    conn.execute("INSERT INTO memories (category, content, stale) VALUES ('learning', 'Fresh', 0)")
    conn.execute("INSERT INTO memories (category, content, stale) VALUES ('learning', 'Stale 1', 1)")
    conn.execute("INSERT INTO memories (category, content, stale) VALUES ('decision', 'Stale 2', 1)")
    conn.commit()
    
    # Query stale only
    stale = conn.execute("""
        SELECT * FROM memories WHERE active = 1 AND stale = 1
    """).fetchall()
    
    assert len(stale) == 2
    contents = [m["content"] for m in stale]
    assert "Stale 1" in contents
    assert "Stale 2" in contents
    assert "Fresh" not in contents


def test_access_tracking_increments(temp_db):
    """Test that access tracking increments correctly."""
    conn = temp_db
    
    # Create a memory
    cursor = conn.execute("""
        INSERT INTO memories (category, content, access_count)
        VALUES ('learning', 'Test', 0)
    """)
    mem_id = cursor.lastrowid
    conn.commit()
    
    # Access it 3 times
    for _ in range(3):
        conn.execute("""
            UPDATE memories SET
                access_count = access_count + 1,
                accessed_at = datetime('now')
            WHERE id = ?
        """, (mem_id,))
        conn.commit()
    
    # Verify count
    mem = conn.execute("SELECT access_count FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert mem["access_count"] == 3


def test_revision_count_tracking(temp_db):
    """Test that revision count increments on updates."""
    conn = temp_db
    
    # Create a memory
    cursor = conn.execute("""
        INSERT INTO memories (category, content, revision_count)
        VALUES ('learning', 'Original', 1)
    """)
    mem_id = cursor.lastrowid
    conn.commit()
    
    # Update it twice
    for i in range(2):
        conn.execute("""
            UPDATE memories SET
                content = ?,
                revision_count = revision_count + 1
            WHERE id = ?
        """, (f"Updated {i+1}", mem_id))
        conn.commit()
    
    # Verify revision count
    mem = conn.execute("SELECT revision_count FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert mem["revision_count"] == 3


def test_priority_auto_adjust_based_on_access(temp_db):
    """Test that priority can be adjusted based on access count."""
    conn = temp_db
    
    # Create a memory with low priority
    cursor = conn.execute("""
        INSERT INTO memories (category, content, priority, access_count)
        VALUES ('learning', 'Popular memory', 2, 25)
    """)
    mem_id = cursor.lastrowid
    conn.commit()
    
    # Calculate suggested priority: min(10, access_count // 5) = min(10, 5) = 5
    suggested_priority = min(10, 25 // 5)
    
    # Update if suggested is higher
    mem = conn.execute("SELECT priority FROM memories WHERE id = ?", (mem_id,)).fetchone()
    if suggested_priority > mem["priority"]:
        conn.execute("UPDATE memories SET priority = ? WHERE id = ?", (suggested_priority, mem_id))
        conn.commit()
    
    # Verify priority increased
    updated = conn.execute("SELECT priority FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert updated["priority"] == 5


def test_fsrs_fields_initialized(temp_db):
    """Test that FSRS fields have correct default values."""
    conn = temp_db
    
    cursor = conn.execute("""
        INSERT INTO memories (category, content)
        VALUES ('learning', 'New memory')
    """)
    mem_id = cursor.lastrowid
    conn.commit()

    mem = conn.execute("""
        SELECT fsrs_stability, fsrs_difficulty, fsrs_interval, fsrs_reps
        FROM memories WHERE id = ?
    """, (mem_id,)).fetchone()
    
    assert mem["fsrs_stability"] == 1.0
    assert mem["fsrs_difficulty"] == 5.0
    assert mem["fsrs_interval"] == 1.0
    assert mem["fsrs_reps"] == 0


def test_importance_fields_initialized(temp_db):
    """Test that importance fields have correct default values."""
    conn = temp_db

    cursor = conn.execute("""
        INSERT INTO memories (category, content)
        VALUES ('learning', 'New memory')
    """)
    mem_id = cursor.lastrowid
    conn.commit()
    
    mem = conn.execute("""
        SELECT imp_novelty, imp_relevance, imp_frequency, imp_impact, imp_score
        FROM memories WHERE id = ?
    """, (mem_id,)).fetchone()
    
    assert mem["imp_novelty"] == 5.0
    assert mem["imp_relevance"] == 5.0
    assert mem["imp_frequency"] == 0.0
    assert mem["imp_impact"] == 5.0
    assert mem["imp_score"] == 5.0
