"""Tests for memory CRUD operations."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_tool import memory_ops, database


def test_add_memory(temp_db):
    """Test adding a basic memory."""
    conn = database.get_db()

    # Add memory
    cursor = conn.execute("""
        INSERT INTO memories (category, content, project, tags, priority)
        VALUES (?, ?, ?, ?, ?)
    """, ("learning", "Test memory content", "TestProject", "test", 5))
    mem_id = cursor.lastrowid
    conn.commit()

    # Verify it exists
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert row is not None
    assert row["content"] == "Test memory content"
    assert row["category"] == "learning"
    assert row["project"] == "TestProject"
    assert row["active"] == 1

    conn.close()


def test_add_memory_with_tags(temp_db):
    """Test adding memory with tags."""
    conn = database.get_db()

    cursor = conn.execute("""
        INSERT INTO memories (category, content, tags)
        VALUES (?, ?, ?)
    """, ("learning", "Memory with tags", "python,testing,pytest"))
    mem_id = cursor.lastrowid
    conn.commit()

    row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert "python" in row["tags"]
    assert "testing" in row["tags"]
    assert "pytest" in row["tags"]

    conn.close()


def test_add_memory_with_project(temp_db):
    """Test adding memory with project filter."""
    conn = database.get_db()

    cursor = conn.execute("""
        INSERT INTO memories (category, content, project)
        VALUES (?, ?, ?)
    """, ("decision", "Project-specific decision", "MyProject"))
    mem_id = cursor.lastrowid
    conn.commit()

    row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert row["project"] == "MyProject"

    conn.close()


def test_update_memory(temp_db, sample_memory):
    """Test updating memory content."""
    conn = database.get_db()

    # Update the memory
    new_content = "Updated memory content"
    conn.execute("""
        UPDATE memories SET content = ?, updated_at = datetime('now')
        WHERE id = ?
    """, (new_content, sample_memory))
    conn.commit()

    # Verify update
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (sample_memory,)).fetchone()
    assert row["content"] == new_content

    conn.close()


def test_delete_memory(temp_db, sample_memory):
    """Test soft-deleting a memory."""
    conn = database.get_db()

    # Soft delete
    conn.execute("""
        UPDATE memories SET active = 0, updated_at = datetime('now')
        WHERE id = ?
    """, (sample_memory,))
    conn.commit()

    # Verify it's inactive
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (sample_memory,)).fetchone()
    assert row["active"] == 0

    conn.close()


def test_list_memories(temp_db, sample_memories):
    """Test listing all memories."""
    conn = database.get_db()

    rows = conn.execute("SELECT * FROM memories WHERE active = 1").fetchall()
    assert len(rows) == len(sample_memories)

    conn.close()


def test_list_by_project(temp_db, sample_memories):
    """Test filtering memories by project."""
    conn = database.get_db()

    # Get TestProject memories (all sample_memories use TestProject)
    rows = conn.execute("""
        SELECT * FROM memories WHERE project = ? AND active = 1
    """, ("TestProject",)).fetchall()

    assert len(rows) == 4  # All 4 sample memories have TestProject

    conn.close()


def test_list_by_category(temp_db, sample_memories):
    """Test filtering memories by category."""
    conn = database.get_db()

    rows = conn.execute("""
        SELECT * FROM memories WHERE category = ? AND active = 1
    """, ("learning",)).fetchall()

    assert len(rows) == 2  # Two learning memories in sample_memories
    assert any("Python" in row["content"] for row in rows)

    conn.close()


def test_get_memory(temp_db, sample_memory):
    """Test getting a single memory by ID."""
    conn = database.get_db()

    row = conn.execute("SELECT * FROM memories WHERE id = ?", (sample_memory,)).fetchone()
    assert row is not None
    assert row["id"] == sample_memory
    assert row["content"] == "Sample memory for testing"  # Match the fixture content

    conn.close()


def test_touch_memory_updates_access_tracking(temp_db, sample_memory):
    """Test that touching a memory updates access tracking."""
    conn = database.get_db()

    # Get initial access count
    row = conn.execute("SELECT access_count FROM memories WHERE id = ?", (sample_memory,)).fetchone()
    initial_count = row["access_count"]

    # Touch the memory
    memory_ops.touch_memory(conn, sample_memory)
    conn.commit()

    # Verify access count increased
    row = conn.execute("SELECT access_count, accessed_at FROM memories WHERE id = ?", (sample_memory,)).fetchone()
    assert row["access_count"] == initial_count + 1
    assert row["accessed_at"] is not None

    conn.close()


def test_memory_defaults(temp_db):
    """Test default values when creating a memory."""
    conn = database.get_db()

    cursor = conn.execute("""
        INSERT INTO memories (category, content)
        VALUES (?, ?)
    """, ("learning", "Minimal memory"))
    mem_id = cursor.lastrowid
    conn.commit()

    row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert row["active"] == 1
    assert row["priority"] == 0
    assert row["access_count"] == 0
    assert row["stale"] == 0
    assert row["source"] == "manual"
    assert row["revision_count"] == 1
    assert row["fsrs_stability"] == 1.0
    assert row["fsrs_difficulty"] == 5.0

    conn.close()


def test_fts_search_basic(temp_db, sample_memories):
    """Test full-text search functionality."""
    conn = database.get_db()

    # Search for "Python" (which is in sample_memories)
    rows = conn.execute("""
        SELECT m.* FROM memories m
        JOIN memories_fts fts ON m.id = fts.rowid
        WHERE memories_fts MATCH 'Python'
    """).fetchall()

    assert len(rows) >= 1
    assert any("python" in row["content"].lower() or "python" in (row["tags"] or "") for row in rows)

    conn.close()


def test_fts_search_no_results(temp_db, sample_memories):
    """Test FTS search with no matching results."""
    conn = database.get_db()

    rows = conn.execute("""
        SELECT m.* FROM memories m
        JOIN memories_fts fts ON m.id = fts.rowid
        WHERE memories_fts MATCH 'nonexistent_term_xyz'
    """).fetchall()

    assert len(rows) == 0

    conn.close()


def test_priority_filtering(temp_db, sample_memories):
    """Test filtering memories by priority."""
    conn = database.get_db()

    # Get high priority memories (>= 7)
    rows = conn.execute("""
        SELECT * FROM memories WHERE priority >= 7 AND active = 1
    """).fetchall()

    assert len(rows) == 2  # decision (priority 8) and learning (priority 7)

    conn.close()


def test_inactive_memories_excluded(temp_db, sample_memories):
    """Test that inactive memories are excluded from active queries."""
    conn = database.get_db()

    # Mark one memory as inactive
    conn.execute("UPDATE memories SET active = 0 WHERE id = ?", (sample_memories[0],))
    conn.commit()

    # Query active only
    rows = conn.execute("SELECT * FROM memories WHERE active = 1").fetchall()
    assert len(rows) == len(sample_memories) - 1

    conn.close()
