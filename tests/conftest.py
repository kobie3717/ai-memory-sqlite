"""Pytest configuration and fixtures for memory-tool tests."""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Create a fresh test database in a temporary directory.

    This fixture:
    1. Creates a temp directory for the test database
    2. Monkey-patches the DB_PATH to use the temp database
    3. Initializes the schema
    4. Returns the connection
    5. Cleans up after the test
    """
    # Create temp database path
    db_path = tmp_path / "test_memories.db"

    # Monkey-patch the config to use temp DB BEFORE importing anything else
    from memory_tool import config
    original_db_path = config.DB_PATH
    monkeypatch.setattr(config, "DB_PATH", db_path)

    # Also patch at module level for get_db() calls
    import memory_tool.database as db_module
    monkeypatch.setattr(db_module, "DB_PATH", db_path)

    # Patch in all modules that import get_db
    for module_name in ['memory_ops', 'relations', 'export', 'graph', 'snapshots', 'sync', 'utils']:
        try:
            module = __import__(f'memory_tool.{module_name}', fromlist=[module_name])
            if hasattr(module, 'DB_PATH'):
                monkeypatch.setattr(module, "DB_PATH", db_path)
        except:
            pass

    # Initialize the database with schema
    from memory_tool.database import init_db
    db_module.init_db()

    # Get connection for the test - use direct SQLite connection to temp DB
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    yield conn

    # Cleanup
    conn.close()


@pytest.fixture
def db_with_samples(temp_db):
    """Database with sample memories for testing.

    Creates a variety of test memories covering different categories,
    projects, and edge cases.
    """
    conn = temp_db

    # Sample memories
    samples = [
        {
            "category": "project",
            "content": "FlashVault VPN backend uses Node.js and PostgreSQL",
            "project": "FlashVault",
            "tags": "nodejs,database,backend",
            "priority": 7
        },
        {
            "category": "decision",
            "content": "Decided to use FSRS-6 for spaced repetition in memory system",
            "project": None,
            "tags": "architecture,fsrs",
            "priority": 9
        },
        {
            "category": "error",
            "content": "PM2 restart loop caused by missing environment variable",
            "project": "FlashVault",
            "tags": "pm2,error,production",
            "priority": 8
        },
        {
            "category": "learning",
            "content": "SQLite FTS5 requires content='table_name' for external content tables",
            "project": None,
            "tags": "database,sqlite,fts",
            "priority": 6
        },
        {
            "category": "pending",
            "content": "Add pytest tests to memory-tool",
            "project": None,
            "tags": "testing,todo",
            "priority": 5
        },
        {
            "category": "preference",
            "content": "Always use pytest over unittest for Python testing",
            "project": None,
            "tags": "testing,python",
            "priority": 4
        },
    ]

    # Insert samples
    for mem in samples:
        conn.execute("""
            INSERT INTO memories (category, content, project, tags, priority)
            VALUES (?, ?, ?, ?, ?)
        """, (mem["category"], mem["content"], mem["project"], mem["tags"], mem["priority"]))

    conn.commit()

    return conn


@pytest.fixture
def sample_graph_entities(temp_db):
    """Database with sample graph entities and relationships."""
    conn = temp_db

    # Add entities
    entities = [
        ("FlashVault", "project", "Mobile-first VPN business for South Africa", 9),
        ("WhatsAuction", "project", "WhatsApp-based auction platform", 8),
        ("Kobus", "person", "Developer and architect", 7),
        ("PostgreSQL", "tool", "Relational database", 6),
        ("Node.js", "tool", "JavaScript runtime", 6),
    ]

    for name, etype, summary, importance in entities:
        conn.execute("""
            INSERT INTO graph_entities (name, type, summary, importance)
            VALUES (?, ?, ?, ?)
        """, (name, etype, summary, importance))

    # Add relationships
    relationships = [
        ("Kobus", "FlashVault", "works_on", "Lead developer"),
        ("Kobus", "WhatsAuction", "works_on", "Architect"),
        ("FlashVault", "PostgreSQL", "uses", "Database layer"),
        ("FlashVault", "Node.js", "built_by", "Backend runtime"),
    ]

    for from_name, to_name, rel_type, note in relationships:
        from_id = conn.execute("SELECT id FROM graph_entities WHERE name = ?", (from_name,)).fetchone()[0]
        to_id = conn.execute("SELECT id FROM graph_entities WHERE name = ?", (to_name,)).fetchone()[0]
        conn.execute("""
            INSERT INTO graph_relationships (from_entity_id, to_entity_id, relation_type, note)
            VALUES (?, ?, ?, ?)
        """, (from_id, to_id, rel_type, note))

    conn.commit()

    return conn


@pytest.fixture
def sample_memory(temp_db):
    """Single sample memory for update/delete tests."""
    conn = temp_db

    cursor = conn.execute("""
        INSERT INTO memories (category, content, project, tags, priority, active)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("learning", "Sample memory for testing", "TestProject", "test,sample", 5, 1))
    mem_id = cursor.lastrowid
    conn.commit()

    return mem_id


@pytest.fixture
def sample_memories(temp_db):
    """Multiple sample memories for relationship and search tests."""
    conn = temp_db

    # Create several test memories
    memories = [
        ("learning", "Python is a high-level programming language", "TestProject", "python,programming", 7),
        ("decision", "Use PostgreSQL for production database", "TestProject", "database,postgres", 8),
        ("error", "Fixed memory leak in query optimization", "TestProject", "bug,performance", 6),
        ("learning", "SQLite supports full-text search with FTS5", "TestProject", "sqlite,fts", 5),
    ]

    mem_ids = []
    for cat, content, project, tags, priority in memories:
        cursor = conn.execute("""
            INSERT INTO memories (category, content, project, tags, priority, active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (cat, content, project, tags, priority, 1))
        mem_ids.append(cursor.lastrowid)

    conn.commit()

    return mem_ids


@pytest.fixture
def sample_entities(temp_db):
    """Sample graph entities for testing."""
    conn = temp_db

    # Create test entities
    entities = {
        "Alice": ("Alice", "person", "Software developer"),
        "Bob": ("Bob", "person", "Project manager"),
        "ProjectX": ("ProjectX", "project", "Main project"),
        "Python": ("Python", "tool", "Programming language"),
    }

    entity_ids = {}
    for key, (name, etype, summary) in entities.items():
        cursor = conn.execute("""
            INSERT INTO graph_entities (name, type, summary, importance)
            VALUES (?, ?, ?, ?)
        """, (name, etype, summary, 5))
        entity_ids[key] = cursor.lastrowid

    conn.commit()

    return entity_ids
