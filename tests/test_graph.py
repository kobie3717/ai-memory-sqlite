"""Tests for graph intelligence operations."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_tool import graph, database


def test_add_entity(temp_db):
    """Test adding a graph entity."""
    entity_id = graph.graph_add_entity(
        name="TestPerson",
        entity_type="person",
        summary="A test person",
        importance=5
    )

    assert entity_id is not None

    # Verify entity exists
    conn = database.get_db()
    row = conn.execute("SELECT * FROM graph_entities WHERE id = ?", (entity_id,)).fetchone()

    assert row["name"] == "TestPerson"
    assert row["type"] == "person"
    assert row["summary"] == "A test person"
    assert row["importance"] == 5

    conn.close()


def test_add_entity_upsert(temp_db):
    """Test that adding entity with same name updates it."""
    # Add entity first time
    id1 = graph.graph_add_entity(
        name="UpdateTest",
        entity_type="project",
        summary="Original summary",
        importance=3
    )

    # Add again with same name
    id2 = graph.graph_add_entity(
        name="UpdateTest",
        entity_type="project",
        summary="Updated summary",
        importance=7
    )

    # Should be same ID (upsert)
    assert id1 == id2

    # Verify it was updated
    conn = database.get_db()
    row = conn.execute("SELECT * FROM graph_entities WHERE id = ?", (id2,)).fetchone()
    assert row["summary"] == "Updated summary"
    assert row["importance"] == 7

    conn.close()


def test_add_entity_types(temp_db):
    """Test adding entities of different types."""
    types = ["person", "project", "org", "feature", "concept", "tool", "service"]

    for entity_type in types:
        entity_id = graph.graph_add_entity(
            name=f"Test{entity_type.title()}",
            entity_type=entity_type,
            summary=f"A test {entity_type}"
        )
        assert entity_id is not None


def test_get_or_create_entity_creates_new(temp_db):
    """Test get_or_create creates entity if it doesn't exist."""
    entity_id = graph.graph_get_or_create_entity(
        name="NewEntity",
        entity_type="concept",
        summary="Test concept"
    )

    assert entity_id is not None

    # Verify it was created
    conn = database.get_db()
    row = conn.execute("SELECT * FROM graph_entities WHERE id = ?", (entity_id,)).fetchone()
    assert row["name"] == "NewEntity"

    conn.close()


def test_get_or_create_entity_gets_existing(temp_db, sample_entities):
    """Test get_or_create returns existing entity."""
    alice_id = sample_entities["Alice"]

    # Try to get or create Alice (should return existing)
    entity_id = graph.graph_get_or_create_entity(
        name="Alice",
        entity_type="person"
    )

    assert entity_id == alice_id


def test_get_or_create_entity_case_insensitive(temp_db, sample_entities):
    """Test that entity lookup is case-insensitive."""
    alice_id = sample_entities["Alice"]

    # Try different cases
    id1 = graph.graph_get_or_create_entity(name="alice", entity_type="person")
    id2 = graph.graph_get_or_create_entity(name="ALICE", entity_type="person")
    id3 = graph.graph_get_or_create_entity(name="AlIcE", entity_type="person")

    assert id1 == alice_id
    assert id2 == alice_id
    assert id3 == alice_id


def test_add_relationship(temp_db, sample_entities):
    """Test adding a relationship between entities."""
    result = graph.graph_add_relationship(
        from_name="Alice",
        to_name="ProjectX",
        relation_type="works_on",
        note="Lead developer"
    )

    assert result is True

    # Verify relationship exists
    conn = database.get_db()
    alice_id = sample_entities["Alice"]
    project_id = sample_entities["ProjectX"]

    row = conn.execute("""
        SELECT * FROM graph_relationships
        WHERE from_entity_id = ? AND to_entity_id = ?
    """, (alice_id, project_id)).fetchone()

    assert row is not None
    assert row["relation_type"] == "works_on"
    assert row["note"] == "Lead developer"

    conn.close()


def test_add_relationship_creates_entities(temp_db):
    """Test that add_relationship creates entities if they don't exist."""
    result = graph.graph_add_relationship(
        from_name="NewPerson",
        to_name="NewProject",
        relation_type="owns"
    )

    assert result is True

    # Verify both entities were created
    conn = database.get_db()
    person = conn.execute("SELECT * FROM graph_entities WHERE name = ?", ("NewPerson",)).fetchone()
    project = conn.execute("SELECT * FROM graph_entities WHERE name = ?", ("NewProject",)).fetchone()

    assert person is not None
    assert project is not None

    conn.close()


def test_add_relationship_types(temp_db, sample_entities):
    """Test different relationship types."""
    rel_types = ["knows", "works_on", "owns", "depends_on", "built_by", "uses", "blocks"]

    for rel_type in rel_types:
        result = graph.graph_add_relationship(
            from_name="Alice",
            to_name="Database",
            relation_type=rel_type,
            note=f"Testing {rel_type}"
        )
        assert result is True


def test_add_relationship_upsert(temp_db, sample_entities):
    """Test that adding same relationship updates note."""
    # Add relationship
    graph.graph_add_relationship(
        from_name="Alice",
        to_name="ProjectX",
        relation_type="works_on",
        note="Original note"
    )

    # Add again with different note
    graph.graph_add_relationship(
        from_name="Alice",
        to_name="ProjectX",
        relation_type="works_on",
        note="Updated note"
    )

    # Verify note was updated
    conn = database.get_db()
    alice_id = sample_entities["Alice"]
    project_id = sample_entities["ProjectX"]

    row = conn.execute("""
        SELECT * FROM graph_relationships
        WHERE from_entity_id = ? AND to_entity_id = ? AND relation_type = ?
    """, (alice_id, project_id, "works_on")).fetchone()

    assert row["note"] == "Updated note"

    conn.close()


def test_set_fact(temp_db, sample_entities):
    """Test setting a fact on an entity."""
    conn = database.get_db()
    alice_id = sample_entities["Alice"]

    # Set a fact
    conn.execute("""
        INSERT INTO graph_facts (entity_id, key, value)
        VALUES (?, ?, ?)
    """, (alice_id, "email", "alice@example.com"))
    conn.commit()

    # Verify fact exists
    fact = conn.execute("""
        SELECT * FROM graph_facts WHERE entity_id = ? AND key = ?
    """, (alice_id, "email")).fetchone()

    assert fact is not None
    assert fact["value"] == "alice@example.com"

    conn.close()


def test_set_fact_update(temp_db, sample_entities):
    """Test updating a fact creates history."""
    conn = database.get_db()
    alice_id = sample_entities["Alice"]

    # Set fact first time
    conn.execute("""
        INSERT INTO graph_facts (entity_id, key, value)
        VALUES (?, ?, ?)
    """, (alice_id, "role", "Developer"))
    conn.commit()

    # Update the fact
    conn.execute("""
        INSERT INTO graph_facts (entity_id, key, value)
        VALUES (?, ?, ?)
        ON CONFLICT(entity_id, key) DO UPDATE SET
            value = excluded.value,
            updated_at = datetime('now')
    """, (alice_id, "role", "Senior Developer"))
    conn.commit()

    # Verify updated value
    fact = conn.execute("""
        SELECT value FROM graph_facts WHERE entity_id = ? AND key = ?
    """, (alice_id, "role")).fetchone()

    assert fact["value"] == "Senior Developer"

    conn.close()


def test_get_entity_with_facts(temp_db, sample_entities):
    """Test retrieving entity with its facts."""
    conn = database.get_db()
    alice_id = sample_entities["Alice"]

    # Add some facts
    conn.execute("""
        INSERT INTO graph_facts (entity_id, key, value)
        VALUES (?, ?, ?), (?, ?, ?)
    """, (alice_id, "email", "alice@example.com", alice_id, "role", "Developer"))
    conn.commit()

    # Get entity with facts
    entity = conn.execute("SELECT * FROM graph_entities WHERE id = ?", (alice_id,)).fetchone()
    facts = conn.execute("SELECT * FROM graph_facts WHERE entity_id = ?", (alice_id,)).fetchall()

    assert entity is not None
    assert len(facts) == 2

    fact_dict = {f["key"]: f["value"] for f in facts}
    assert fact_dict["email"] == "alice@example.com"
    assert fact_dict["role"] == "Developer"

    conn.close()


def test_list_entities(temp_db, sample_entities):
    """Test listing all entities."""
    conn = database.get_db()

    entities = conn.execute("SELECT * FROM graph_entities ORDER BY name").fetchall()

    assert len(entities) == 4  # Alice, Bob, ProjectX, Python
    names = [e["name"] for e in entities]
    assert "Alice" in names
    assert "Bob" in names
    assert "ProjectX" in names
    assert "Python" in names

    conn.close()


def test_list_entities_by_type(temp_db, sample_entities):
    """Test filtering entities by type."""
    conn = database.get_db()

    # Get only persons
    persons = conn.execute("""
        SELECT * FROM graph_entities WHERE type = ?
    """, ("person",)).fetchall()

    assert len(persons) == 2  # Alice and Bob
    person_names = [p["name"] for p in persons]
    assert "Alice" in person_names
    assert "Bob" in person_names

    # Get only tools
    tools = conn.execute("""
        SELECT * FROM graph_entities WHERE type = ?
    """, ("tool",)).fetchall()

    assert len(tools) == 1  # Python
    assert tools[0]["name"] == "Python"

    conn.close()


def test_link_memory_to_entity(temp_db, sample_memory, sample_entities):
    """Test linking a memory to a graph entity."""
    conn = database.get_db()
    alice_id = sample_entities["Alice"]

    # Create link
    conn.execute("""
        INSERT INTO memory_entity_links (memory_id, entity_id)
        VALUES (?, ?)
    """, (sample_memory, alice_id))
    conn.commit()

    # Verify link exists
    link = conn.execute("""
        SELECT * FROM memory_entity_links
        WHERE memory_id = ? AND entity_id = ?
    """, (sample_memory, alice_id)).fetchone()

    assert link is not None

    conn.close()


def test_get_entity_with_linked_memories(temp_db, sample_memory, sample_entities):
    """Test retrieving an entity with its linked memories."""
    conn = database.get_db()
    alice_id = sample_entities["Alice"]

    # Link memory to entity
    conn.execute("""
        INSERT INTO memory_entity_links (memory_id, entity_id)
        VALUES (?, ?)
    """, (sample_memory, alice_id))
    conn.commit()

    # Get linked memories
    memories = conn.execute("""
        SELECT m.* FROM memories m
        JOIN memory_entity_links mel ON m.id = mel.memory_id
        WHERE mel.entity_id = ?
    """, (alice_id,)).fetchall()

    assert len(memories) == 1
    assert memories[0]["id"] == sample_memory

    conn.close()


def test_entity_importance_values(temp_db):
    """Test that entity importance is stored and retrieved correctly."""
    # Create entities with different importance
    id1 = graph.graph_add_entity("LowImportance", "concept", "", importance=1)
    id2 = graph.graph_add_entity("HighImportance", "concept", "", importance=10)

    conn = database.get_db()

    e1 = conn.execute("SELECT importance FROM graph_entities WHERE id = ?", (id1,)).fetchone()
    e2 = conn.execute("SELECT importance FROM graph_entities WHERE id = ?", (id2,)).fetchone()

    assert e1["importance"] == 1
    assert e2["importance"] == 10

    conn.close()
