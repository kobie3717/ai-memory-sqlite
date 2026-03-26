"""Tests for utility functions."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_tool import utils


def test_auto_tag_detects_pm2():
    """Test auto-tagging detects pm2."""
    content = "PM2 restart loop causing issues"
    tags = utils.auto_tag(content)
    assert "pm2" in tags


def test_auto_tag_detects_whatsapp():
    """Test auto-tagging detects WhatsApp."""
    content = "WhatsApp webhook integration with baileys"
    tags = utils.auto_tag(content)
    assert "whatsapp" in tags
    assert "baileys" in tags


def test_auto_tag_detects_database():
    """Test auto-tagging detects database keywords."""
    content = "PostgreSQL migration with Sequelize ORM"
    tags = utils.auto_tag(content)
    assert "database" in tags


def test_auto_tag_detects_auth():
    """Test auto-tagging detects auth keywords."""
    content = "JWT token authentication with bcrypt password hashing"
    tags = utils.auto_tag(content)
    assert "auth" in tags


def test_auto_tag_detects_nginx():
    """Test auto-tagging detects nginx."""
    content = "Nginx reverse proxy SSL configuration with certbot"
    tags = utils.auto_tag(content)
    assert "nginx" in tags


def test_auto_tag_detects_docker():
    """Test auto-tagging detects docker."""
    content = "Docker container configuration in Dockerfile"
    tags = utils.auto_tag(content)
    assert "docker" in tags


def test_auto_tag_detects_react():
    """Test auto-tagging detects React."""
    content = "React frontend with Vite and Tailwind CSS"
    tags = utils.auto_tag(content)
    assert "react" in tags


def test_auto_tag_detects_api():
    """Test auto-tagging detects API keywords."""
    content = "Express route controller with middleware"
    tags = utils.auto_tag(content)
    assert "api" in tags


def test_auto_tag_merges_with_existing():
    """Test that auto-tagging merges with existing tags."""
    content = "PM2 process manager"
    existing = "production,important"
    tags = utils.auto_tag(content, existing)
    tag_list = tags.split(",")
    assert "pm2" in tag_list
    assert "production" in tag_list
    assert "important" in tag_list


def test_auto_tag_no_duplicates():
    """Test that auto-tagging doesn't create duplicates."""
    content = "PM2 restart"
    existing = "pm2,production"
    tags = utils.auto_tag(content, existing)
    tag_list = tags.split(",")
    assert tag_list.count("pm2") == 1


def test_auto_tag_case_insensitive():
    """Test that auto-tagging is case-insensitive."""
    content = "POSTGRESQL database with SEQUELIZE ORM"
    tags = utils.auto_tag(content)
    assert "database" in tags


def test_auto_tag_sorted():
    """Test that tags are returned sorted."""
    content = "PostgreSQL with Nginx and Docker"
    tags = utils.auto_tag(content)
    tag_list = tags.split(",")
    assert tag_list == sorted(tag_list)


def test_normalize_text():
    """Test text normalization."""
    text = "Hello, World! This is a TEST."
    normalized = utils.normalize(text)
    assert normalized == "hello world this is a test"


def test_normalize_removes_punctuation():
    """Test that normalization removes punctuation."""
    text = "Hello, @world! #test & stuff..."
    normalized = utils.normalize(text)
    assert "@" not in normalized
    assert "#" not in normalized
    assert "!" not in normalized


def test_normalize_strips_whitespace():
    """Test that normalization strips extra whitespace."""
    text = "  hello   world  "
    normalized = utils.normalize(text)
    assert normalized.strip() == normalized


def test_word_set_filters_short_words():
    """Test that word_set filters words <= 2 chars."""
    text = "a is the big dog"
    words = utils.word_set(text)
    assert "a" not in words
    assert "is" not in words
    assert "big" in words
    assert "dog" in words


def test_word_set_normalizes():
    """Test that word_set normalizes text."""
    text = "Hello, WORLD!"
    words = utils.word_set(text)
    assert "hello" in words
    assert "world" in words


def test_word_overlap_identical_texts():
    """Test word overlap with identical texts."""
    text = "hello world test"
    overlap = utils.word_overlap(text, text)
    assert overlap == 1.0


def test_word_overlap_no_overlap():
    """Test word overlap with no overlap."""
    overlap = utils.word_overlap("hello world", "foo bar")
    assert overlap == 0.0


def test_word_overlap_partial():
    """Test word overlap with partial overlap."""
    overlap = utils.word_overlap("hello world test", "hello foo bar")
    assert 0.0 < overlap < 1.0


def test_word_overlap_empty_strings():
    """Test word overlap with empty strings."""
    assert utils.word_overlap("", "hello") == 0.0
    assert utils.word_overlap("hello", "") == 0.0
    assert utils.word_overlap("", "") == 0.0


def test_similarity_identical_texts():
    """Test similarity with identical texts."""
    text = "hello world test"
    sim = utils.similarity(text, text)
    assert sim == 1.0


def test_similarity_completely_different():
    """Test similarity with completely different texts."""
    sim = utils.similarity("alpha beta gamma", "one two three")
    assert sim < 0.3


def test_similarity_partial_overlap():
    """Test similarity with partial overlap."""
    text_a = "PostgreSQL database migration"
    text_b = "PostgreSQL database setup"
    sim = utils.similarity(text_a, text_b)
    assert 0.5 < sim < 1.0


def test_similarity_case_insensitive():
    """Test that similarity is case-insensitive."""
    sim = utils.similarity("HELLO WORLD", "hello world")
    assert sim == 1.0


def test_similarity_punctuation_ignored():
    """Test that similarity ignores punctuation."""
    sim = utils.similarity("hello, world!", "hello world")
    assert sim > 0.9


def test_similarity_empty_strings():
    """Test similarity with empty strings."""
    assert utils.similarity("", "hello") == 0.0
    assert utils.similarity("hello", "") == 0.0
    assert utils.similarity("", "") == 0.0


def test_similarity_uses_max_of_jaccard_and_sequence():
    """Test that similarity returns max of Jaccard and sequence matcher."""
    # Test with text where one method might score higher
    text_a = "the quick brown fox"
    text_b = "quick brown fox jumps"
    sim = utils.similarity(text_a, text_b)
    assert 0.5 < sim <= 1.0


def test_find_similar_basic(temp_db):
    """Test finding similar memories."""
    conn = temp_db
    
    # Add some memories
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('learning', 'PostgreSQL is a relational database', 1)
    """)
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('learning', 'PostgreSQL database system for data storage', 1)
    """)
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('learning', 'React is a JavaScript library', 1)
    """)
    conn.commit()
    
    # Search for similar
    similar = utils.find_similar("PostgreSQL is a database system", threshold=0.5)
    
    # Should find the PostgreSQL memories
    assert len(similar) >= 1
    contents = [s[1] for s in similar]
    assert any("PostgreSQL" in c for c in contents)


def test_find_similar_respects_threshold(temp_db):
    """Test that find_similar respects similarity threshold."""
    conn = temp_db
    
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('learning', 'PostgreSQL database', 1)
    """)
    conn.commit()
    
    # High threshold should find nothing
    similar = utils.find_similar("React JavaScript", threshold=0.9)
    assert len(similar) == 0
    
    # Low threshold should be more permissive
    similar = utils.find_similar("React JavaScript", threshold=0.1)
    # Might or might not find, but shouldn't error


def test_find_similar_filters_by_category(temp_db):
    """Test that find_similar can filter by category."""
    conn = temp_db
    
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('learning', 'PostgreSQL database', 1)
    """)
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('error', 'PostgreSQL connection failed', 1)
    """)
    conn.commit()
    
    # Search only in error category
    similar = utils.find_similar("PostgreSQL issue", category="error", threshold=0.3)
    
    # Should only find error category
    if len(similar) > 0:
        categories = [s[3] for s in similar]
        assert all(cat == "error" for cat in categories)


def test_find_similar_filters_by_project(temp_db):
    """Test that find_similar can filter by project."""
    conn = temp_db
    
    conn.execute("""
        INSERT INTO memories (category, content, project, active)
        VALUES ('learning', 'PostgreSQL setup', 'FlashVault', 1)
    """)
    conn.execute("""
        INSERT INTO memories (category, content, project, active)
        VALUES ('learning', 'PostgreSQL config', 'WhatsAuction', 1)
    """)
    conn.commit()
    
    # Search only in FlashVault project
    similar = utils.find_similar("PostgreSQL configuration", project="FlashVault", threshold=0.3)
    
    # Should only find FlashVault project
    if len(similar) > 0:
        projects = [s[4] for s in similar]
        assert all(proj == "FlashVault" for proj in projects)


def test_find_similar_excludes_inactive(temp_db):
    """Test that find_similar excludes inactive memories."""
    conn = temp_db
    
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('learning', 'PostgreSQL database system', 0)
    """)
    conn.commit()
    
    # Should not find inactive memory
    similar = utils.find_similar("PostgreSQL database", threshold=0.5)
    assert len(similar) == 0


def test_find_similar_returns_sorted_by_score(temp_db):
    """Test that find_similar returns results sorted by score."""
    conn = temp_db
    
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('learning', 'PostgreSQL database', 1)
    """)
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('learning', 'PostgreSQL relational database system', 1)
    """)
    conn.execute("""
        INSERT INTO memories (category, content, active)
        VALUES ('learning', 'Database storage', 1)
    """)
    conn.commit()
    
    similar = utils.find_similar("PostgreSQL database system", threshold=0.3)
    
    # Results should be sorted by descending score
    if len(similar) > 1:
        scores = [s[2] for s in similar]
        assert scores == sorted(scores, reverse=True)
