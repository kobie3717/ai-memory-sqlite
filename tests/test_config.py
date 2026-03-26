"""Tests for configuration constants."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_tool import config


def test_db_path_exists():
    """Test that DB_PATH is defined."""
    assert hasattr(config, "DB_PATH")
    assert isinstance(config.DB_PATH, Path)


def test_memory_md_path_exists():
    """Test that MEMORY_MD_PATH is defined."""
    assert hasattr(config, "MEMORY_MD_PATH")
    assert isinstance(config.MEMORY_MD_PATH, Path)


def test_staleness_thresholds_sane():
    """Test that staleness thresholds are reasonable."""
    assert config.STALE_PENDING_DAYS > 0
    assert config.STALE_GENERAL_DAYS > 0
    assert config.DEPRIORITIZE_DAYS > 0
    assert config.STALE_GENERAL_DAYS > config.STALE_PENDING_DAYS


def test_similarity_threshold_bounded():
    """Test that similarity threshold is between 0 and 1."""
    assert 0.0 <= config.SIMILARITY_THRESHOLD <= 1.0


def test_max_memory_md_bytes_sane():
    """Test that MEMORY.md size cap is reasonable."""
    assert config.MAX_MEMORY_MD_BYTES > 0
    assert config.MAX_MEMORY_MD_BYTES >= 1024  # At least 1KB


def test_embedding_dim_valid():
    """Test that embedding dimension is a positive integer."""
    assert isinstance(config.EMBEDDING_DIM, int)
    assert config.EMBEDDING_DIM > 0
    assert config.EMBEDDING_DIM == 384  # all-MiniLM-L6-v2


def test_rrf_k_valid():
    """Test that RRF constant is positive."""
    assert isinstance(config.RRF_K, int)
    assert config.RRF_K > 0


def test_project_paths_defined():
    """Test that project path mappings exist."""
    assert hasattr(config, "PROJECT_PATHS")
    assert isinstance(config.PROJECT_PATHS, dict)
    assert len(config.PROJECT_PATHS) > 0


def test_project_paths_valid():
    """Test that project paths are strings."""
    for path, project in config.PROJECT_PATHS.items():
        assert isinstance(path, str)
        assert isinstance(project, str)
        assert len(project) > 0


def test_auto_tag_rules_defined():
    """Test that auto-tag rules exist."""
    assert hasattr(config, "AUTO_TAG_RULES")
    assert isinstance(config.AUTO_TAG_RULES, dict)
    assert len(config.AUTO_TAG_RULES) > 0


def test_auto_tag_rules_valid():
    """Test that auto-tag rules have valid structure."""
    for tag, keywords in config.AUTO_TAG_RULES.items():
        assert isinstance(tag, str)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(kw, str) for kw in keywords)


def test_backup_dir_defined():
    """Test that backup directory is defined."""
    assert hasattr(config, "BACKUP_DIR")
    assert isinstance(config.BACKUP_DIR, Path)


def test_topics_dir_defined():
    """Test that topics directory is defined."""
    assert hasattr(config, "TOPICS_DIR")
    assert isinstance(config.TOPICS_DIR, Path)


def test_memory_dir_defined():
    """Test that memory directory is defined."""
    assert hasattr(config, "MEMORY_DIR")
    assert isinstance(config.MEMORY_DIR, Path)


def test_openclaw_paths_defined():
    """Test that OpenClaw bridge paths are defined."""
    assert hasattr(config, "OPENCLAW_MEMORY_DIR")
    assert hasattr(config, "OPENCLAW_GRAPH_DB")
    assert hasattr(config, "SYNC_STATE_FILE")


def test_model_dir_defined():
    """Test that model directory is defined."""
    assert hasattr(config, "MODEL_DIR")
    assert isinstance(config.MODEL_DIR, Path)


def test_common_auto_tags_present():
    """Test that common auto-tags are present."""
    assert "pm2" in config.AUTO_TAG_RULES
    assert "whatsapp" in config.AUTO_TAG_RULES
    assert "database" in config.AUTO_TAG_RULES
    assert "auth" in config.AUTO_TAG_RULES
    assert "nginx" in config.AUTO_TAG_RULES
    assert "docker" in config.AUTO_TAG_RULES


def test_flashvault_project_mapped():
    """Test that FlashVault paths are mapped."""
    flashvault_paths = [p for p, proj in config.PROJECT_PATHS.items() if proj == "FlashVault"]
    assert len(flashvault_paths) > 0


def test_whatsauction_project_mapped():
    """Test that WhatsAuction paths are mapped."""
    whatsauction_paths = [p for p, proj in config.PROJECT_PATHS.items() if proj == "WhatsAuction"]
    assert len(whatsauction_paths) > 0
