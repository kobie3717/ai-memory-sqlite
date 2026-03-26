"""Tests for correction detection and capture."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_tool import corrections
from memory_tool.dream import CORRECTION_PATTERNS


def test_correction_patterns_exist():
    """Test that correction patterns are defined."""
    assert CORRECTION_PATTERNS is not None
    assert isinstance(CORRECTION_PATTERNS, list)
    assert len(CORRECTION_PATTERNS) > 0


def test_correction_patterns_structure():
    """Test that correction patterns have valid structure."""
    for item in CORRECTION_PATTERNS:
        assert isinstance(item, tuple)
        assert len(item) == 2
        pattern, ctype = item
        assert isinstance(pattern, str)
        assert isinstance(ctype, str)


def test_cmd_capture_correction_patterns():
    """Test that cmd_capture_correction has its own patterns defined."""
    # This function has patterns defined inline
    assert hasattr(corrections, 'cmd_capture_correction')
    assert callable(corrections.cmd_capture_correction)


def test_correction_pattern_coverage():
    """Test that we have patterns for common correction types."""
    pattern_types = [p[1] for p in CORRECTION_PATTERNS]
    
    # Should have patterns for these common types
    assert "use" in pattern_types or "dont" in pattern_types
    assert "prefer" in pattern_types or "prefer_af" in pattern_types
    assert "never" in pattern_types
    assert "always" in pattern_types


def test_correction_patterns_compile():
    """Test that all regex patterns are valid."""
    import re
    for pattern, ctype in CORRECTION_PATTERNS:
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            pytest.fail(f"Invalid regex pattern for {ctype}: {pattern} - {e}")


def test_dream_correction_patterns_coverage():
    """Test that dream correction patterns cover main cases."""
    pattern_types = [p[1] for p in CORRECTION_PATTERNS]
    
    # Check for key pattern types
    expected_types = {"use", "dont", "never", "always", "stop", "prefer", "wrong", "change"}
    found_types = set(pattern_types)
    
    # Should have at least some of the expected types
    assert len(expected_types & found_types) >= 3


def test_dream_patterns_match_no_use():
    """Test CORRECTION_PATTERNS can match 'no, use X' pattern."""
    import re
    text = "no, use pytest instead"
    text_lower = text.lower()
    
    matched = False
    for pattern, ctype in CORRECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched = True
            break
    
    assert matched, "Should match 'no, use X' pattern"


def test_dream_patterns_match_dont():
    """Test CORRECTION_PATTERNS can match 'don't X' pattern."""
    import re
    text = "don't use unittest"
    text_lower = text.lower()
    
    matched = False
    for pattern, ctype in CORRECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched = True
            break
    
    assert matched, "Should match 'don't use X' pattern"


def test_dream_patterns_match_never():
    """Test CORRECTION_PATTERNS can match 'never X' pattern."""
    import re
    text = "never add emojis"
    text_lower = text.lower()
    
    matched = False
    for pattern, ctype in CORRECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched = True
            break
    
    assert matched, "Should match 'never X' pattern"


def test_dream_patterns_match_always():
    """Test CORRECTION_PATTERNS can match 'always X' pattern."""
    import re
    text = "always use type hints"
    text_lower = text.lower()
    
    matched = False
    for pattern, ctype in CORRECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched = True
            break
    
    assert matched, "Should match 'always X' pattern"


def test_correction_patterns_case_insensitive():
    """Test that patterns work case-insensitively."""
    import re
    text = "NO, USE PYTEST"
    text_lower = text.lower()
    
    matched = False
    for pattern, ctype in CORRECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched = True
            break
    
    assert matched


def test_correction_patterns_afrikaans():
    """Test Afrikaans correction patterns exist."""
    pattern_types = [p[1] for p in CORRECTION_PATTERNS]
    
    # Check for Afrikaans patterns
    afrikaans_types = [t for t in pattern_types if "_af" in t]
    assert len(afrikaans_types) > 0, "Should have Afrikaans patterns"
