"""Correction capture and detection."""

import sqlite3
import sys
import os
import re
import json
import shutil
import subprocess
import hashlib
import math
from datetime import datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher

# Import from our modular components
from .config import *
from .database import get_db, has_vec_support
from .utils import auto_tag, word_set, normalize, find_similar, word_overlap, similarity
from .fsrs import fsrs_retention, fsrs_new_stability, fsrs_new_difficulty, fsrs_next_interval, fsrs_auto_rating
from .importance import update_importance
from .embedding import embed_and_store, embed_text, semantic_search

# Lazy imports for optional dependencies
try:
    import numpy as np
    import sqlite_vec
except ImportError:
    pass


def detect_correction(text):
    """Check if text contains a correction pattern. Returns dict with details or None."""
    import re
    text_lower = text.lower().strip()
    for pattern, ctype in CORRECTION_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip() if match.lastindex and match.lastindex > 0 else text_lower
            return {
                "type": ctype,
                "raw": text,
                "correction": extracted,
                "full_match": match.group(0)
            }
    return None




def cmd_capture_correction(text):
    """Detect and store corrections from user messages."""
    import re

    # Patterns that indicate a correction
    correction_patterns = [
        # "no, use X" / "no, do X"
        (r"(?i)\bno[,.]?\s+(use|do|try|make|put|set|run|call|add)\s+(.+)", "preference"),
        # "don't X" / "do not X" / "never X"
        (r"(?i)\b(don'?t|do\s+not|never|stop)\s+(use|do|suggest|add|put|make|run|call|include)\s+(.+)", "preference"),
        # "actually it's X" / "actually, X"
        (r"(?i)\bactually[,.]?\s+(.+)", "learning"),
        # "wrong, X" / "that's wrong"
        (r"(?i)\b(wrong|incorrect|that'?s\s+not\s+right)[,.]?\s*(.+)", "learning"),
        # "instead, X" / "rather X"
        (r"(?i)\b(instead|rather)[,.]?\s+(.+)", "preference"),
        # "I prefer X" / "I want X"
        (r"(?i)\bi\s+(prefer|want|need|like)\s+(.+)", "preference"),
        # "X not Y" pattern
        (r"(?i)\buse\s+(\S+)\s+not\s+(\S+)", "preference"),
        # "always X" / "make sure to X"
        (r"(?i)\b(always|make\s+sure\s+to|remember\s+to)\s+(.+)", "preference"),
    ]

    matches = []
    for pattern, category in correction_patterns:
        m = re.search(pattern, text)
        if m:
            matches.append((category, m.group(0).strip()))

    if not matches:
        print("No corrections detected.")
        return

    conn = get_db()

    for category, correction in matches:
        # Check for duplicates using search
        existing = conn.execute(
            "SELECT id, content FROM memories WHERE active = 1 AND content LIKE ? LIMIT 1",
            (f"%{correction[:50]}%",)
        ).fetchone()

        if existing:
            print(f"  Similar correction already stored: #{existing['id']}")
            continue

        # Store as a learning/preference
        content = f"CORRECTION: {correction}"
        mem_id = add_memory(category, content, tags="correction,user-feedback")
        print(f"  ✅ Captured [{category}]: {correction} → #{mem_id}")

    conn.close()

    print(f"\n{len(matches)} correction(s) processed.")



