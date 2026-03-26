"""Utility functions for text processing and deduplication."""

import re
from difflib import SequenceMatcher
from .config import AUTO_TAG_RULES, SIMILARITY_THRESHOLD


def auto_tag(content, existing_tags=""):
    """Auto-detect tags from content keywords."""
    content_lower = content.lower()
    detected = set()
    for tag, keywords in AUTO_TAG_RULES.items():
        for kw in keywords:
            if kw in content_lower:
                detected.add(tag)
                break
    # Merge with existing
    existing = set(filter(None, existing_tags.split(",")))
    merged = existing | detected
    return ",".join(sorted(merged))


def normalize(text):
    """Normalize text for comparison."""
    return re.sub(r'[^\w\s]', '', text.lower().strip())


def word_set(text):
    """Extract word set from text (words > 2 chars)."""
    return set(w for w in normalize(text).split() if len(w) > 2)


def word_overlap(text_a, text_b):
    """Calculate word-level Jaccard similarity between two texts."""
    if not text_a or not text_b:
        return 0.0
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def similarity(text_a, text_b):
    """Calculate similarity between two texts using Jaccard and SequenceMatcher."""
    new_words = word_set(text_a)
    existing_words = word_set(text_b)

    if not new_words or not existing_words:
        return 0.0

    # Jaccard similarity
    intersection = new_words & existing_words
    union = new_words | existing_words
    jaccard = len(intersection) / len(union) if union else 0

    # Sequence similarity
    seq_score = SequenceMatcher(None, normalize(text_a), normalize(text_b)).ratio()

    # Return max of both
    return max(jaccard, seq_score)


def find_similar(content, category=None, project=None, threshold=None):
    """Find similar memories based on content similarity."""
    from .database import get_db

    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    conn = get_db()
    query = "SELECT id, content, category, project FROM memories WHERE active = 1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if project:
        query += " AND project = ?"
        params.append(project)
    rows = conn.execute(query, params).fetchall()

    new_words = word_set(content)
    similar = []
    for row in rows:
        existing_words = word_set(row["content"])
        if not new_words or not existing_words:
            continue
        intersection = new_words & existing_words
        union = new_words | existing_words
        jaccard = len(intersection) / len(union) if union else 0
        seq_score = SequenceMatcher(None, normalize(content), normalize(row["content"])).ratio()
        score = max(jaccard, seq_score)
        if score >= threshold:
            similar.append((row["id"], row["content"], score, row["category"], row["project"]))
    conn.close()
    return sorted(similar, key=lambda x: -x[2])
