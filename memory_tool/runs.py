"""Run tracking for agent tasks."""

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


def start_run(task, agent="claw", project=None, tags=""):
    """Start a new run. Returns run ID."""
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO runs (task, agent, project, tags) VALUES (?, ?, ?, ?)",
        (task, agent, project, tags)
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id




def add_run_step(run_id, step_description):
    """Append a step to the run's steps array."""
    conn = get_db()
    
    # Get current steps
    row = conn.execute("SELECT steps FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        conn.close()
        return False
        
    try:
        steps = json.loads(row['steps'])
    except (json.JSONDecodeError, TypeError):
        steps = []
    
    # Append new step
    steps.append(step_description)
    
    # Update in DB
    conn.execute(
        "UPDATE runs SET steps = ? WHERE id = ?",
        (json.dumps(steps), run_id)
    )
    conn.commit()
    conn.close()
    return True




def complete_run(run_id, outcome):
    """Mark a run as completed."""
    conn = get_db()
    conn.execute(
        "UPDATE runs SET status = 'completed', completed_at = datetime('now'), outcome = ? WHERE id = ?",
        (outcome, run_id)
    )
    conn.commit()
    conn.close()




def fail_run(run_id, reason):
    """Mark a run as failed."""
    conn = get_db()
    conn.execute(
        "UPDATE runs SET status = 'failed', completed_at = datetime('now'), outcome = ? WHERE id = ?",
        (reason, run_id)
    )
    conn.commit()
    conn.close()




def cancel_run(run_id):
    """Mark a run as cancelled."""
    conn = get_db()
    conn.execute(
        "UPDATE runs SET status = 'cancelled', completed_at = datetime('now') WHERE id = ?",
        (run_id,)
    )
    conn.commit()
    conn.close()




def list_runs(status=None, project=None, limit=10):
    """List runs with optional filters."""
    conn = get_db()
    
    query = "SELECT * FROM runs WHERE 1=1"
    params = []
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    if project:
        query += " AND project = ?"
        params.append(project)
    
    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows




def show_run(run_id):
    """Show detailed information for a run."""
    conn = get_db()
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    return row




def format_duration(start_time, end_time=None):
    """Format duration in human-readable format."""
    if not start_time:
        return "unknown"
    
    try:
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        else:
            end_dt = datetime.now()
        
        delta = end_dt - start_dt
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    except (ValueError, TypeError):
        return "unknown"


# ── Phase 4: OpenClaw Bridge (Memory Sync) ──
# (Paths imported from config.py)



