"""Session snapshot and project detection."""

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


def save_snapshot(summary, project=None, files_touched="", memories_added="", memories_updated=""):
    conn = get_db()
    conn.execute(
        "INSERT INTO session_snapshots (summary, project, files_touched, memories_added, memories_updated) VALUES (?, ?, ?, ?, ?)",
        (summary, project, files_touched, memories_added, memories_updated)
    )
    conn.commit()
    conn.close()
    print(f"Session snapshot saved.")




def get_snapshots(limit=5):
    conn = get_db()
    rows = conn.execute("SELECT * FROM session_snapshots ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows


# ── Upgrade #2: Auto-Snapshot from git/file changes ──



def auto_snapshot():
    """Auto-generate session snapshot from recent git activity and file changes."""
    from .config import PROJECT_PATHS
    parts = []
    projects_touched = set()

    # Check git repos for recent changes (using configured PROJECT_PATHS)
    repo_projects = [(path, name) for path, name in PROJECT_PATHS.items()]
    if not repo_projects:
        # No projects configured, skip auto-snapshot
        return

    for repo_path, project_name in repo_projects:
        if not Path(repo_path).exists():
            continue
        try:
            # Files modified in last 2 hours
            result = subprocess.run(
                ["find", repo_path, "-maxdepth", "4", "-name", "*.ts", "-o", "-name", "*.js",
                 "-o", "-name", "*.py", "-o", "-name", "*.tsx", "-o", "-name", "*.jsx",
                 "-newer", str(DB_PATH)],
                capture_output=True, text=True, timeout=5, cwd=repo_path
            )
            changed = [f for f in result.stdout.strip().split("\n") if f and "node_modules" not in f and ".next" not in f]
            if changed:
                projects_touched.add(project_name)
                # Summarize
                dirs = set()
                for f in changed[:20]:
                    rel = os.path.relpath(f, repo_path)
                    parts_list = rel.split("/")
                    if len(parts_list) > 1:
                        dirs.add(parts_list[0] + "/" + parts_list[1] if len(parts_list) > 2 else parts_list[0])
                parts.append(f"{project_name}: modified {len(changed)} files in {', '.join(sorted(dirs)[:5])}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Check for git commits in last 2 hours
    for repo_path, project_name in repo_projects:
        if not Path(repo_path / Path(".git")).exists() if isinstance(repo_path, Path) else not Path(repo_path + "/.git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--since=2 hours ago", "--no-merges"],
                capture_output=True, text=True, timeout=5, cwd=repo_path
            )
            commits = result.stdout.strip().split("\n")
            commits = [c for c in commits if c]
            if commits:
                parts.append(f"{project_name}: {len(commits)} commit(s) - {commits[0]}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if not parts:
        # Check memory DB itself for recent additions
        conn = get_db()
        recent = conn.execute("""
            SELECT COUNT(*) as c FROM memories
            WHERE created_at > datetime('now', '-2 hours')
        """).fetchone()["c"]
        conn.close()
        if recent:
            parts.append(f"Added {recent} new memories")
        else:
            print("No recent activity detected for auto-snapshot.")
            return

    summary = "; ".join(parts)
    project = list(projects_touched)[0] if len(projects_touched) == 1 else None
    save_snapshot(summary, project)

    # Auto-consolidate during session end (like REM sleep)
    try:
        conn = get_db()
        results = consolidate_memories(conn)
        conn.close()
        if any(results.values()):
            print(f"  Consolidated: {sum(results.values())} changes")
    except Exception as e:
        pass  # Silently continue if consolidation fails

    export_memory_md(project)


# ── Upgrade #1: Error Capture ──



def log_error(command, error_output, project=None):
    """Log a failed command as an error memory."""
    # Truncate long errors
    error_clean = error_output.strip()[:300]
    # Extract key error message (last meaningful line)
    lines = [l.strip() for l in error_clean.split("\n") if l.strip()]
    key_error = lines[-1] if lines else error_clean

    content = f"Command `{command[:100]}` failed: {key_error}"

    # Auto-detect project from command
    if not project:
        for path, proj in PROJECT_PATHS.items():
            if path in command:
                project = proj
                break

    # Check if we already logged this exact error
    similar = find_similar(content, "error", project, threshold=0.75)
    if similar and similar[0][2] > 0.85:
        # Already known, just touch it
        conn = get_db()
        touch_memory(conn, similar[0][0])
        conn.commit()
        conn.close()
        print(f"Known error (memory #{similar[0][0]}), access count updated.")
        return similar[0][0]

    return add_memory("error", content, project=project, source="auto-hook", skip_dedup=True)


# ── Upgrade #4: Import from Session Markdown ──



def import_session_md(filepath):
    """Import memories from a session summary markdown file."""
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return

    text = path.read_text()
    imported = 0

    # Extract project name from title
    project = None
    if "WhatsAuction" in text:
        project = "WhatsAuction"
    elif "FlashVault" in text or "VPN" in text:
        project = "FlashVault"

    # Parse sections
    current_section = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            current_section = line[2:].strip().lower()
            continue

        if not line or line.startswith("_") or line.startswith("```"):
            continue

        # Map sections to categories
        category = None
        if "error" in current_section:
            category = "error"
        elif "learning" in current_section:
            category = "learning"
        elif "workflow" in current_section:
            category = "workflow"
        elif "codebase" in current_section or "system" in current_section or "documentation" in current_section:
            category = "architecture"
        elif "key result" in current_section:
            category = "learning"

        if category and len(line) > 20 and line.startswith("- "):
            content = line[2:].strip()
            # Skip items that look like headers or formatting
            if content.startswith("**") and content.endswith("**"):
                continue
            result = add_memory(category, content, project=project, source="import", skip_dedup=False)
            if result:
                imported += 1

    print(f"Imported {imported} memories from {filepath}")


# ── Project Detection ──



def detect_project(cwd=None):
    if cwd is None:
        cwd = os.getcwd()
    for path, project in PROJECT_PATHS.items():
        if cwd.startswith(path):
            return project
    return None


# ── Upgrade #7: Backup & Restore ──


