"""User Model Layer — builds structured user profiles from memory patterns.

Honcho-style user modeling that mines memories for:
- Identity: role, expertise, projects
- Preferences: communication style, code style, tools
- Patterns: recurring workflows, common requests
- Corrections: things the AI got wrong
- Dialect: shorthand/jargon the user uses
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
from .database import get_db
from .config import get_logger

logger = get_logger(__name__)

# Default output path
USER_MODEL_PATH = Path.home() / ".claude/projects/-root/memory/user-model.md"


def scan_user_memories(db: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    """Scan memories for user-modeling signals.

    Returns categorized memories grouped by user model section:
    - identity: role, expertise, projects
    - preferences: style, tools, workflows
    - patterns: recurring behaviors
    - corrections: AI errors and fixes
    - dialect: shorthand terms and jargon

    Args:
        db: Database connection

    Returns:
        Dict with keys: identity, preferences, patterns, corrections, dialect
    """
    result = {
        'identity': [],
        'preferences': [],
        'patterns': [],
        'corrections': [],
        'dialect': []
    }

    # Identity signals: user profile, role, expertise
    identity_rows = db.execute("""
        SELECT id, content, category, project, tags, created_at
        FROM memories
        WHERE active = 1
        AND (
            category IN ('profile', 'expertise', 'role', 'context')
            OR content LIKE '%I am%'
            OR content LIKE '%I work%'
            OR content LIKE '%my role%'
            OR content LIKE '%my email%'
            OR tags LIKE '%userEmail%'
            OR tags LIKE '%identity%'
            OR tags LIKE '%profile%'
        )
        ORDER BY created_at DESC
        LIMIT 50
    """).fetchall()

    for row in identity_rows:
        result['identity'].append({
            'id': row['id'],
            'content': row['content'],
            'category': row['category'],
            'project': row['project'],
            'tags': row['tags'],
            'created_at': row['created_at']
        })

    # Preferences: user likes/dislikes, tool preferences, communication style
    pref_rows = db.execute("""
        SELECT id, content, category, project, tags, created_at
        FROM memories
        WHERE active = 1
        AND (
            category IN ('preference', 'style', 'workflow')
            OR content LIKE '%prefer%'
            OR content LIKE '%always use%'
            OR content LIKE '%never use%'
            OR content LIKE '%I like%'
            OR content LIKE '%I hate%'
            OR content LIKE '%instead of%'
            OR tags LIKE '%preference%'
            OR tags LIKE '%style%'
        )
        ORDER BY created_at DESC
        LIMIT 50
    """).fetchall()

    for row in pref_rows:
        result['preferences'].append({
            'id': row['id'],
            'content': row['content'],
            'category': row['category'],
            'project': row['project'],
            'tags': row['tags'],
            'created_at': row['created_at']
        })

    # Patterns: recurring workflows, common requests, behavioral patterns
    pattern_rows = db.execute("""
        SELECT content, category, project, tags, COUNT(*) as occurrences
        FROM memories
        WHERE active = 1
        AND category IN ('workflow', 'pattern', 'routine', 'decision')
        GROUP BY content
        HAVING occurrences >= 2
        ORDER BY occurrences DESC
        LIMIT 30
    """).fetchall()

    for row in pattern_rows:
        result['patterns'].append({
            'content': row['content'],
            'category': row['category'],
            'project': row['project'],
            'tags': row['tags'],
            'occurrences': row['occurrences']
        })

    # Corrections: things AI got wrong + correct behavior
    # Check if corrections table exists
    try:
        correction_rows = db.execute("""
            SELECT c.id, c.raw_text, c.correction, c.category,
                   c.status, c.created_at, c.applied_at, c.memory_id
            FROM corrections c
            WHERE c.status = 'applied'
            ORDER BY c.created_at DESC
            LIMIT 50
        """).fetchall()

        for row in correction_rows:
            result['corrections'].append({
                'id': row['id'],
                'original': row['raw_text'],
                'corrected': row['correction'],
                'category': row['category'],
                'created_at': row['created_at']
            })
    except sqlite3.OperationalError:
        # Corrections table doesn't exist
        pass

    # Also get feedback memories (user corrections in feedback form)
    feedback_rows = db.execute("""
        SELECT id, content, created_at
        FROM memories
        WHERE active = 1
        AND (
            category = 'feedback'
            OR content LIKE '%actually%'
            OR content LIKE '%correction:%'
            OR content LIKE '%fix:%'
            OR content LIKE '%wrong%'
        )
        ORDER BY created_at DESC
        LIMIT 30
    """).fetchall()

    for row in feedback_rows:
        # Add feedback-based corrections
        result['corrections'].append({
            'id': row['id'],
            'content': row['content'],
            'created_at': row['created_at']
        })

    # Dialect: shorthand terms, jargon, abbreviations user uses
    # Look for short terms, acronyms, specialized vocabulary
    dialect_rows = db.execute("""
        SELECT DISTINCT tags, project
        FROM memories
        WHERE active = 1
        AND tags IS NOT NULL
        AND tags != ''
        ORDER BY created_at DESC
        LIMIT 100
    """).fetchall()

    # Extract unique tags (these represent user's vocabulary)
    tag_set = set()
    for row in dialect_rows:
        if row['tags']:
            for tag in row['tags'].split(','):
                tag = tag.strip()
                if tag:
                    tag_set.add(tag)

    # Also look for project names (user's domain)
    project_rows = db.execute("""
        SELECT DISTINCT project, COUNT(*) as count
        FROM memories
        WHERE active = 1
        AND project IS NOT NULL
        GROUP BY project
        ORDER BY count DESC
        LIMIT 20
    """).fetchall()

    result['dialect'] = {
        'tags': sorted(tag_set),
        'projects': [{'name': r['project'], 'count': r['count']} for r in project_rows]
    }

    return result


def build_user_model_md(user_data: Dict[str, Any]) -> str:
    """Generate markdown document from user model data.

    Args:
        user_data: Dict with sections (identity, preferences, patterns, corrections, dialect)

    Returns:
        Formatted markdown string
    """
    lines = []

    # Header
    lines.append("# User Model")
    lines.append("")
    lines.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("This document is auto-generated from memory patterns. It captures:")
    lines.append("- **Identity**: Who you are, your role, expertise")
    lines.append("- **Preferences**: How you like to work, tools you prefer")
    lines.append("- **Patterns**: Your recurring workflows and behaviors")
    lines.append("- **Corrections**: Things I've gotten wrong + correct behavior")
    lines.append("- **Dialect**: Your shorthand, jargon, and vocabulary")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Identity section
    lines.append("## Identity")
    lines.append("")
    if user_data['identity']:
        for item in user_data['identity'][:15]:  # Limit to top 15
            lines.append(f"- {item['content']}")
            if item['project']:
                lines.append(f"  *Project: {item['project']}*")
    else:
        lines.append("*No identity information captured yet.*")
    lines.append("")

    # Preferences section
    lines.append("## Preferences")
    lines.append("")
    if user_data['preferences']:
        for item in user_data['preferences'][:15]:
            lines.append(f"- {item['content']}")
    else:
        lines.append("*No preferences captured yet.*")
    lines.append("")

    # Patterns section
    lines.append("## Patterns")
    lines.append("")
    lines.append("Recurring workflows and behaviors:")
    lines.append("")
    if user_data['patterns']:
        for item in user_data['patterns'][:15]:
            lines.append(f"- **{item['content']}** ({item['occurrences']}x)")
            if item['project']:
                lines.append(f"  *Project: {item['project']}*")
    else:
        lines.append("*No recurring patterns detected yet.*")
    lines.append("")

    # Corrections section
    lines.append("## Corrections")
    lines.append("")
    lines.append("Things I've gotten wrong + correct behavior:")
    lines.append("")
    if user_data['corrections']:
        for item in user_data['corrections'][:15]:
            if 'original' in item and 'corrected' in item:
                lines.append(f"- **Original**: {item['original'][:100]}...")
                lines.append(f"  **Corrected**: {item['corrected'][:100]}...")
            elif 'content' in item:
                lines.append(f"- {item['content']}")
    else:
        lines.append("*No corrections logged yet.*")
    lines.append("")

    # Dialect section
    lines.append("## Dialect")
    lines.append("")
    lines.append("Your shorthand, jargon, and vocabulary:")
    lines.append("")

    if user_data['dialect']['projects']:
        lines.append("**Active Projects:**")
        for proj in user_data['dialect']['projects'][:10]:
            lines.append(f"- {proj['name']} ({proj['count']} memories)")
        lines.append("")

    if user_data['dialect']['tags']:
        lines.append("**Common Tags:**")
        # Show top 30 tags
        lines.append(", ".join(user_data['dialect']['tags'][:30]))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Run `memory-tool user-model --update` to refresh with recent session data.*")

    return "\n".join(lines)


def generate_user_model(db: sqlite3.Connection, output_path: Optional[Path] = None, dry_run: bool = False) -> str:
    """Generate user model document from existing memories.

    Args:
        db: Database connection
        output_path: Where to write the file (default: ~/.claude/projects/-root/memory/user-model.md)
        dry_run: If True, return markdown without writing file

    Returns:
        Generated markdown content
    """
    if output_path is None:
        output_path = USER_MODEL_PATH

    logger.info("Scanning memories for user model signals...")
    user_data = scan_user_memories(db)

    logger.info("Building user model document...")
    markdown = build_user_model_md(user_data)

    if dry_run:
        logger.info("Dry run - not writing file")
        return markdown

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    output_path.write_text(markdown)
    logger.info(f"User model written to {output_path}")

    return markdown


def update_user_model_with_transcript(
    db: sqlite3.Connection,
    transcript: Optional[str] = None,
    days: int = 7,
    output_path: Optional[Path] = None,
    dry_run: bool = False
) -> str:
    """Dialectic update mode: analyze recent session data via claude CLI.

    This mode:
    1. Gathers recent memories (last N days) or uses provided transcript
    2. Uses `claude` CLI to extract user model signals
    3. Updates the user-model.md with new insights
    4. Saves extracted signals as memories (category='user_model')

    Args:
        db: Database connection
        transcript: Optional session transcript to analyze
        days: How many days back to scan for memories (default: 7)
        output_path: Where to write user-model.md
        dry_run: If True, don't write files or add memories

    Returns:
        Generated markdown content
    """
    import subprocess
    import re
    import shutil

    if output_path is None:
        output_path = USER_MODEL_PATH

    if not shutil.which("claude"):
        logger.error("claude CLI not found. Install Claude Code or ensure it is in PATH.")
        return ""

    # Build context from recent memories if no transcript provided
    if transcript is None:
        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        recent = db.execute("""
            SELECT content, category, project, tags, created_at
            FROM memories
            WHERE active = 1
            AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT 100
        """, (since_date,)).fetchall()

        transcript_lines = [f"Recent memories (last {days} days):", ""]
        for mem in recent:
            transcript_lines.append(f"[{mem['category']}] {mem['content']}")
            if mem['project']:
                transcript_lines.append(f"  Project: {mem['project']}")
        transcript = "\n".join(transcript_lines)

    prompt = f"""Analyze this session data and extract user model signals.

Session Data:
{transcript}

Extract insights in these categories:

1. **Identity** - Who is this user? What's their role, expertise, email?
2. **Preferences** - Communication style, code style, tools they prefer/avoid
3. **Patterns** - Recurring workflows, common requests, behavioral patterns
4. **Corrections** - Times the AI got something wrong + correct behavior
5. **Dialect** - Shorthand terms, jargon, abbreviations they use

Respond with ONLY a JSON object, no markdown fences:
{{
  "identity": ["insight 1", "insight 2"],
  "preferences": ["preference 1"],
  "patterns": ["pattern 1"],
  "corrections": ["correction 1"],
  "dialect": ["term 1"]
}}

Focus on actionable, specific insights. Skip generic observations."""

    logger.info("Calling claude CLI to analyze session data...")

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            logger.error(f"claude CLI error: {result.stderr}")
            return ""

        content = result.stdout.strip()

        # Parse JSON — strip markdown fences if present
        json_match = re.search(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.error("Could not extract JSON from claude CLI response")
                return ""

        insights = json.loads(json_str)

    except subprocess.TimeoutExpired:
        logger.error("claude CLI timed out after 120s")
        return ""
    except Exception as e:
        logger.error(f"claude CLI error: {e}")
        return ""

    # Save insights as memories (category='user_model')
    if not dry_run:
        from .memory_ops import add_memory

        for category, items in insights.items():
            for item in items:
                if item:  # Skip empty strings
                    add_memory(
                        category='user_model',
                        content=item,
                        tags=f'{category},user_modeling',
                        source='user_model_update'
                    )

        logger.info(f"Added {sum(len(v) for v in insights.values())} new user model memories")

    # Now regenerate the user model document with all data
    user_data = scan_user_memories(db)
    markdown = build_user_model_md(user_data)

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
        logger.info(f"Updated user model written to {output_path}")

    return markdown


def cmd_user_model(args: List[str]) -> None:
    """CLI command handler for user-model.

    Usage:
        memory-tool user-model                    # Generate from existing memories
        memory-tool user-model --update           # Dialectic mode with Claude API
        memory-tool user-model --update --days 14 # Analyze last 14 days
        memory-tool user-model --dry-run          # Print without writing
        memory-tool user-model --output /path/to/file.md
    """
    db = get_db()

    # Parse flags
    update_mode = '--update' in args
    dry_run = '--dry-run' in args

    days = 7
    if '--days' in args:
        days_idx = args.index('--days')
        if days_idx + 1 < len(args):
            try:
                days = int(args[days_idx + 1])
            except ValueError:
                logger.error("--days requires an integer value")
                db.close()
                return

    output_path = None
    if '--output' in args:
        output_idx = args.index('--output')
        if output_idx + 1 < len(args):
            output_path = Path(args[output_idx + 1])

    if update_mode:
        # Dialectic mode with Claude API
        markdown = update_user_model_with_transcript(
            db=db,
            days=days,
            output_path=output_path,
            dry_run=dry_run
        )
    else:
        # Simple mode: scan existing memories
        markdown = generate_user_model(
            db=db,
            output_path=output_path,
            dry_run=dry_run
        )

    if dry_run:
        print(markdown)
    else:
        path = output_path or USER_MODEL_PATH
        print(f"User model generated: {path}")
        print(f"\nPreview:\n")
        # Print first 30 lines
        lines = markdown.split('\n')
        for line in lines[:30]:
            print(line)
        if len(lines) > 30:
            print(f"\n... ({len(lines) - 30} more lines)")

    db.close()
