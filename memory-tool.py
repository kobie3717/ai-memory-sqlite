#!/usr/bin/env python3
"""
AI Memory SQLite - Persistent Memory System v5
SQLite-backed memory with hybrid search (FTS5 + semantic embeddings + RRF fusion),
graph intelligence (entities, relationships, facts), smart deduplication,
auto-tagging, decay/expiry, session snapshots, and cross-tool sync.

Usage:
  memory-tool add <category> <content> [--tags t1,t2] [--project X] [--priority N] [--related ID] [--expires YYYY-MM-DD] [--key topic-key] [--derived-from ID1,ID2] [--citations "URL1;path2"] [--reasoning "why"]
  memory-tool search <query> [--full] [--semantic] [--keyword]  # Hybrid search (default), --semantic for semantic-only, --keyword for FTS-only
  memory-tool get <id>                          # Show full detail for single memory
  memory-tool list [--category X] [--project X] [--tag X] [--stale] [--expired]
  memory-tool update <id> <content>
  memory-tool delete <id>
  memory-tool tag <id> <tags>
  memory-tool relate <id1> <id2> [type]         # Link related memories
  memory-tool conflicts                         # Find potential duplicate memories
  memory-tool merge <id1> <id2>                 # Merge two similar memories
  memory-tool supersede <old_id> <new_id>       # Mark old memory as superseded by new
  memory-tool pending                           # Show pending/todo items
  memory-tool projects                          # Project summary
  memory-tool topics                            # Generate topic .md files per project
  memory-tool export [--project X]              # Regenerate MEMORY.md (smart context)
  memory-tool stats                             # Full statistics (includes vector index & graph)
  memory-tool next                              # Suggest next actions based on current memory state
  memory-tool stale                             # Review stale memories
  memory-tool decay                             # Flag stale, deprioritize, expire
  memory-tool reindex                           # Bulk-embed all active memories for vector search
  memory-tool snapshot <summary> [--project X]  # Save session snapshot
  memory-tool auto-snapshot                     # Auto-generate snapshot from git/file changes
  memory-tool snapshots [--limit N]             # View recent snapshots
  memory-tool detect-project                    # Auto-detect project from cwd
  memory-tool gc [days]                         # Garbage collect old inactive memories
  memory-tool log-error <command> <error>       # Log a failed command as error memory
  memory-tool import-md <file>                  # Import memories from session summary markdown
  memory-tool backup                            # Backup database
  memory-tool restore <file>                    # Restore database from backup

Graph Intelligence (Phase 3):
  memory-tool graph                             # Show graph summary
  memory-tool graph add <type> <name> [summary] # Add entity (types: person/project/org/feature/concept/tool/service)
  memory-tool graph rel <from> <rel_type> <to> [note]  # Add relationship (types: knows/works_on/owns/depends_on/built_by/uses/blocks/related_to)
  memory-tool graph fact <entity> <key> <value> # Set fact on entity
  memory-tool graph get <name>                  # Show entity with facts & relationships
  memory-tool graph list [type]                 # List entities
  memory-tool graph delete <name>               # Delete entity
  memory-tool graph spread <name> [depth]       # Spreading activation (default depth=2)
  memory-tool graph link <memory_id> <entity>   # Link memory to entity
  memory-tool graph auto-link                   # Auto-link all memories to entities
  memory-tool graph import-openclaw             # Import from OpenClaw graph DB
  memory-tool graph stats                       # Graph statistics

OpenClaw Bridge (Phase 4):
  memory-tool sync                              # Bidirectional sync (to + from OpenClaw)
  memory-tool sync-to                           # Export only (Claude Code → OpenClaw)
  memory-tool sync-from                         # Import only (OpenClaw → Claude Code)

Run Tracking (Phase 5):
  memory-tool run start "task description" [--agent claw|claude] [--project X] [--tags x,y]
  memory-tool run step <id> "step description"
  memory-tool run complete <id> "outcome summary"
  memory-tool run fail <id> "reason"
  memory-tool run list [--status running|completed|failed] [--project X] [--limit 10]
  memory-tool run show <id>                     # Show full run detail including all steps
  memory-tool run cancel <id>

Categories: project, decision, preference, error, learning, pending, architecture, workflow, contact
Priority: 0 (low) to 10 (high). Auto-adjusts based on access frequency.
Vector search: Requires sqlite-vec, onnxruntime, tokenizers, numpy. Model: all-MiniLM-L6-v2 (384-dim).
"""

import sqlite3
import sys
import os
import re
import json
import shutil
import subprocess
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher

# Lazy imports for vector search (optional dependencies)
_EMBEDDING_MODEL = None
_VEC_AVAILABLE = None

try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    import sqlite_vec
    _VEC_LIBS_AVAILABLE = True
except ImportError:
    _VEC_LIBS_AVAILABLE = False

MEMORY_DIR = Path(__file__).parent
DB_PATH = MEMORY_DIR / "memories.db"
MEMORY_MD_PATH = MEMORY_DIR / "MEMORY.md"
TOPICS_DIR = MEMORY_DIR / "topics"
BACKUP_DIR = Path(os.getenv("MEMORY_BACKUP_DIR", str(MEMORY_DIR / "backups")))
MAX_MEMORY_MD_BYTES = 5120  # 5KB hard cap

# Staleness thresholds (days)
STALE_PENDING_DAYS = 30
STALE_GENERAL_DAYS = 90
DEPRIORITIZE_DAYS = 60

# Dedup similarity threshold (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.65

# Vector search configuration
MODEL_DIR = Path.home() / ".cache/models/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RRF_K = 60  # Reciprocal Rank Fusion constant

# Project detection paths (load from config file or auto-detect from git)
PROJECT_PATHS = {}

# Auto-tag keywords (generic defaults, extensible via config)
AUTO_TAG_RULES = {
    "database": ["postgresql", "mysql", "sqlite", "mongodb", "prisma", "sequelize", "migration", "schema"],
    "auth": ["jwt", "login", "password", "token", "auth", "oauth", "bcrypt", "session"],
    "nginx": ["nginx", "reverse proxy", "ssl", "certbot", "letsencrypt", "apache", "caddy"],
    "docker": ["docker", "container", "dockerfile", "compose", "kubernetes", "k8s"],
    "react": ["react", "vite", "webpack", "tailwind", "frontend", "tsx", "jsx", "component"],
    "api": ["endpoint", "route", "controller", "middleware", "rest", "graphql", "express", "fastapi"],
    "git": ["git", "commit", "branch", "merge", "rebase", "pull request", "github", "gitlab"],
    "test": ["test", "jest", "pytest", "unittest", "vitest", "cypress", "playwright"],
    "deploy": ["deploy", "deployment", "ci/cd", "pipeline", "production", "staging"],
    "security": ["security", "vulnerability", "cve", "encryption", "sanitize", "xss", "csrf"],
}


# ── Vector Search Functions ──

def has_vec_support():
    """Check if vector search dependencies are available."""
    global _VEC_AVAILABLE
    if _VEC_AVAILABLE is not None:
        return _VEC_AVAILABLE

    if not _VEC_LIBS_AVAILABLE:
        _VEC_AVAILABLE = False
        return False

    # Check if model files exist
    if not MODEL_DIR.exists():
        _VEC_AVAILABLE = False
        return False

    required_files = [
        MODEL_DIR / "tokenizer.json",
        MODEL_DIR / "onnx" / "model.onnx"
    ]
    _VEC_AVAILABLE = all(f.exists() for f in required_files)
    return _VEC_AVAILABLE


def get_embedding_model():
    """Lazy-load the embedding model (singleton)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    if not has_vec_support():
        return None

    try:
        tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        tokenizer.enable_truncation(max_length=256)

        session = ort.InferenceSession(
            str(MODEL_DIR / "onnx" / "model.onnx"),
            providers=['CPUExecutionProvider']
        )

        _EMBEDDING_MODEL = (tokenizer, session)
        return _EMBEDDING_MODEL
    except Exception as e:
        print(f"Warning: Failed to load embedding model: {e}", file=sys.stderr)
        return None


def embed_text(text):
    """Generate embedding for a single text string. Returns bytes for sqlite-vec."""
    model = get_embedding_model()
    if model is None:
        return None

    tokenizer, session = model

    try:
        # Encode text
        encoding = tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Run inference
        outputs = session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        })

        # Mean pooling
        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
        embeddings = summed / counts

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / norms).astype(np.float32)

        # Return as bytes
        return embeddings[0].tobytes()
    except Exception as e:
        print(f"Warning: Embedding failed: {e}", file=sys.stderr)
        return None


def embed_texts_batch(texts):
    """Generate embeddings for multiple texts. Returns list of bytes."""
    model = get_embedding_model()
    if model is None:
        return [None] * len(texts)

    tokenizer, session = model

    try:
        # Encode batch
        encodings = tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Run inference
        outputs = session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        })

        # Mean pooling
        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
        embeddings = summed / counts

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / norms).astype(np.float32)

        # Return as list of bytes
        return [emb.tobytes() for emb in embeddings]
    except Exception as e:
        print(f"Warning: Batch embedding failed: {e}", file=sys.stderr)
        return [None] * len(texts)


def embed_and_store(conn, mem_id, content):
    """Generate embedding for content and store in vector table."""
    if not has_vec_support():
        return

    embedding = embed_text(content)
    if embedding is None:
        return

    try:
        # Insert or replace embedding (rowid must match memory id)
        conn.execute(
            "INSERT OR REPLACE INTO memory_vec(rowid, embedding) VALUES (?, ?)",
            (mem_id, embedding)
        )
    except Exception as e:
        # Silently fail if vec table doesn't exist
        pass


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec extension if available
    if has_vec_support():
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except:
            pass

    return conn


def init_db():
    conn = get_db()

    # Add columns if upgrading (must run BEFORE triggers reference them)
    for col, coltype, default in [
        ("accessed_at", "TEXT", "NULL"),
        ("access_count", "INTEGER", "0"),
        ("stale", "INTEGER", "0"),
        ("expires_at", "TEXT", "NULL"),
        ("source", "TEXT", "'manual'"),
        ("topic_key", "TEXT", "NULL"),
        ("revision_count", "INTEGER", "1"),
        ("derived_from", "TEXT", "NULL"),
        ("citations", "TEXT", "NULL"),
        ("reasoning", "TEXT", "NULL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {coltype} DEFAULT {default}")
        except sqlite3.OperationalError:
            pass
    conn.commit()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            project TEXT DEFAULT NULL,
            tags TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            accessed_at TEXT DEFAULT NULL,
            access_count INTEGER DEFAULT 0,
            priority INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            stale INTEGER DEFAULT 0,
            expires_at TEXT DEFAULT NULL,
            source TEXT DEFAULT 'manual',
            topic_key TEXT DEFAULT NULL,
            revision_count INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS memory_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relation_type TEXT DEFAULT 'related',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (source_id) REFERENCES memories(id),
            FOREIGN KEY (target_id) REFERENCES memories(id),
            UNIQUE(source_id, target_id)
        );

        CREATE TABLE IF NOT EXISTS session_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT NOT NULL,
            project TEXT DEFAULT NULL,
            files_touched TEXT DEFAULT '',
            memories_added TEXT DEFAULT '',
            memories_updated TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_category ON memories(category);
        CREATE INDEX IF NOT EXISTS idx_project ON memories(project);
        CREATE INDEX IF NOT EXISTS idx_active ON memories(active);
        CREATE INDEX IF NOT EXISTS idx_stale ON memories(stale);
        CREATE INDEX IF NOT EXISTS idx_accessed ON memories(accessed_at);
        CREATE INDEX IF NOT EXISTS idx_expires ON memories(expires_at);
        CREATE INDEX IF NOT EXISTS idx_source ON memories(source);
        CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_id);
        CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_topic_key ON memories(topic_key) WHERE topic_key IS NOT NULL;

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content, tags, project, category,
            content='memories',
            content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, tags, project, category)
            VALUES (new.id, new.content, new.tags, new.project, new.category);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags, project, category)
            VALUES ('delete', old.id, old.content, old.tags, old.project, old.category);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags, project, category)
            VALUES ('delete', old.id, old.content, old.tags, old.project, old.category);
            INSERT INTO memories_fts(rowid, content, tags, project, category)
            VALUES (new.id, new.content, new.tags, new.project, new.category);
        END;
    """)

    # Create vector table if sqlite-vec is available
    if has_vec_support():
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec
                USING vec0(embedding float[{EMBEDDING_DIM}])
            """)
        except Exception as e:
            # Silently fail if vec is not available
            pass

    # Phase 3: Graph Intelligence tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS graph_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('person','project','org','feature','concept','tool','service')),
            summary TEXT DEFAULT '',
            importance INTEGER DEFAULT 3,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS graph_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_entity_id INTEGER NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
            to_entity_id INTEGER NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL CHECK(relation_type IN ('knows','works_on','owns','depends_on','built_by','uses','blocks','related_to')),
            note TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(from_entity_id, to_entity_id, relation_type)
        );

        CREATE TABLE IF NOT EXISTS graph_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(entity_id, key)
        );

        CREATE TABLE IF NOT EXISTS graph_fact_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            old_value TEXT NOT NULL,
            new_value TEXT NOT NULL,
            changed_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS memory_entity_links (
            memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            entity_id INTEGER NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (memory_id, entity_id)
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            agent TEXT DEFAULT 'claw',
            status TEXT DEFAULT 'running',
            started_at DATETIME DEFAULT (datetime('now')),
            completed_at DATETIME,
            steps TEXT DEFAULT '[]',
            outcome TEXT,
            project TEXT,
            tags TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_graph_entity_type ON graph_entities(type);
        CREATE INDEX IF NOT EXISTS idx_graph_entity_name ON graph_entities(name);
        CREATE INDEX IF NOT EXISTS idx_graph_rel_from ON graph_relationships(from_entity_id);
        CREATE INDEX IF NOT EXISTS idx_graph_rel_to ON graph_relationships(to_entity_id);
        CREATE INDEX IF NOT EXISTS idx_graph_facts_entity ON graph_facts(entity_id);
        CREATE INDEX IF NOT EXISTS idx_mem_entity_memory ON memory_entity_links(memory_id);
        CREATE INDEX IF NOT EXISTS idx_mem_entity_entity ON memory_entity_links(entity_id);
        CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
        CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project);
        CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent);
    """)

    conn.commit()
    conn.close()


# ── Auto-Tagging (Upgrade #5) ──

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


# ── Deduplication ──

def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())


def word_set(text):
    return set(w for w in normalize(text).split() if len(w) > 2)


def find_similar(content, category=None, project=None, threshold=SIMILARITY_THRESHOLD):
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
    conn.close()

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
    return sorted(similar, key=lambda x: -x[2])


# ── Access Tracking & Priority Auto-Adjust ──

def touch_memory(conn, mem_id):
    conn.execute("""
        UPDATE memories SET
            accessed_at = datetime('now'),
            access_count = access_count + 1,
            stale = 0
        WHERE id = ?
    """, (mem_id,))


def auto_adjust_priority(conn, mem_id):
    row = conn.execute(
        "SELECT access_count, priority FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()
    if row:
        suggested = min(10, row["access_count"] // 5)
        if suggested > row["priority"]:
            conn.execute("UPDATE memories SET priority = ? WHERE id = ?", (suggested, mem_id))


# ── Smart Ingest (v4 Feature #4) ──

def smart_ingest(category, content, tags="", project=None, priority=0, related_to=None,
                 expires_at=None, source="manual", topic_key=None, derived_from=None,
                 citations=None, reasoning=None):
    """
    Smart ingestion with 4-tier similarity handling:
    - SKIP: >85% (duplicate blocked)
    - UPDATE: 70-85% same category/project (auto-update existing)
    - SUPERSEDE: 50-70% same project (insert new, mark old superseded)
    - CREATE: <50% (normal insert)
    """
    tags = auto_tag(content, tags)

    # Check for topic_key upsert
    if topic_key:
        conn = get_db()
        existing = conn.execute(
            "SELECT id, tags, revision_count FROM memories WHERE topic_key = ? AND active = 1",
            (topic_key,)
        ).fetchone()

        if existing:
            # Upsert: update content, merge tags, bump revision
            existing_tags = set(filter(None, existing["tags"].split(",")))
            new_tags = set(filter(None, tags.split(",")))
            merged_tags = ",".join(sorted(existing_tags | new_tags))
            new_revision = existing["revision_count"] + 1

            conn.execute("""
                UPDATE memories SET
                    content = ?,
                    tags = ?,
                    updated_at = datetime('now'),
                    revision_count = ?,
                    stale = 0
                WHERE id = ?
            """, (content, merged_tags, new_revision, existing["id"]))
            touch_memory(conn, existing["id"])
            embed_and_store(conn, existing["id"], content)
            conn.commit()
            conn.close()
            export_memory_md()
            print(f"Updated memory #{existing['id']} (revision {new_revision}) key:{topic_key}")
            return existing["id"]
        else:
            # New topic_key, insert normally
            conn.close()
            # Fall through to normal insert with topic_key set

    # Similarity-based dedup/smart-ingest
    similar = find_similar(content, category, project, threshold=0.5)

    if similar:
        best_id, best_content, score, best_cat, best_proj = similar[0]

        # SKIP: >85% (blocked)
        if score > 0.85:
            print(f"DUPLICATE BLOCKED (score={score:.0%}): similar to #{best_id}")
            print(f"  Existing: {best_content}")
            print(f"  Use 'memory-tool update {best_id} \"{content}\"' to update instead.")
            return None

        # UPDATE: 70-85% same category and project
        elif score > 0.70 and category == best_cat and project == best_proj:
            conn = get_db()
            existing = conn.execute(
                "SELECT tags, revision_count FROM memories WHERE id = ?", (best_id,)
            ).fetchone()
            existing_tags = set(filter(None, existing["tags"].split(",")))
            new_tags = set(filter(None, tags.split(",")))
            merged_tags = ",".join(sorted(existing_tags | new_tags))
            new_revision = existing["revision_count"] + 1

            conn.execute("""
                UPDATE memories SET
                    content = ?,
                    tags = ?,
                    updated_at = datetime('now'),
                    revision_count = ?,
                    stale = 0
                WHERE id = ?
            """, (content, merged_tags, new_revision, best_id))
            touch_memory(conn, best_id)
            embed_and_store(conn, best_id, content)
            conn.commit()
            conn.close()
            export_memory_md()
            print(f"AUTO-UPDATED memory #{best_id} ({score:.0%} match, revision {new_revision})")
            return best_id

        # SUPERSEDE: 50-70% same project
        elif score > 0.50 and project == best_proj:
            # Insert new, mark old as superseded
            conn = get_db()
            cur = conn.execute(
                """INSERT INTO memories (category, content, tags, project, priority, accessed_at, expires_at, source, topic_key, derived_from, citations, reasoning)
                   VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)""",
                (category, content, tags, project, priority, expires_at, source, topic_key, derived_from, citations, reasoning)
            )
            new_id = cur.lastrowid

            # Deactivate old
            conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (best_id,))

            # Create supersedes relation
            conn.execute(
                "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'supersedes')",
                (new_id, best_id)
            )

            if related_to:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'related')",
                        (new_id, int(related_to))
                    )
                except (ValueError, sqlite3.IntegrityError):
                    pass

            embed_and_store(conn, new_id, content)
            conn.commit()
            conn.close()
            export_memory_md()
            print(f"Added memory #{new_id}, supersedes #{best_id} ({score:.0%} overlap, different content)")
            return new_id

        # CREATE with warning: <50% or different category/project
        else:
            print(f"WARNING: Similar memory (score={score:.0%}): #{best_id}: {best_content}")

    # CREATE: Normal insert
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO memories (category, content, tags, project, priority, accessed_at, expires_at, source, topic_key, derived_from, citations, reasoning)
           VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)""",
        (category, content, tags, project, priority, expires_at, source, topic_key, derived_from, citations, reasoning)
    )
    mem_id = cur.lastrowid

    if related_to:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'related')",
                (mem_id, int(related_to))
            )
        except (ValueError, sqlite3.IntegrityError):
            pass

    embed_and_store(conn, mem_id, content)
    conn.commit()
    conn.close()

    # Phase 3: Auto-link to graph entities
    auto_link_memory(mem_id, content)

    export_memory_md()
    key_str = f" key:{topic_key}" if topic_key else ""
    print(f"Added memory #{mem_id} [{category}]{key_str}{' tags:' + tags if tags else ''}")
    return mem_id


def add_memory(category, content, tags="", project=None, priority=0, related_to=None,
               expires_at=None, source="manual", topic_key=None, skip_dedup=False,
               derived_from=None, citations=None, reasoning=None):
    """Legacy add_memory wrapper for backward compatibility."""
    if skip_dedup:
        # Old behavior: skip dedup entirely
        tags = auto_tag(content, tags)
        conn = get_db()
        cur = conn.execute(
            """INSERT INTO memories (category, content, tags, project, priority, accessed_at, expires_at, source, topic_key, derived_from, citations, reasoning)
               VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)""",
            (category, content, tags, project, priority, expires_at, source, topic_key, derived_from, citations, reasoning)
        )
        mem_id = cur.lastrowid
        if related_to:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'related')",
                    (mem_id, int(related_to))
                )
            except (ValueError, sqlite3.IntegrityError):
                pass
        embed_and_store(conn, mem_id, content)
        conn.commit()
        conn.close()

        # Phase 3: Auto-link to graph entities
        auto_link_memory(mem_id, content)

        export_memory_md()
        print(f"Added memory #{mem_id} [{category}]{' tags:' + tags if tags else ''}")
        return mem_id
    else:
        return smart_ingest(category, content, tags, project, priority, related_to, expires_at, source, topic_key, derived_from, citations, reasoning)


def search_memories(query, mode="hybrid"):
    """
    Search memories with multiple modes:
    - hybrid: Combine FTS and vector search with RRF (default)
    - keyword: FTS only
    - semantic: Vector only
    """
    conn = get_db()
    fts_results = []
    vec_results = []

    # 1. FTS keyword search
    if mode in ("hybrid", "keyword"):
        try:
            rows = conn.execute("""
                SELECT m.id FROM memories m
                JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ? AND m.active = 1
                ORDER BY rank LIMIT 20
            """, (query,)).fetchall()
            fts_results = [(r['id'], i) for i, r in enumerate(rows)]
        except sqlite3.OperationalError:
            pass

    # 2. Vector semantic search
    if mode in ("hybrid", "semantic") and has_vec_support():
        query_vec = embed_text(query)
        if query_vec is not None:
            try:
                # Get vec results (sqlite-vec requires k parameter)
                rows = conn.execute("""
                    SELECT rowid as id, distance FROM memory_vec
                    WHERE embedding MATCH ?
                    AND k = 20
                    ORDER BY distance
                """, (query_vec,)).fetchall()

                # Filter to active only
                active_ids = set(r['id'] for r in conn.execute(
                    "SELECT id FROM memories WHERE active = 1"
                ).fetchall())
                vec_results = [(r['id'], i) for i, r in enumerate(rows) if r['id'] in active_ids]
            except Exception:
                # Silently fail if vec table doesn't exist yet
                pass

    # 3. Reciprocal Rank Fusion (combine scores)
    if mode == "hybrid" and (fts_results or vec_results):
        scores = {}
        for mem_id, rank in fts_results:
            scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (RRF_K + rank + 1)
        for mem_id, rank in vec_results:
            scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (RRF_K + rank + 1)

        # Sort by combined RRF score
        ranked_ids = sorted(scores.keys(), key=lambda x: -scores[x])[:20]

        # Fetch full rows
        if ranked_ids:
            placeholders = ','.join('?' * len(ranked_ids))
            rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", ranked_ids).fetchall()
            # Re-sort by RRF score
            id_to_row = {r['id']: r for r in rows}
            rows = [id_to_row[mid] for mid in ranked_ids if mid in id_to_row]
        else:
            rows = []
    elif mode == "keyword" and fts_results:
        # Keyword-only mode: use FTS results
        mem_ids = [mid for mid, _ in fts_results]
        placeholders = ','.join('?' * len(mem_ids))
        rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", mem_ids).fetchall()
        id_to_row = {r['id']: r for r in rows}
        rows = [id_to_row[mid] for mid in mem_ids if mid in id_to_row]
    elif mode == "semantic" and vec_results:
        # Semantic-only mode: use vector results
        mem_ids = [mid for mid, _ in vec_results]
        placeholders = ','.join('?' * len(mem_ids))
        rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", mem_ids).fetchall()
        id_to_row = {r['id']: r for r in rows}
        rows = [id_to_row[mid] for mid in mem_ids if mid in id_to_row]
    else:
        rows = []

    # Fallback to LIKE if no results
    if not rows:
        rows = conn.execute("""
            SELECT * FROM memories
            WHERE active = 1 AND (content LIKE ? OR tags LIKE ? OR project LIKE ?)
            ORDER BY updated_at DESC LIMIT 20
        """, (f"%{query}%", f"%{query}%", f"%{query}%")).fetchall()

    # Touch accessed memories
    for r in rows:
        touch_memory(conn, r["id"])
        auto_adjust_priority(conn, r["id"])
    conn.commit()
    conn.close()
    return rows


def get_memory(mem_id):
    """Get full detail for a single memory."""
    conn = get_db()
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    conn.close()
    return row


def list_memories(category=None, project=None, tag=None, stale_only=False, expired_only=False):
    conn = get_db()
    query = "SELECT * FROM memories WHERE active = 1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if project:
        query += " AND project = ?"
        params.append(project)
    if tag:
        query += " AND tags LIKE ?"
        params.append(f"%{tag}%")
    if stale_only:
        query += " AND stale = 1"
    if expired_only:
        query += " AND expires_at IS NOT NULL AND expires_at < datetime('now')"
    query += " ORDER BY priority DESC, updated_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def update_memory(mem_id, content):
    conn = get_db()
    # Auto-tag the new content
    existing = conn.execute("SELECT tags, revision_count FROM memories WHERE id = ?", (mem_id,)).fetchone()
    tags = auto_tag(content, existing["tags"] if existing else "")
    new_revision = existing["revision_count"] + 1 if existing else 1
    conn.execute(
        "UPDATE memories SET content = ?, tags = ?, updated_at = datetime('now'), revision_count = ?, stale = 0 WHERE id = ?",
        (content, tags, new_revision, mem_id)
    )
    touch_memory(conn, mem_id)
    embed_and_store(conn, mem_id, content)
    conn.commit()
    conn.close()
    export_memory_md()
    print(f"Updated memory #{mem_id} (revision {new_revision})")


def delete_memory(mem_id):
    conn = get_db()
    conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (mem_id,))
    conn.commit()
    conn.close()
    export_memory_md()
    print(f"Deactivated memory #{mem_id}")


def tag_memory(mem_id, tags):
    conn = get_db()
    existing = conn.execute("SELECT tags FROM memories WHERE id = ?", (mem_id,)).fetchone()
    if existing:
        current = set(filter(None, existing["tags"].split(",")))
        new_tags = set(filter(None, tags.split(",")))
        merged = ",".join(sorted(current | new_tags))
        conn.execute("UPDATE memories SET tags = ?, updated_at = datetime('now') WHERE id = ?", (merged, mem_id))
        conn.commit()
        print(f"Tagged memory #{mem_id}: {merged}")
    conn.close()
    export_memory_md()


# ── Relationships ──

def relate_memories(id1, id2, relation_type="related"):
    conn = get_db()
    try:
        conn.execute("INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, ?)",
                     (id1, id2, relation_type))
        conn.execute("INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, ?)",
                     (id2, id1, relation_type))
        conn.commit()
        print(f"Linked #{id1} <-> #{id2} ({relation_type})")
    except sqlite3.IntegrityError as e:
        print(f"Failed: {e}")
    conn.close()


def get_related(mem_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT m.*, mr.relation_type FROM memories m
        JOIN memory_relations mr ON m.id = mr.target_id
        WHERE mr.source_id = ? AND m.active = 1
    """, (mem_id,)).fetchall()
    conn.close()
    return rows


# ── Conflict Detection (v4 Feature #3) ──

def find_conflicts():
    """Find memories with 50-85% similarity (potential conflicts)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, content, category, project FROM memories WHERE active = 1 ORDER BY project, category"
    ).fetchall()
    conn.close()

    conflicts = []
    seen = set()

    for i, row1 in enumerate(rows):
        for row2 in rows[i+1:]:
            pair_key = tuple(sorted([row1["id"], row2["id"]]))
            if pair_key in seen:
                continue

            words1 = word_set(row1["content"])
            words2 = word_set(row2["content"])
            if not words1 or not words2:
                continue

            intersection = words1 & words2
            union = words1 | words2
            jaccard = len(intersection) / len(union) if union else 0
            seq_score = SequenceMatcher(None, normalize(row1["content"]), normalize(row2["content"])).ratio()
            score = max(jaccard, seq_score)

            # Only report 50-85% (below dedup threshold but suspicious)
            if 0.50 <= score < 0.85:
                conflicts.append({
                    "id1": row1["id"],
                    "id2": row2["id"],
                    "content1": row1["content"],
                    "content2": row2["content"],
                    "score": score,
                    "project": row1["project"] or row2["project"] or "Unknown",
                    "category": f"{row1['category']}/{row2['category']}" if row1["category"] != row2["category"] else row1["category"],
                })
                seen.add(pair_key)

    return sorted(conflicts, key=lambda x: (-x["score"], x["project"]))


def merge_memories(id1, id2):
    """Merge two memories: keep newer, deactivate older, merge tags and relations."""
    conn = get_db()
    mem1 = conn.execute("SELECT * FROM memories WHERE id = ?", (id1,)).fetchone()
    mem2 = conn.execute("SELECT * FROM memories WHERE id = ?", (id2,)).fetchone()

    if not mem1 or not mem2:
        print("One or both memories not found.")
        conn.close()
        return

    # Determine newer (higher updated_at)
    if mem1["updated_at"] >= mem2["updated_at"]:
        keep_id, discard_id = id1, id2
        keep_mem, discard_mem = mem1, mem2
    else:
        keep_id, discard_id = id2, id1
        keep_mem, discard_mem = mem2, mem1

    # Merge tags
    tags1 = set(filter(None, keep_mem["tags"].split(",")))
    tags2 = set(filter(None, discard_mem["tags"].split(",")))
    merged_tags = ",".join(sorted(tags1 | tags2))

    # Update keeper
    conn.execute(
        "UPDATE memories SET tags = ?, updated_at = datetime('now') WHERE id = ?",
        (merged_tags, keep_id)
    )

    # Deactivate discarded
    conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (discard_id,))

    # Transfer relations from discarded to keeper
    conn.execute("""
        UPDATE memory_relations SET source_id = ?
        WHERE source_id = ? AND target_id != ?
    """, (keep_id, discard_id, keep_id))
    conn.execute("""
        UPDATE memory_relations SET target_id = ?
        WHERE target_id = ? AND source_id != ?
    """, (keep_id, discard_id, keep_id))

    # Remove any duplicate relations
    conn.execute("DELETE FROM memory_relations WHERE source_id = target_id")

    conn.commit()
    conn.close()
    export_memory_md()
    print(f"Merged #{discard_id} into #{keep_id} (deactivated #{discard_id})")


def supersede_memory(old_id, new_id):
    """Mark old memory as superseded by new."""
    conn = get_db()

    # Verify both exist
    old = conn.execute("SELECT id FROM memories WHERE id = ?", (old_id,)).fetchone()
    new = conn.execute("SELECT id FROM memories WHERE id = ?", (new_id,)).fetchone()

    if not old or not new:
        print("One or both memories not found.")
        conn.close()
        return

    # Deactivate old
    conn.execute("UPDATE memories SET active = 0, updated_at = datetime('now') WHERE id = ?", (old_id,))

    # Create supersedes relation
    conn.execute(
        "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type) VALUES (?, ?, 'supersedes')",
        (new_id, old_id)
    )

    conn.commit()
    conn.close()
    export_memory_md()
    print(f"#{new_id} supersedes #{old_id} (deactivated #{old_id})")


# ── Topic File Export (v4 Feature #5) ──

def export_topics():
    """Generate topic .md files per project."""
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_db()

    # Get all projects
    projects = conn.execute(
        "SELECT DISTINCT project FROM memories WHERE active = 1 AND project IS NOT NULL ORDER BY project"
    ).fetchall()

    # Generate per-project topic files
    for proj_row in projects:
        project = proj_row["project"]
        filename = TOPICS_DIR / f"{project.lower().replace(' ', '_')}.md"

        lines = [f"# {project} (Auto-generated from memory DB)", ""]

        # Group by category
        categories = conn.execute(
            "SELECT DISTINCT category FROM memories WHERE active = 1 AND project = ? ORDER BY category",
            (project,)
        ).fetchall()

        for cat_row in categories:
            category = cat_row["category"]
            lines.append(f"## {category.title()}")

            mems = conn.execute(
                "SELECT * FROM memories WHERE active = 1 AND project = ? AND category = ? ORDER BY priority DESC, updated_at DESC",
                (project, category)
            ).fetchall()

            for m in mems:
                lines.append(f"- #{m['id']} {m['content']}")

            lines.append("")

        filename.write_text("\n".join(lines))
        print(f"  Exported {filename}")

    # Special: people.md (all contacts)
    contacts = conn.execute(
        "SELECT * FROM memories WHERE active = 1 AND category = 'contact' ORDER BY project, priority DESC"
    ).fetchall()
    if contacts:
        lines = ["# People (Auto-generated from memory DB)", ""]
        for c in contacts:
            proj = f" [{c['project']}]" if c["project"] else ""
            lines.append(f"- #{c['id']} {c['content']}{proj}")
        (TOPICS_DIR / "people.md").write_text("\n".join(lines))
        print(f"  Exported {TOPICS_DIR / 'people.md'}")

    # Special: todo.md (all pending)
    pending = conn.execute(
        "SELECT * FROM memories WHERE active = 1 AND category = 'pending' ORDER BY project, priority DESC"
    ).fetchall()
    if pending:
        lines = ["# TODO (Auto-generated from memory DB)", ""]
        for p in pending:
            proj = f" [{p['project']}]" if p["project"] else ""
            lines.append(f"- [ ] #{p['id']} {p['content']}{proj}")
        (TOPICS_DIR / "todo.md").write_text("\n".join(lines))
        print(f"  Exported {TOPICS_DIR / 'todo.md'}")

    conn.close()
    print(f"Topic files generated in {TOPICS_DIR}")


# ── Decay & Expiry (Upgrade #6: expiry added) ──

def run_decay():
    conn = get_db()
    now = datetime.now()
    changes = {"stale": 0, "deprioritized": 0, "expired": 0}

    # Expire items past their expiry date (Upgrade #6)
    cur = conn.execute("""
        UPDATE memories SET active = 0, stale = 0
        WHERE active = 1 AND expires_at IS NOT NULL AND expires_at < datetime('now')
    """)
    changes["expired"] = cur.rowcount

    # Flag stale pending (>30d)
    cutoff = (now - timedelta(days=STALE_PENDING_DAYS)).isoformat()
    cur = conn.execute("""
        UPDATE memories SET stale = 1
        WHERE active = 1 AND stale = 0 AND category = 'pending'
        AND created_at < ? AND (accessed_at IS NULL OR accessed_at < ?)
    """, (cutoff, cutoff))
    changes["stale"] += cur.rowcount

    # Flag stale general (>90d without access)
    cutoff = (now - timedelta(days=STALE_GENERAL_DAYS)).isoformat()
    cur = conn.execute("""
        UPDATE memories SET stale = 1
        WHERE active = 1 AND stale = 0 AND category NOT IN ('pending', 'preference', 'project')
        AND (accessed_at IS NULL OR accessed_at < ?) AND updated_at < ?
    """, (cutoff, cutoff))
    changes["stale"] += cur.rowcount

    # Deprioritize (>60d without access)
    cutoff = (now - timedelta(days=DEPRIORITIZE_DAYS)).isoformat()
    cur = conn.execute("""
        UPDATE memories SET priority = MAX(0, priority - 1)
        WHERE active = 1 AND priority > 0
        AND category NOT IN ('preference', 'project')
        AND (accessed_at IS NULL OR accessed_at < ?)
    """, (cutoff,))
    changes["deprioritized"] = cur.rowcount

    conn.commit()
    conn.close()
    print(f"Decay: {changes['stale']} stale, {changes['deprioritized']} deprioritized, {changes['expired']} expired")
    return changes


def get_stale():
    conn = get_db()
    rows = conn.execute("SELECT * FROM memories WHERE active = 1 AND stale = 1 ORDER BY category, updated_at ASC").fetchall()
    conn.close()
    return rows


# ── Session Snapshots ──

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
    parts = []
    projects_touched = set()

    # Check git repos for recent changes
    for repo_path, project_name in [    ]:
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
    for repo_path, project_name in []:  # Configure via PROJECT_PATHS
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
    # Auto-detect project from content keywords (customize for your projects)

    # Example: if "YourProject" in text: project = "YourProject"

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

def backup_db():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"memories_{timestamp}.db"

    # Use SQLite backup API for consistency
    source = sqlite3.connect(str(DB_PATH))
    dest = sqlite3.connect(str(backup_path))
    source.backup(dest)
    dest.close()
    source.close()

    # Keep only last 7 backups
    backups = sorted(BACKUP_DIR.glob("memories_*.db"), key=lambda p: p.stat().st_mtime)
    for old in backups[:-7]:
        old.unlink()

    size_kb = backup_path.stat().st_size / 1024
    print(f"Backup saved: {backup_path} ({size_kb:.1f} KB)")
    print(f"Keeping last {min(7, len(backups))} backups")
    return backup_path


def restore_db(backup_file):
    path = Path(backup_file)
    if not path.exists():
        print(f"Backup not found: {backup_file}")
        return False

    # Verify it's a valid SQLite DB
    try:
        test = sqlite3.connect(str(path))
        test.execute("SELECT COUNT(*) FROM memories")
        test.close()
    except sqlite3.Error as e:
        print(f"Invalid backup file: {e}")
        return False

    # Backup current before restoring
    if DB_PATH.exists():
        emergency = DB_PATH.with_suffix(".db.pre-restore")
        shutil.copy2(str(DB_PATH), str(emergency))
        print(f"Current DB backed up to {emergency}")

    shutil.copy2(str(path), str(DB_PATH))
    print(f"Restored from {backup_file}")
    export_memory_md()
    return True


# ── Phase 3: Graph Intelligence ──

def graph_add_entity(name, entity_type, summary="", importance=3):
    """Add or update an entity. Returns entity id."""
    conn = get_db()
    try:
        cursor = conn.execute(
            """INSERT INTO graph_entities (name, type, summary, importance)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   type = excluded.type,
                   summary = excluded.summary,
                   importance = excluded.importance,
                   updated_at = datetime('now')
               RETURNING id""",
            (name, entity_type, summary, importance)
        )
        entity_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return entity_id
    except sqlite3.IntegrityError as e:
        print(f"Error adding entity: {e}")
        conn.close()
        return None


def graph_get_or_create_entity(name, entity_type="concept", summary=""):
    """Get entity by name, create if doesn't exist. Case-insensitive lookup."""
    conn = get_db()
    # Case-insensitive lookup
    row = conn.execute(
        "SELECT id FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (name,)
    ).fetchone()

    if row:
        entity_id = row[0]
        conn.close()
        return entity_id

    conn.close()
    return graph_add_entity(name, entity_type, summary)


def graph_add_relationship(from_name, to_name, relation_type, note=""):
    """Add a relationship between two entities (by name). Creates entities if they don't exist."""
    from_id = graph_get_or_create_entity(from_name)
    to_id = graph_get_or_create_entity(to_name)

    if not from_id or not to_id:
        return False

    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO graph_relationships (from_entity_id, to_entity_id, relation_type, note)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(from_entity_id, to_entity_id, relation_type) DO UPDATE SET
                   note = excluded.note""",
            (from_id, to_id, relation_type, note)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError as e:
        print(f"Error adding relationship: {e}")
        conn.close()
        return False


def graph_set_fact(entity_name, key, value, confidence=1.0, source=""):
    """Set a fact on an entity. If key exists, update it and log history."""
    entity_id = graph_get_or_create_entity(entity_name)
    if not entity_id:
        return False

    conn = get_db()

    # Check if fact exists
    existing = conn.execute(
        "SELECT value FROM graph_facts WHERE entity_id = ? AND key = ?",
        (entity_id, key)
    ).fetchone()

    if existing and existing[0] != value:
        # Log to history
        conn.execute(
            """INSERT INTO graph_fact_history (entity_id, key, old_value, new_value)
               VALUES (?, ?, ?, ?)""",
            (entity_id, key, existing[0], value)
        )

    # Insert or update fact
    conn.execute(
        """INSERT INTO graph_facts (entity_id, key, value, confidence, source)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(entity_id, key) DO UPDATE SET
               value = excluded.value,
               confidence = excluded.confidence,
               source = excluded.source,
               updated_at = datetime('now')""",
        (entity_id, key, value, confidence, source)
    )
    conn.commit()
    conn.close()
    return True


def graph_get_entity(name):
    """Get entity with all its facts and relationships."""
    conn = get_db()

    # Get entity
    entity = conn.execute(
        "SELECT * FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (name,)
    ).fetchone()

    if not entity:
        conn.close()
        return None

    entity_id = entity['id']

    # Get facts
    facts = conn.execute(
        "SELECT * FROM graph_facts WHERE entity_id = ? ORDER BY key",
        (entity_id,)
    ).fetchall()

    # Get outgoing relationships
    out_rels = conn.execute(
        """SELECT r.*, e.name as to_name, e.type as to_type
           FROM graph_relationships r
           JOIN graph_entities e ON r.to_entity_id = e.id
           WHERE r.from_entity_id = ?
           ORDER BY r.relation_type, e.name""",
        (entity_id,)
    ).fetchall()

    # Get incoming relationships
    in_rels = conn.execute(
        """SELECT r.*, e.name as from_name, e.type as from_type
           FROM graph_relationships r
           JOIN graph_entities e ON r.from_entity_id = e.id
           WHERE r.to_entity_id = ?
           ORDER BY r.relation_type, e.name""",
        (entity_id,)
    ).fetchall()

    # Get linked memories
    linked_memories = conn.execute(
        """SELECT m.* FROM memories m
           JOIN memory_entity_links l ON m.id = l.memory_id
           WHERE l.entity_id = ? AND m.active = 1
           ORDER BY m.updated_at DESC LIMIT 5""",
        (entity_id,)
    ).fetchall()

    conn.close()

    return {
        'entity': dict(entity),
        'facts': [dict(f) for f in facts],
        'outgoing': [dict(r) for r in out_rels],
        'incoming': [dict(r) for r in in_rels],
        'memories': [dict(m) for m in linked_memories]
    }


def graph_list_entities(entity_type=None):
    """List all entities, optionally filtered by type."""
    conn = get_db()
    if entity_type:
        rows = conn.execute(
            "SELECT * FROM graph_entities WHERE type = ? ORDER BY importance DESC, name",
            (entity_type,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM graph_entities ORDER BY type, importance DESC, name"
        ).fetchall()
    conn.close()
    return rows


def graph_delete_entity(name):
    """Delete an entity and its relationships/facts."""
    conn = get_db()
    result = conn.execute(
        "DELETE FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (name,)
    )
    deleted = result.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


def graph_remove_relationship(from_name, to_name, relation_type=None):
    """Remove a relationship."""
    conn = get_db()

    # Get entity IDs
    from_id = conn.execute(
        "SELECT id FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (from_name,)
    ).fetchone()
    to_id = conn.execute(
        "SELECT id FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (to_name,)
    ).fetchone()

    if not from_id or not to_id:
        conn.close()
        return False

    if relation_type:
        result = conn.execute(
            "DELETE FROM graph_relationships WHERE from_entity_id = ? AND to_entity_id = ? AND relation_type = ?",
            (from_id[0], to_id[0], relation_type)
        )
    else:
        result = conn.execute(
            "DELETE FROM graph_relationships WHERE from_entity_id = ? AND to_entity_id = ?",
            (from_id[0], to_id[0])
        )

    deleted = result.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


def graph_remove_fact(entity_name, key):
    """Remove a fact from an entity."""
    conn = get_db()

    entity = conn.execute(
        "SELECT id FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (entity_name,)
    ).fetchone()

    if not entity:
        conn.close()
        return False

    result = conn.execute(
        "DELETE FROM graph_facts WHERE entity_id = ? AND key = ?",
        (entity[0], key)
    )
    deleted = result.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


def graph_spread(start_entity_name, depth=2):
    """
    Spreading activation: starting from an entity, find connected entities
    up to `depth` hops. Returns entities with activation scores (closer = higher).

    depth=1: direct connections only
    depth=2: friends-of-friends (default)
    depth=3: 3 hops

    Activation decays by 0.5 per hop.
    """
    conn = get_db()

    # Get start entity
    start = conn.execute(
        "SELECT id, name, type FROM graph_entities WHERE LOWER(name) = LOWER(?)",
        (start_entity_name,)
    ).fetchone()

    if not start:
        conn.close()
        return []

    start_id = start[0]

    # BFS to find connected entities
    visited = {start_id: 1.0}  # entity_id -> activation score
    queue = [(start_id, 0)]  # (entity_id, current_depth)

    while queue:
        entity_id, current_depth = queue.pop(0)

        if current_depth >= depth:
            continue

        # Get neighbors (both directions)
        neighbors = conn.execute(
            """SELECT DISTINCT
                   CASE
                       WHEN from_entity_id = ? THEN to_entity_id
                       ELSE from_entity_id
                   END as neighbor_id
               FROM graph_relationships
               WHERE from_entity_id = ? OR to_entity_id = ?""",
            (entity_id, entity_id, entity_id)
        ).fetchall()

        decay = 0.5 ** (current_depth + 1)

        for neighbor in neighbors:
            neighbor_id = neighbor[0]
            if neighbor_id not in visited:
                visited[neighbor_id] = decay
                queue.append((neighbor_id, current_depth + 1))
            else:
                # Update activation if this path gives higher score
                visited[neighbor_id] = max(visited[neighbor_id], decay)

    # Get entity details for all visited (except start)
    visited.pop(start_id, None)

    if not visited:
        conn.close()
        return []

    placeholders = ','.join('?' * len(visited))
    entities = conn.execute(
        f"SELECT * FROM graph_entities WHERE id IN ({placeholders})",
        list(visited.keys())
    ).fetchall()

    conn.close()

    # Build result with activation scores
    result = []
    for e in entities:
        entity_dict = dict(e)
        entity_dict['activation'] = visited[e['id']]
        result.append(entity_dict)

    # Sort by activation score (highest first)
    result.sort(key=lambda x: -x['activation'])

    return result


def link_memory_to_entity(memory_id, entity_name):
    """Link a memory to a graph entity."""
    entity_id = graph_get_or_create_entity(entity_name)
    if not entity_id:
        return False

    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO memory_entity_links (memory_id, entity_id) VALUES (?, ?)",
            (memory_id, entity_id)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error linking memory to entity: {e}")
        conn.close()
        return False


def auto_link_memory(memory_id, content):
    """Auto-detect entity mentions in content and create links."""
    conn = get_db()

    # Get all entity names
    entities = conn.execute("SELECT id, name FROM graph_entities").fetchall()

    content_lower = content.lower()
    linked_count = 0

    for entity in entities:
        entity_id, entity_name = entity[0], entity[1]

        # Simple substring match (case-insensitive)
        if entity_name.lower() in content_lower:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_entity_links (memory_id, entity_id) VALUES (?, ?)",
                    (memory_id, entity_id)
                )
                linked_count += 1
            except sqlite3.Error:
                pass

    conn.commit()
    conn.close()
    return linked_count


def graph_auto_link_all():
    """Auto-link all existing memories to entities."""
    conn = get_db()
    memories = conn.execute("SELECT id, content FROM memories WHERE active = 1").fetchall()
    conn.close()

    total_links = 0
    for mem in memories:
        links = auto_link_memory(mem[0], mem[1])
        total_links += links

    return total_links, len(memories)


def graph_import_openclaw():
    """Import entities, relationships, and facts from OpenClaw's graph DB."""
    openclaw_path = Path.home() / ".openclaw" / "memory-graph.db"

    if not openclaw_path.exists():
        print(f"OpenClaw graph DB not found at {openclaw_path}")
        return

    try:
        source = sqlite3.connect(str(openclaw_path))
        source.row_factory = sqlite3.Row

        # Import entities
        entities = source.execute("SELECT * FROM entities").fetchall()
        entity_map = {}  # old_id -> new_id

        for e in entities:
            new_id = graph_add_entity(
                e['name'],
                e['type'],
                e['summary'] or "",
                e['importance'] or 3
            )
            entity_map[e['id']] = new_id

        print(f"Imported {len(entities)} entities")

        # Import relationships
        relationships = source.execute("SELECT * FROM relationships").fetchall()
        for r in relationships:
            from_id = entity_map.get(r['from_entity_id'])
            to_id = entity_map.get(r['to_entity_id'])

            if from_id and to_id:
                conn = get_db()
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO graph_relationships
                           (from_entity_id, to_entity_id, relation_type, note)
                           VALUES (?, ?, ?, ?)""",
                        (from_id, to_id, r['relation_type'], r['note'] or "")
                    )
                    conn.commit()
                    conn.close()
                except sqlite3.Error:
                    conn.close()

        print(f"Imported {len(relationships)} relationships")

        # Import facts
        facts = source.execute("SELECT * FROM facts").fetchall()
        imported_facts = 0
        for f in facts:
            entity_id = entity_map.get(f['entity_id'])
            if entity_id:
                conn = get_db()
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO graph_facts
                           (entity_id, key, value, confidence, source)
                           VALUES (?, ?, ?, ?, ?)""",
                        (entity_id, f['key'], f['value'], f['confidence'] or 1.0, f['source'] or "")
                    )
                    conn.commit()
                    conn.close()
                    imported_facts += 1
                except sqlite3.Error:
                    conn.close()

        print(f"Imported {imported_facts} facts")

        source.close()

    except sqlite3.Error as e:
        print(f"Error importing from OpenClaw: {e}")


def graph_stats():
    """Get graph statistics."""
    conn = get_db()

    entities = conn.execute("SELECT COUNT(*) FROM graph_entities").fetchone()[0]
    relationships = conn.execute("SELECT COUNT(*) FROM graph_relationships").fetchone()[0]
    facts = conn.execute("SELECT COUNT(*) FROM graph_facts").fetchone()[0]
    links = conn.execute("SELECT COUNT(*) FROM memory_entity_links").fetchone()[0]

    # Entity breakdown by type
    by_type = conn.execute(
        "SELECT type, COUNT(*) as count FROM graph_entities GROUP BY type ORDER BY count DESC"
    ).fetchall()

    conn.close()

    return {
        'entities': entities,
        'relationships': relationships,
        'facts': facts,
        'memory_links': links,
        'by_type': [dict(t) for t in by_type]
    }


# ── Smart MEMORY.md Export ──

def export_memory_md(focus_project=None):
    conn = get_db()
    lines = []
    lines.append("# Persistent Memory (Auto-Generated)")
    lines.append(f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                 f"v4: progressive disclosure, topic upserts, conflicts, smart ingest, topics export, budget cap_")
    lines.append("")

    # Latest session snapshot (limit to 3 recent, auto-prune older)
    snaps = conn.execute("SELECT * FROM session_snapshots ORDER BY created_at DESC LIMIT 3").fetchall()
    if snaps:
        lines.append(f"## Last Session ({snaps[0]['created_at'][:16]})")
        lines.append(f"{snaps[0]['summary']}")
        if snaps[0]["project"]:
            lines.append(f"Project: {snaps[0]['project']}")
        lines.append("")

    # Projects
    projects = conn.execute(
        "SELECT DISTINCT project FROM memories WHERE active = 1 AND project IS NOT NULL ORDER BY project"
    ).fetchall()
    if projects:
        lines.append("## Active Projects")
        for proj in projects:
            p = proj["project"]
            is_focus = (focus_project and p == focus_project)
            details = conn.execute(
                "SELECT * FROM memories WHERE active = 1 AND project = ? AND category = 'project' ORDER BY priority DESC, updated_at DESC",
                (p,)
            ).fetchall()
            marker = " (ACTIVE)" if is_focus else ""
            lines.append(f"### {p}{marker}")
            for d in details:
                lines.append(f"- {d['content']}")
            lines.append("")

    # Pending
    pending = conn.execute(
        "SELECT * FROM memories WHERE active = 1 AND category = 'pending' ORDER BY priority DESC, created_at DESC LIMIT 10"
    ).fetchall()
    if pending:
        lines.append("## Pending / TODO")
        for p in pending:
            proj = f" [{p['project']}]" if p["project"] else ""
            stale_mark = " (STALE)" if p["stale"] else ""
            exp = ""
            if p["expires_at"]:
                exp_date = p["expires_at"][:10]
                if p["expires_at"] < datetime.now().isoformat():
                    exp = " (EXPIRED)"
                else:
                    exp = f" (due {exp_date})"
            lines.append(f"- [ ] {p['content']}{proj}{stale_mark}{exp}")
        lines.append("")

    # Decisions (prioritize focus project)
    decisions = conn.execute("""
        SELECT * FROM memories WHERE active = 1 AND category = 'decision'
        ORDER BY CASE WHEN project = ? THEN 0 ELSE 1 END, priority DESC, updated_at DESC LIMIT 10
    """, (focus_project,)).fetchall()
    if decisions:
        lines.append("## Key Decisions")
        for d in decisions:
            proj = f" [{d['project']}]" if d["project"] else ""
            lines.append(f"- {d['content']}{proj}")
        lines.append("")

    # Preferences
    prefs = conn.execute(
        "SELECT * FROM memories WHERE active = 1 AND category = 'preference' ORDER BY priority DESC"
    ).fetchall()
    if prefs:
        lines.append("## User Preferences")
        for p in prefs:
            lines.append(f"- {p['content']}")
        lines.append("")

    # Errors & Learnings (limit to 5 if over budget)
    errors_limit = 10
    errors = conn.execute("""
        SELECT * FROM memories WHERE active = 1 AND stale = 0 AND category IN ('error', 'learning')
        ORDER BY CASE WHEN project = ? THEN 0 ELSE 1 END, access_count DESC, updated_at DESC LIMIT ?
    """, (focus_project, errors_limit)).fetchall()
    if errors:
        lines.append("## Errors & Learnings")
        for e in errors:
            proj = f" [{e['project']}]" if e["project"] else ""
            tag = f" ({e['tags']})" if e["tags"] else ""
            acc = f" [x{e['access_count']}]" if e["access_count"] > 2 else ""
            src = " [auto]" if e["source"] == "auto-hook" else ""
            lines.append(f"- [{e['category']}] {e['content']}{proj}{tag}{acc}{src}")
        lines.append("")

    # Architecture (limit to 4 if over budget)
    arch_limit = 8
    arch = conn.execute("""
        SELECT * FROM memories WHERE active = 1 AND category = 'architecture'
        ORDER BY CASE WHEN project = ? THEN 0 ELSE 1 END, priority DESC LIMIT ?
    """, (focus_project, arch_limit)).fetchall()
    if arch:
        lines.append("## Architecture")
        for a in arch:
            proj = f" [{a['project']}]" if a["project"] else ""
            lines.append(f"- {a['content']}{proj}")
        lines.append("")

    # Workflow (limit to 3 if over budget)
    workflow_limit = 6
    workflow = conn.execute("""
        SELECT * FROM memories WHERE active = 1 AND category = 'workflow'
        ORDER BY CASE WHEN project = ? THEN 0 ELSE 1 END, priority DESC LIMIT ?
    """, (focus_project, workflow_limit)).fetchall()
    if workflow:
        lines.append("## Workflow")
        for w in workflow:
            proj = f" [{w['project']}]" if w["project"] else ""
            lines.append(f"- {w['content']}{proj}")
        lines.append("")

    # Stale/expired counts
    stale_count = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1 AND stale = 1").fetchone()["c"]
    expired_count = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE active = 1 AND expires_at IS NOT NULL AND expires_at < datetime('now')"
    ).fetchone()["c"]
    notes = []
    if stale_count:
        notes.append(f"{stale_count} stale")
    if expired_count:
        notes.append(f"{expired_count} expired")
    if notes:
        lines.append(f"_{', '.join(notes)} memories hidden. Run `memory-tool stale` to review._")
        lines.append("")

    # Footer
    total = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1").fetchone()["c"]
    lines.append("---")
    lines.append(f"_Total: {total} memories | Manage: `memory-tool help`_")

    # Budget cap check (v4 Feature #6) - BEFORE closing connection
    content = "\n".join(lines)
    if len(content.encode('utf-8')) > MAX_MEMORY_MD_BYTES:
        # Simplified approach: just truncate with warning
        lines_text = "\n".join(lines)
        max_chars = MAX_MEMORY_MD_BYTES - 100  # Leave room for warning
        if len(lines_text) > max_chars:
            lines_text = lines_text[:max_chars]
            lines_text += "\n\n_[Over budget — run `memory-tool topics` for full view]_"
        content = lines_text

    conn.close()
    MEMORY_MD_PATH.write_text(content + "\n")


# ── Garbage Collection ──

def garbage_collect(days=180):
    conn = get_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    cur = conn.execute("DELETE FROM memories WHERE active = 0 AND updated_at < ?", (cutoff,))
    count = cur.rowcount
    conn.execute("""
        DELETE FROM memory_relations
        WHERE source_id NOT IN (SELECT id FROM memories) OR target_id NOT IN (SELECT id FROM memories)
    """)

    # Prune old session snapshots (keep max 30)
    cur2 = conn.execute("""
        DELETE FROM session_snapshots
        WHERE id NOT IN (
            SELECT id FROM session_snapshots ORDER BY created_at DESC LIMIT 30
        )
    """)
    snapshot_count = cur2.rowcount

    conn.commit()
    conn.close()
    print(f"GC: purged {count} inactive memories older than {days} days, {snapshot_count} old snapshots")


def reindex_embeddings():
    """Bulk-embed all active memories that don't have embeddings yet."""
    if not has_vec_support():
        print("Error: Vector search not available (missing dependencies or model files)")
        return

    conn = get_db()

    # Get all active memories
    all_memories = conn.execute(
        "SELECT id, content FROM memories WHERE active = 1 ORDER BY id"
    ).fetchall()

    if not all_memories:
        print("No active memories to index")
        conn.close()
        return

    # Check which ones already have embeddings
    try:
        existing_ids = set(
            r['rowid'] for r in conn.execute("SELECT rowid FROM memory_vec").fetchall()
        )
    except:
        # Vec table doesn't exist yet
        existing_ids = set()

    # Find missing embeddings
    to_embed = [(r['id'], r['content']) for r in all_memories if r['id'] not in existing_ids]

    if not to_embed:
        print(f"All {len(all_memories)} active memories already have embeddings")
        conn.close()
        return

    print(f"Indexing {len(to_embed)} memories (batch size 32)...")

    # Batch process
    BATCH_SIZE = 32
    indexed = 0

    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i:i+BATCH_SIZE]
        batch_ids = [mem_id for mem_id, _ in batch]
        batch_texts = [content for _, content in batch]

        # Generate embeddings
        embeddings = embed_texts_batch(batch_texts)

        # Store embeddings
        for mem_id, embedding in zip(batch_ids, embeddings):
            if embedding is not None:
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO memory_vec(rowid, embedding) VALUES (?, ?)",
                        (mem_id, embedding)
                    )
                    indexed += 1
                except Exception as e:
                    print(f"Warning: Failed to store embedding for #{mem_id}: {e}")

        # Progress update
        if (i + BATCH_SIZE) % 128 == 0:
            print(f"  Indexed {min(i + BATCH_SIZE, len(to_embed))}/{len(to_embed)}...")

    conn.commit()
    conn.close()
    print(f"Reindex complete: {indexed}/{len(to_embed)} embeddings stored")


# ── Display ──

def format_row(row):
    """Full verbose format."""
    tags = f" tags:{row['tags']}" if row["tags"] else ""
    proj = f" project:{row['project']}" if row["project"] else ""
    stale = " [STALE]" if row["stale"] else ""
    acc = f" acc:{row['access_count']}" if row["access_count"] else ""
    exp = ""
    if row["expires_at"]:
        if row["expires_at"] < datetime.now().isoformat():
            exp = " [EXPIRED]"
        else:
            exp = f" [expires:{row['expires_at'][:10]}]"
    src = f" src:{row['source']}" if row["source"] != "manual" else ""
    key = ""
    rev = ""
    try:
        if row["topic_key"]:
            key = f" key:{row['topic_key']}"
    except (KeyError, IndexError):
        pass
    try:
        if row["revision_count"] and row["revision_count"] > 1:
            rev = f" rev:{row['revision_count']}"
    except (KeyError, IndexError):
        pass
    derived = ""
    try:
        if row["derived_from"]:
            derived = f" derived:{row['derived_from']}"
    except (KeyError, IndexError, TypeError):
        pass
    return (f"  #{row['id']} [{row['category']}]{proj}{tags}{acc}{stale}{exp}{src}{key}{rev}{derived}"
            f" ({row['updated_at'][:10]})\n    {row['content']}")


def format_row_compact(row):
    """Compact format (v4 Feature #1)."""
    content_preview = row['content'][:100]
    if len(row['content']) > 100:
        content_preview += "..."
    proj = f" project:{row['project']}" if row["project"] else ""
    acc = f" ({row['access_count']}x)" if row["access_count"] else ""
    return f"#{row['id']} [{row['category']}]{proj} {content_preview}{acc}"


def print_memory_full(mem_id):
    """Print full detail for a single memory (v4 Feature #1)."""
    mem = get_memory(mem_id)
    if not mem:
        print(f"Memory #{mem_id} not found.")
        return

    print(f"\n=== Memory #{mem['id']} ===")
    print(f"Category: {mem['category']}")
    print(f"Content: {mem['content']}")
    if mem["project"]:
        print(f"Project: {mem['project']}")
    if mem["tags"]:
        print(f"Tags: {mem['tags']}")
    print(f"Priority: {mem['priority']}")
    print(f"Created: {mem['created_at']}")
    print(f"Updated: {mem['updated_at']}")
    if mem["accessed_at"]:
        print(f"Last accessed: {mem['accessed_at']}")
    print(f"Access count: {mem['access_count']}")
    if mem["stale"]:
        print(f"Status: STALE")
    if mem["expires_at"]:
        print(f"Expires: {mem['expires_at']}")
    print(f"Source: {mem['source']}")
    try:
        if mem["topic_key"]:
            print(f"Topic key: {mem['topic_key']}")
    except (KeyError, IndexError):
        pass
    try:
        if mem["revision_count"] and mem["revision_count"] > 1:
            print(f"Revisions: {mem['revision_count']}")
    except (KeyError, IndexError):
        pass

    # Provenance fields
    try:
        if mem["derived_from"]:
            print(f"Derived from: {mem['derived_from']}")
    except (KeyError, IndexError, TypeError):
        pass
    try:
        if mem["citations"]:
            print(f"Citations: {mem['citations']}")
    except (KeyError, IndexError, TypeError):
        pass
    try:
        if mem["reasoning"]:
            print(f"Reasoning: {mem['reasoning']}")
    except (KeyError, IndexError, TypeError):
        pass

    # Related memories
    related = get_related(mem_id)
    if related:
        print("\nRelated memories:")
        for r in related:
            print(f"  -> #{r['id']} ({r['relation_type']}): {r['content']}")
    print()


def print_help():
    print(__doc__)


# ── Run Tracking System ──

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

OPENCLAW_MEMORY_DIR = Path.home() / ".openclaw" / "workspace" / "memory"
OPENCLAW_GRAPH_DB = Path.home() / ".openclaw" / "memory-graph.db"
SYNC_STATE_FILE = MEMORY_DIR / ".sync-state.json"


def load_sync_state():
    """Load sync state (checksums) from JSON."""
    if SYNC_STATE_FILE.exists():
        try:
            return json.loads(SYNC_STATE_FILE.read_text())
        except:
            return {}
    return {}


def save_sync_state(state):
    """Save sync state to JSON."""
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))


def file_checksum(content):
    """Generate MD5 checksum for content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def sync_to_openclaw():
    """Export memories and graph data to OpenClaw's workspace format."""
    if not OPENCLAW_MEMORY_DIR.exists():
        print(f"OpenClaw memory directory not found: {OPENCLAW_MEMORY_DIR}")
        return

    conn = get_db()
    state = load_sync_state()
    files_written = []

    # 1. Export claude-code-bridge.md (session handoff)
    bridge_path = OPENCLAW_MEMORY_DIR / "claude-code-bridge.md"

    # Get recent changes
    recent_mems = conn.execute("""
        SELECT * FROM memories
        WHERE active = 1 AND updated_at >= datetime('now', '-24 hours')
        ORDER BY updated_at DESC LIMIT 20
    """).fetchall()

    recent_snaps = conn.execute("""
        SELECT * FROM session_snapshots
        ORDER BY created_at DESC LIMIT 3
    """).fetchall()

    # Get pending items
    pending = conn.execute("""
        SELECT * FROM memories
        WHERE active = 1 AND category = 'pending'
        ORDER BY priority DESC, created_at DESC LIMIT 15
    """).fetchall()

    # Get recent decisions
    decisions = conn.execute("""
        SELECT * FROM memories
        WHERE active = 1 AND category = 'decision'
        ORDER BY updated_at DESC LIMIT 10
    """).fetchall()

    # Build bridge content
    bridge_lines = []
    bridge_lines.append("# Claude Code Memory Bridge")
    bridge_lines.append(f"_Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    bridge_lines.append("")

    # Active context from recent snapshots
    if recent_snaps:
        bridge_lines.append("## Active Context")
        for snap in recent_snaps[:2]:
            bridge_lines.append(f"- **{snap['created_at'][:16]}**: {snap['summary']}")
            if snap['project']:
                bridge_lines.append(f"  - Project: {snap['project']}")
        bridge_lines.append("")

    # Recent changes
    if recent_mems:
        bridge_lines.append("## Recent Changes (Last 24h)")
        for mem in recent_mems[:10]:
            cat = mem['category']
            proj = f" [{mem['project']}]" if mem['project'] else ""
            bridge_lines.append(f"- [{cat}] {mem['content'][:120]}{proj}")
        bridge_lines.append("")

    # Key decisions
    if decisions:
        bridge_lines.append("## Key Decisions")
        for dec in decisions:
            proj = f" [{dec['project']}]" if dec['project'] else ""
            bridge_lines.append(f"- {dec['content']}{proj}")
        bridge_lines.append("")

    # Pending items
    if pending:
        bridge_lines.append("## Pending Items")
        for p in pending:
            proj = f" [{p['project']}]" if p['project'] else ""
            exp = ""
            if p['expires_at']:
                exp_date = p['expires_at'][:10]
                if p['expires_at'] < datetime.now().isoformat():
                    exp = " (EXPIRED)"
                else:
                    exp = f" (due {exp_date})"
            bridge_lines.append(f"- [ ] {p['content']}{proj}{exp}")
        bridge_lines.append("")

    # Graph summary
    g_stats = graph_stats()
    bridge_lines.append("## Shared Graph")
    bridge_lines.append(f"- Entities: {g_stats['entities']} | Relationships: {g_stats['relationships']} | Facts: {g_stats['facts']}")
    if g_stats['by_type']:
        bridge_lines.append(f"- Entity types: " + ", ".join([f"{t['type']}({t['count']})" for t in g_stats['by_type']]))
    bridge_lines.append("")

    # Memory stats
    total = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1").fetchone()['c']
    stale = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1 AND stale = 1").fetchone()['c']
    bridge_lines.append("## Memory Stats")
    bridge_lines.append(f"- Total active: {total} | Stale: {stale}")
    bridge_lines.append(f"- Source: `{DB_PATH}`")
    bridge_lines.append("")

    bridge_content = "\n".join(bridge_lines)

    # Write if changed
    old_checksum = state.get('bridge_checksum', '')
    new_checksum = file_checksum(bridge_content)
    if old_checksum != new_checksum:
        bridge_path.write_text(bridge_content)
        state['bridge_checksum'] = new_checksum
        state['last_sync'] = datetime.now().isoformat()
        files_written.append(str(bridge_path))

    # 2. Export graph-sync.md (graph data in human-readable format)
    graph_path = OPENCLAW_MEMORY_DIR / "graph-sync.md"

    entities = conn.execute("""
        SELECT * FROM graph_entities
        ORDER BY importance DESC, updated_at DESC
    """).fetchall()

    graph_lines = []
    graph_lines.append("# Claude Code Graph Export")
    graph_lines.append(f"_Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    graph_lines.append("")
    graph_lines.append(f"Total: {g_stats['entities']} entities, {g_stats['relationships']} relationships, {g_stats['facts']} facts")
    graph_lines.append("")

    # Group entities by type
    for entity_type in ['project', 'person', 'org', 'feature', 'tool', 'service', 'concept']:
        type_entities = [e for e in entities if e['type'] == entity_type]
        if type_entities:
            graph_lines.append(f"## {entity_type.title()}s")
            for e in type_entities:
                graph_lines.append(f"### {e['name']}")
                if e['summary']:
                    graph_lines.append(f"{e['summary']}")
                graph_lines.append(f"_Importance: {e['importance']}, Updated: {e['updated_at'][:16]}_")

                # Facts
                facts = conn.execute(
                    "SELECT * FROM graph_facts WHERE entity_id = ? ORDER BY key",
                    (e['id'],)
                ).fetchall()
                if facts:
                    graph_lines.append("**Facts:**")
                    for f in facts:
                        conf = f" (confidence: {f['confidence']})" if f['confidence'] < 1.0 else ""
                        graph_lines.append(f"- {f['key']}: {f['value']}{conf}")

                # Relationships
                rels_out = conn.execute("""
                    SELECT r.relation_type, r.note, e2.name, e2.type
                    FROM graph_relationships r
                    JOIN graph_entities e2 ON r.to_entity_id = e2.id
                    WHERE r.from_entity_id = ?
                """, (e['id'],)).fetchall()

                rels_in = conn.execute("""
                    SELECT r.relation_type, r.note, e1.name, e1.type
                    FROM graph_relationships r
                    JOIN graph_entities e1 ON r.from_entity_id = e1.id
                    WHERE r.to_entity_id = ?
                """, (e['id'],)).fetchall()

                if rels_out:
                    graph_lines.append("**Relationships (outgoing):**")
                    for r in rels_out:
                        note = f" - {r['note']}" if r['note'] else ""
                        graph_lines.append(f"- --{r['relation_type']}--> {r['name']} ({r['type']}){note}")

                if rels_in:
                    graph_lines.append("**Relationships (incoming):**")
                    for r in rels_in:
                        note = f" - {r['note']}" if r['note'] else ""
                        graph_lines.append(f"- <--{r['relation_type']}-- {r['name']} ({r['type']}){note}")

                graph_lines.append("")

    graph_content = "\n".join(graph_lines)

    # Write if changed
    old_checksum = state.get('graph_checksum', '')
    new_checksum = file_checksum(graph_content)
    if old_checksum != new_checksum:
        graph_path.write_text(graph_content)
        state['graph_checksum'] = new_checksum
        files_written.append(str(graph_path))

    # 3. Sync graph DB to OpenClaw's graph DB
    if OPENCLAW_GRAPH_DB.exists():
        try:
            graph_sync_to_openclaw_db()
            files_written.append(str(OPENCLAW_GRAPH_DB))
        except Exception as e:
            print(f"Warning: Graph DB sync failed: {e}")

    # Save state
    save_sync_state(state)
    conn.close()

    if files_written:
        print(f"Synced {len(files_written)} files to OpenClaw:")
        for f in files_written:
            print(f"  - {f}")
    else:
        print("No changes to sync (all files up to date)")


def graph_sync_to_openclaw_db():
    """Sync graph entities/relationships/facts to OpenClaw's graph DB."""
    if not OPENCLAW_GRAPH_DB.exists():
        print("OpenClaw graph DB not found, skipping DB sync")
        return

    source_conn = get_db()
    target_conn = sqlite3.connect(str(OPENCLAW_GRAPH_DB))
    target_conn.row_factory = sqlite3.Row

    # Ensure OpenClaw DB has the same schema
    target_conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL,
            summary TEXT DEFAULT '',
            importance INTEGER DEFAULT 3,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_entity_id INTEGER NOT NULL,
            to_entity_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            note TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(from_entity_id, to_entity_id, relation_type)
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(entity_id, key)
        );

        CREATE TABLE IF NOT EXISTS fact_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            old_value TEXT NOT NULL,
            new_value TEXT NOT NULL,
            changed_at TEXT DEFAULT (datetime('now'))
        );
    """)
    target_conn.commit()

    # Map entity names to IDs in both DBs
    entity_map = {}  # source_id -> target_id

    # Sync entities
    source_entities = source_conn.execute("SELECT * FROM graph_entities").fetchall()
    for e in source_entities:
        # Map entity types to OpenClaw's supported types
        entity_type = e['type']
        if entity_type not in ('person', 'project', 'org', 'feature', 'concept'):
            # Map 'tool' and 'service' to 'concept'
            entity_type = 'concept'

        # Check if entity exists in target
        existing = target_conn.execute(
            "SELECT id FROM entities WHERE name = ?", (e['name'],)
        ).fetchone()

        if existing:
            # Update existing
            target_conn.execute("""
                UPDATE entities SET type = ?, summary = ?, importance = ?, updated_at = ?
                WHERE name = ?
            """, (entity_type, e['summary'], e['importance'], e['updated_at'], e['name']))
            entity_map[e['id']] = existing['id']
        else:
            # Insert new
            cursor = target_conn.execute("""
                INSERT INTO entities (name, type, summary, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (e['name'], entity_type, e['summary'], e['importance'], e['created_at'], e['updated_at']))
            entity_map[e['id']] = cursor.lastrowid

    target_conn.commit()

    # Sync relationships
    source_rels = source_conn.execute("SELECT * FROM graph_relationships").fetchall()
    for r in source_rels:
        from_id = entity_map.get(r['from_entity_id'])
        to_id = entity_map.get(r['to_entity_id'])

        if from_id and to_id:
            try:
                target_conn.execute("""
                    INSERT OR REPLACE INTO relationships
                    (from_entity_id, to_entity_id, relation_type, note, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (from_id, to_id, r['relation_type'], r['note'], r['created_at']))
            except sqlite3.Error:
                pass

    target_conn.commit()

    # Sync facts
    source_facts = source_conn.execute("SELECT * FROM graph_facts").fetchall()
    for f in source_facts:
        entity_id = entity_map.get(f['entity_id'])
        if entity_id:
            # Check if fact exists
            existing = target_conn.execute(
                "SELECT value FROM facts WHERE entity_id = ? AND key = ?",
                (entity_id, f['key'])
            ).fetchone()

            if existing and existing['value'] != f['value']:
                # Record history
                target_conn.execute("""
                    INSERT INTO fact_history (entity_id, key, old_value, new_value)
                    VALUES (?, ?, ?, ?)
                """, (entity_id, f['key'], existing['value'], f['value']))

            # Upsert fact
            target_conn.execute("""
                INSERT OR REPLACE INTO facts
                (entity_id, key, value, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (entity_id, f['key'], f['value'], f['confidence'], f['source'],
                  f['created_at'], f['updated_at']))

    target_conn.commit()
    target_conn.close()
    source_conn.close()


def sync_from_openclaw():
    """Import new memories from OpenClaw's daily notes and topic files."""
    if not OPENCLAW_MEMORY_DIR.exists():
        print(f"OpenClaw memory directory not found: {OPENCLAW_MEMORY_DIR}")
        return

    imported_count = 0

    # Read today's daily note
    today = datetime.now().strftime("%Y-%m-%d")
    daily_path = OPENCLAW_MEMORY_DIR / f"{today}.md"

    if daily_path.exists():
        content = daily_path.read_text()

        # Simple pattern matching for structured entries
        # Look for patterns like "## [Category]" or "**Decision:**" etc.

        # Example: extract decision-like statements
        for line in content.split('\n'):
            line = line.strip()

            # Look for decision markers
            if line.startswith('**Decision:**') or line.startswith('**Recommendation:**'):
                decision = line.split(':', 1)[1].strip()
                if decision and len(decision) > 20:
                    # Check if similar memory exists
                    similar = find_similar(decision, category='decision', threshold=0.75)
                    if not similar:
                        add_memory('decision', decision, project='OpenClaw', source='openclaw-import')
                        imported_count += 1

            # Look for todo items
            elif line.startswith('- [ ]'):
                todo = line[5:].strip()
                if todo and len(todo) > 15:
                    similar = find_similar(todo, category='pending', threshold=0.75)
                    if not similar:
                        add_memory('pending', todo, project='OpenClaw', source='openclaw-import')
                        imported_count += 1

    # Import from OpenClaw's graph DB
    if OPENCLAW_GRAPH_DB.exists():
        try:
            graph_import_openclaw()
        except Exception as e:
            print(f"Warning: Failed to import from OpenClaw graph DB: {e}")

    if imported_count > 0:
        print(f"Imported {imported_count} new memories from OpenClaw")
    else:
        print("No new memories to import from OpenClaw")


def sync_bidirectional():
    """Run both sync-to and sync-from."""
    print("=== Syncing to OpenClaw ===")
    sync_to_openclaw()
    print("\n=== Syncing from OpenClaw ===")
    sync_from_openclaw()
    print("\n=== Sync complete ===")


def suggest_next():
    """Suggest next actions based on current memory state."""
    conn = get_db()
    suggestions = []

    # 1. Expiring soon (within 7 days)
    expiring = conn.execute("""
        SELECT COUNT(*) as c FROM memories
        WHERE active = 1 AND expires_at IS NOT NULL
        AND expires_at > datetime('now') AND expires_at < datetime('now', '+7 days')
    """).fetchone()["c"]
    if expiring:
        suggestions.append(f"⏰ {expiring} memories expiring within 7 days — run: memory-tool list --expired")

    # 2. Stale memories
    stale = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1 AND stale = 1").fetchone()["c"]
    if stale:
        suggestions.append(f"🕸️ {stale} stale memories need review — run: memory-tool stale")

    # 3. Pending items
    pending = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1 AND category = 'pending'").fetchone()["c"]
    if pending:
        suggestions.append(f"📋 {pending} pending items to complete or clean up — run: memory-tool pending")

    # 4. Conflicts (close connection first to avoid interference with find_conflicts)
    conn.close()
    conflicts = find_conflicts()
    if conflicts:
        suggestions.append(f"⚠️ {len(conflicts)} potential duplicate memories — run: memory-tool conflicts")

    # Re-open connection for remaining checks
    conn = get_db()

    # 5. Unembedded memories (vector index gaps)
    if has_vec_support():
        try:
            active = conn.execute("SELECT COUNT(*) as c FROM memories WHERE active = 1").fetchone()["c"]
            embedded = conn.execute("SELECT COUNT(*) as c FROM memory_vec").fetchone()["c"]
            gap = active - embedded
            if gap > 5:
                suggestions.append(f"🔍 {gap} memories not indexed for semantic search — run: memory-tool reindex")
        except:
            pass

    # 6. Orphan memories (no tags, no project, no relations)
    orphans = conn.execute("""
        SELECT COUNT(*) as c FROM memories m
        WHERE m.active = 1 AND (m.tags = '' OR m.tags IS NULL)
        AND m.project IS NULL
        AND m.id NOT IN (SELECT source_id FROM memory_relations UNION SELECT target_id FROM memory_relations)
    """).fetchone()["c"]
    if orphans:
        suggestions.append(f"🏷️ {orphans} orphan memories (no tags/project/relations) — consider tagging")

    # 7. Running runs that might be stale
    try:
        stale_runs = conn.execute("""
            SELECT COUNT(*) as c FROM runs
            WHERE status = 'running' AND started_at < datetime('now', '-24 hours')
        """).fetchone()["c"]
        if stale_runs:
            suggestions.append(f"🏃 {stale_runs} runs still 'running' for 24h+ — run: memory-tool run list --status running")
    except:
        pass  # runs table might not exist in older versions

    # 8. Backup age
    if BACKUP_DIR.exists():
        backups = sorted(BACKUP_DIR.glob("memories_*.db"))
        if backups:
            newest = backups[-1].stat().st_mtime
            days_ago = (datetime.now().timestamp() - newest) / 86400
            if days_ago > 7:
                suggestions.append(f"💾 Last backup was {int(days_ago)} days ago — run: memory-tool backup")
        else:
            suggestions.append("💾 No backups found — run: memory-tool backup")
    else:
        suggestions.append("💾 No backups found — run: memory-tool backup")

    # 9. Unlinked graph entities
    try:
        unlinked = conn.execute("""
            SELECT COUNT(*) as c FROM graph_entities ge
            WHERE ge.id NOT IN (SELECT entity_id FROM memory_entity_links)
        """).fetchone()["c"]
        if unlinked:
            suggestions.append(f"🔗 {unlinked} graph entities not linked to any memories — run: memory-tool graph auto-link")
    except:
        pass  # graph tables might not exist

    conn.close()

    if suggestions:
        print("Next actions suggested:\n")
        for s in suggestions:
            print(f"  {s}")
        print(f"\n({len(suggestions)} suggestions)")
    else:
        print("✅ Everything looks good! No actions needed.")


# ── CLI ──

def parse_flags(argv, start=2):
    """Parse --key value flags from argv starting at index."""
    flags = {}
    i = start
    content_parts = []
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            key = argv[i][2:]
            flags[key] = argv[i + 1]
            i += 2
        elif argv[i].startswith("--"):
            # Boolean flag
            flags[argv[i][2:]] = True
            i += 1
        else:
            content_parts.append(argv[i])
            i += 1
    return flags, content_parts


def main():
    init_db()

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "add" and len(sys.argv) >= 4:
        category = sys.argv[2]
        content = sys.argv[3]
        flags, _ = parse_flags(sys.argv, 4)
        add_memory(
            category, content,
            tags=flags.get("tags", ""),
            project=flags.get("project"),
            priority=int(flags.get("priority", 0)),
            related_to=flags.get("related"),
            expires_at=flags.get("expires"),
            source=flags.get("source", "manual"),
            topic_key=flags.get("key"),
            derived_from=flags.get("derived-from"),
            citations=flags.get("citations"),
            reasoning=flags.get("reasoning"),
        )

    elif cmd == "search" and len(sys.argv) >= 3:
        flags, query_parts = parse_flags(sys.argv, 2)
        query = " ".join(query_parts)

        # Determine search mode
        search_mode = "hybrid"  # default
        if flags.get("semantic"):
            search_mode = "semantic"
        elif flags.get("keyword"):
            search_mode = "keyword"

        rows = search_memories(query, mode=search_mode)
        if rows:
            full_mode = flags.get("full", False)
            for r in rows:
                if full_mode:
                    print(format_row(r))
                else:
                    print(format_row_compact(r))
                # Related in compact mode too
                if not full_mode:
                    for rel in get_related(r["id"]):
                        print(f"  -> #{rel['id']}: {rel['content'][:60]}")
                else:
                    for rel in get_related(r["id"]):
                        print(f"      -> #{rel['id']} ({rel['relation_type']}): {rel['content'][:80]}")
        else:
            print("No memories found.")

    elif cmd == "get" and len(sys.argv) >= 3:
        print_memory_full(int(sys.argv[2]))

    elif cmd == "list":
        flags, _ = parse_flags(sys.argv, 2)
        rows = list_memories(
            category=flags.get("category"),
            project=flags.get("project"),
            tag=flags.get("tag"),
            stale_only="stale" in sys.argv,
            expired_only="--expired" in sys.argv,
        )
        for r in rows:
            print(format_row(r))
        print(f"\n({len(rows)} memories)")

    elif cmd == "update" and len(sys.argv) >= 4:
        update_memory(int(sys.argv[2]), " ".join(sys.argv[3:]))

    elif cmd == "delete" and len(sys.argv) >= 3:
        delete_memory(int(sys.argv[2]))

    elif cmd == "tag" and len(sys.argv) >= 4:
        tag_memory(int(sys.argv[2]), sys.argv[3])

    elif cmd == "relate" and len(sys.argv) >= 4:
        rel_type = sys.argv[4] if len(sys.argv) > 4 else "related"
        relate_memories(int(sys.argv[2]), int(sys.argv[3]), rel_type)

    elif cmd == "conflicts":
        conflicts = find_conflicts()
        if conflicts:
            print(f"Potential conflicts ({len(conflicts)} found):\n")
            for c in conflicts:
                suggest = "MERGE" if c["score"] > 0.70 else "REVIEW"
                print(f"  #{c['id1']} vs #{c['id2']} ({c['score']:.0%} similar) [{c['project']}/{c['category']}]")
                print(f"    A: {c['content1'][:80]}...")
                print(f"    B: {c['content2'][:80]}...")
                print(f"    Suggest: {suggest} (memory-tool merge {c['id1']} {c['id2']})\n")
        else:
            print("No conflicts found.")

    elif cmd == "merge" and len(sys.argv) >= 4:
        merge_memories(int(sys.argv[2]), int(sys.argv[3]))

    elif cmd == "supersede" and len(sys.argv) >= 4:
        supersede_memory(int(sys.argv[2]), int(sys.argv[3]))

    elif cmd == "pending":
        rows = list_memories(category="pending")
        for r in rows:
            print(format_row(r))
        print(f"\n({len(rows)} pending items)")

    elif cmd == "projects":
        conn = get_db()
        projects = conn.execute("""
            SELECT project, COUNT(*) as count,
                   GROUP_CONCAT(DISTINCT category) as categories,
                   SUM(CASE WHEN category='pending' THEN 1 ELSE 0 END) as pending,
                   SUM(CASE WHEN stale=1 THEN 1 ELSE 0 END) as stale
            FROM memories WHERE active = 1 AND project IS NOT NULL
            GROUP BY project ORDER BY count DESC
        """).fetchall()
        conn.close()
        for p in projects:
            print(f"  {p['project']}: {p['count']} memories, {p['pending']} pending, {p['stale']} stale ({p['categories']})")

    elif cmd == "topics":
        export_topics()

    elif cmd == "export":
        flags, _ = parse_flags(sys.argv, 2)
        project = flags.get("project") or detect_project()
        export_memory_md(project)
        focus = f" (focused on {project})" if project else ""
        print(f"Exported to {MEMORY_MD_PATH}{focus}")

    elif cmd == "stats":
        conn = get_db()
        stats = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) as active,
                   SUM(CASE WHEN stale = 1 AND active = 1 THEN 1 ELSE 0 END) as stale,
                   SUM(CASE WHEN expires_at IS NOT NULL AND expires_at < datetime('now') AND active = 1 THEN 1 ELSE 0 END) as expired,
                   COUNT(DISTINCT project) as projects,
                   COUNT(DISTINCT category) as categories,
                   SUM(access_count) as total_accesses
            FROM memories
        """).fetchone()
        cats = conn.execute("""
            SELECT category, COUNT(*) as count, SUM(access_count) as accesses
            FROM memories WHERE active = 1 GROUP BY category ORDER BY count DESC
        """).fetchall()
        relations = conn.execute("SELECT COUNT(*) as c FROM memory_relations").fetchone()["c"]
        snapshots = conn.execute("SELECT COUNT(*) as c FROM session_snapshots").fetchone()["c"]
        sources = conn.execute("""
            SELECT source, COUNT(*) as c FROM memories WHERE active = 1 GROUP BY source ORDER BY c DESC
        """).fetchall()
        topic_keys = conn.execute("SELECT COUNT(*) as c FROM memories WHERE topic_key IS NOT NULL AND active = 1").fetchone()["c"]
        # Backup info
        backup_count = len(list(BACKUP_DIR.glob("memories_*.db"))) if BACKUP_DIR.exists() else 0

        # Vector index status
        vec_indexed = 0
        if has_vec_support():
            try:
                vec_indexed = conn.execute("SELECT COUNT(*) as c FROM memory_vec").fetchone()["c"]
            except:
                pass

        # Graph stats
        g_stats = graph_stats()

        conn.close()

        print(f"Memories: {stats['total']} total ({stats['active']} active, {stats['stale']} stale, {stats['expired'] or 0} expired)")
        print(f"Projects: {stats['projects']} | Categories: {stats['categories']}")
        print(f"Relations: {relations} | Snapshots: {snapshots} | Backups: {backup_count}")
        print(f"Topic keys: {topic_keys}")
        print(f"Total accesses: {stats['total_accesses'] or 0}")
        print(f"Graph: {g_stats['entities']} entities, {g_stats['relationships']} relationships, {g_stats['facts']} facts, {g_stats['memory_links']} memory links")

        # Vector search status
        if has_vec_support():
            vec_pct = (vec_indexed / stats['active'] * 100) if stats['active'] > 0 else 0
            print(f"Vector index: {vec_indexed}/{stats['active']} embeddings ({vec_pct:.0f}%)")
        else:
            print("Vector index: Not available (install sqlite-vec, onnxruntime, tokenizers, numpy)")
        # Bridge status
        sync_state = load_sync_state()
        if sync_state.get('last_sync'):
            last_sync = sync_state['last_sync'][:16]
            openclaw_files = len(list(OPENCLAW_MEMORY_DIR.glob("*.md"))) if OPENCLAW_MEMORY_DIR.exists() else 0
            print(f"Bridge: last sync {last_sync}, {openclaw_files} OpenClaw files")
        else:
            print("Bridge: never synced (run 'memory-tool sync')")

        print("\nBy category:")
        for c in cats:
            print(f"  {c['category']}: {c['count']} (accessed {c['accesses'] or 0}x)")
        print("\nBy source:")
        for s in sources:
            print(f"  {s['source']}: {s['c']}")

    elif cmd == "stale":
        rows = get_stale()
        if rows:
            for r in rows:
                print(format_row(r))
            print(f"\n({len(rows)} stale items)")
            print("Actions: 'memory-tool delete <id>' or 'memory-tool update <id> ...'")
        else:
            print("No stale memories.")

    elif cmd == "decay":
        run_decay()

    elif cmd == "snapshot" and len(sys.argv) >= 3:
        flags, content_parts = parse_flags(sys.argv, 2)
        summary = " ".join(content_parts) if content_parts else " ".join(sys.argv[2:])
        project = flags.get("project") or detect_project()
        save_snapshot(summary, project)
        export_memory_md(project)

    elif cmd == "auto-snapshot":
        auto_snapshot()

    elif cmd == "snapshots":
        flags, _ = parse_flags(sys.argv, 2)
        limit = int(flags.get("limit", 5))
        snaps = get_snapshots(limit)
        for s in snaps:
            proj = f" [{s['project']}]" if s["project"] else ""
            files = f"\n    Files: {s['files_touched'][:150]}" if s["files_touched"] else ""
            print(f"  [{s['created_at'][:16]}]{proj} {s['summary']}{files}")
        print(f"\n({len(snaps)} snapshots)")

    elif cmd == "detect-project":
        print(detect_project() or "Unknown project")

    elif cmd == "gc":
        garbage_collect(int(sys.argv[2]) if len(sys.argv) >= 3 else 180)

    elif cmd == "log-error" and len(sys.argv) >= 4:
        command = sys.argv[2]
        error = " ".join(sys.argv[3:])
        flags, _ = parse_flags(sys.argv, 4)
        log_error(command, error, project=flags.get("project"))

    elif cmd == "import-md" and len(sys.argv) >= 3:
        import_session_md(sys.argv[2])

    elif cmd == "backup":
        backup_db()

    elif cmd == "restore" and len(sys.argv) >= 3:
        restore_db(sys.argv[2])

    elif cmd == "reindex":
        reindex_embeddings()

    elif cmd == "graph":
        # Graph intelligence subcommands
        if len(sys.argv) < 3:
            # Show graph summary
            g_stats = graph_stats()
            print(f"Graph Intelligence Summary:")
            print(f"  Entities: {g_stats['entities']}")
            print(f"  Relationships: {g_stats['relationships']}")
            print(f"  Facts: {g_stats['facts']}")
            print(f"  Memory links: {g_stats['memory_links']}")
            if g_stats['by_type']:
                print(f"\n  By type:")
                for t in g_stats['by_type']:
                    print(f"    {t['type']}: {t['count']}")
        else:
            subcmd = sys.argv[2]

            if subcmd == "add" and len(sys.argv) >= 5:
                entity_type = sys.argv[3]
                name = sys.argv[4]
                summary = " ".join(sys.argv[5:]) if len(sys.argv) > 5 else ""
                entity_id = graph_add_entity(name, entity_type, summary)
                print(f"Added entity #{entity_id}: {name} ({entity_type})")

            elif subcmd == "rel" and len(sys.argv) >= 6:
                from_name = sys.argv[3]
                relation_type = sys.argv[4]
                to_name = sys.argv[5]
                note = " ".join(sys.argv[6:]) if len(sys.argv) > 6 else ""
                success = graph_add_relationship(from_name, to_name, relation_type, note)
                if success:
                    print(f"Added relationship: {from_name} --{relation_type}--> {to_name}")
                else:
                    print("Failed to add relationship")

            elif subcmd == "fact" and len(sys.argv) >= 6:
                entity_name = sys.argv[3]
                key = sys.argv[4]
                value = " ".join(sys.argv[5:])
                success = graph_set_fact(entity_name, key, value)
                if success:
                    print(f"Set fact: {entity_name}.{key} = {value}")
                else:
                    print("Failed to set fact")

            elif subcmd == "get" and len(sys.argv) >= 4:
                name = sys.argv[3]
                entity = graph_get_entity(name)
                if entity:
                    e = entity['entity']
                    print(f"\n{e['name']} ({e['type']}) - Importance: {e['importance']}")
                    if e['summary']:
                        print(f"  Summary: {e['summary']}")
                    print(f"  Created: {e['created_at'][:16]} | Updated: {e['updated_at'][:16]}")

                    if entity['facts']:
                        print(f"\n  Facts ({len(entity['facts'])}):")
                        for f in entity['facts']:
                            conf = f" (confidence: {f['confidence']})" if f['confidence'] < 1.0 else ""
                            print(f"    {f['key']}: {f['value']}{conf}")

                    if entity['outgoing']:
                        print(f"\n  Relationships (outgoing):")
                        for r in entity['outgoing']:
                            note = f" - {r['note']}" if r['note'] else ""
                            print(f"    --{r['relation_type']}--> {r['to_name']} ({r['to_type']}){note}")

                    if entity['incoming']:
                        print(f"\n  Relationships (incoming):")
                        for r in entity['incoming']:
                            note = f" - {r['note']}" if r['note'] else ""
                            print(f"    <--{r['relation_type']}-- {r['from_name']} ({r['from_type']}){note}")

                    if entity['memories']:
                        print(f"\n  Linked memories ({len(entity['memories'])}):")
                        for m in entity['memories']:
                            print(f"    #{m['id']} [{m['category']}] {m['content'][:80]}")
                else:
                    print(f"Entity not found: {name}")

            elif subcmd == "list":
                entity_type = sys.argv[3] if len(sys.argv) >= 4 else None
                entities = graph_list_entities(entity_type)
                if entities:
                    for e in entities:
                        summary = f" - {e['summary'][:60]}" if e['summary'] else ""
                        print(f"  {e['name']} ({e['type']}) [importance: {e['importance']}]{summary}")
                    print(f"\n({len(entities)} entities)")
                else:
                    print("No entities found")

            elif subcmd == "delete" and len(sys.argv) >= 4:
                name = sys.argv[3]
                success = graph_delete_entity(name)
                if success:
                    print(f"Deleted entity: {name}")
                else:
                    print(f"Entity not found: {name}")

            elif subcmd == "spread" and len(sys.argv) >= 4:
                name = sys.argv[3]
                depth = int(sys.argv[4]) if len(sys.argv) >= 5 else 2
                results = graph_spread(name, depth)
                if results:
                    print(f"Spreading activation from '{name}' (depth={depth}):")
                    for e in results:
                        print(f"  [{e['activation']:.2f}] {e['name']} ({e['type']}) - {e['summary'][:60] if e['summary'] else '(no summary)'}")
                    print(f"\n({len(results)} entities)")
                else:
                    print(f"No connected entities found or entity '{name}' doesn't exist")

            elif subcmd == "link" and len(sys.argv) >= 5:
                memory_id = int(sys.argv[3])
                entity_name = sys.argv[4]
                success = link_memory_to_entity(memory_id, entity_name)
                if success:
                    print(f"Linked memory #{memory_id} to entity '{entity_name}'")
                else:
                    print("Failed to link memory to entity")

            elif subcmd == "auto-link":
                total_links, total_mems = graph_auto_link_all()
                print(f"Auto-linked {total_links} connections across {total_mems} memories")

            elif subcmd == "import-openclaw":
                graph_import_openclaw()

            elif subcmd == "stats":
                g_stats = graph_stats()
                print(f"Graph Statistics:")
                print(f"  Entities: {g_stats['entities']}")
                print(f"  Relationships: {g_stats['relationships']}")
                print(f"  Facts: {g_stats['facts']}")
                print(f"  Memory links: {g_stats['memory_links']}")
                if g_stats['by_type']:
                    print(f"\n  Entities by type:")
                    for t in g_stats['by_type']:
                        print(f"    {t['type']}: {t['count']}")

            else:
                print(f"Unknown graph subcommand: {subcmd}")
                print("\nGraph commands:")
                print("  memory-tool graph                         # Show graph summary")
                print("  memory-tool graph add <type> <name> [summary]")
                print("  memory-tool graph rel <from> <rel_type> <to> [note]")
                print("  memory-tool graph fact <entity> <key> <value>")
                print("  memory-tool graph get <name>")
                print("  memory-tool graph list [type]")
                print("  memory-tool graph delete <name>")
                print("  memory-tool graph spread <name> [depth]")
                print("  memory-tool graph link <memory_id> <entity_name>")
                print("  memory-tool graph auto-link")
                print("  memory-tool graph import-openclaw")
                print("  memory-tool graph stats")

    elif cmd == "sync":
        sync_bidirectional()

    elif cmd == "sync-to":
        sync_to_openclaw()

    elif cmd == "sync-from":
        sync_from_openclaw()

    elif cmd == "run":
        # Run tracking subcommands
        if len(sys.argv) < 3:
            print("Usage: memory-tool run <subcommand>")
            print("Subcommands: start, step, complete, fail, list, show, cancel")
            sys.exit(1)
            
        subcmd = sys.argv[2]
        
        if subcmd == "start" and len(sys.argv) >= 4:
            task = sys.argv[3]
            flags, _ = parse_flags(sys.argv, 4)
            agent = flags.get("agent", "claw")
            project = flags.get("project")
            tags = flags.get("tags", "")
            
            run_id = start_run(task, agent, project, tags)
            print(f"Started run #{run_id}")
            
        elif subcmd == "step" and len(sys.argv) >= 5:
            try:
                run_id = int(sys.argv[3])
                step_description = " ".join(sys.argv[4:])
                success = add_run_step(run_id, step_description)
                if success:
                    print(f"Added step to run #{run_id}")
                else:
                    print(f"Run #{run_id} not found")
            except ValueError:
                print("Invalid run ID")
                
        elif subcmd == "complete" and len(sys.argv) >= 5:
            try:
                run_id = int(sys.argv[3])
                outcome = " ".join(sys.argv[4:])
                complete_run(run_id, outcome)
                print(f"Completed run #{run_id}")
            except ValueError:
                print("Invalid run ID")
                
        elif subcmd == "fail" and len(sys.argv) >= 5:
            try:
                run_id = int(sys.argv[3])
                reason = " ".join(sys.argv[4:])
                fail_run(run_id, reason)
                print(f"Failed run #{run_id}")
            except ValueError:
                print("Invalid run ID")
                
        elif subcmd == "cancel" and len(sys.argv) >= 4:
            try:
                run_id = int(sys.argv[3])
                cancel_run(run_id)
                print(f"Cancelled run #{run_id}")
            except ValueError:
                print("Invalid run ID")
                
        elif subcmd == "list":
            flags, _ = parse_flags(sys.argv, 3)
            status = flags.get("status")
            project = flags.get("project")
            limit = int(flags.get("limit", 10))
            
            runs = list_runs(status, project, limit)
            
            if runs:
                print(f"{'ID':<4} {'Task':<50} {'Agent':<8} {'Status':<12} {'Started':<16} {'Duration':<12}")
                print("-" * 108)
                for r in runs:
                    task_preview = r['task'][:47] + "..." if len(r['task']) > 50 else r['task']
                    duration = format_duration(r['started_at'], r['completed_at'])
                    started_short = r['started_at'][:16] if r['started_at'] else "unknown"
                    print(f"{r['id']:<4} {task_preview:<50} {r['agent']:<8} {r['status']:<12} {started_short:<16} {duration:<12}")
                print(f"\n({len(runs)} runs)")
            else:
                print("No runs found")
                
        elif subcmd == "show" and len(sys.argv) >= 4:
            try:
                run_id = int(sys.argv[3])
                run = show_run(run_id)
                if run:
                    print(f"\n=== Run #{run['id']} ===")
                    print(f"Task: {run['task']}")
                    print(f"Agent: {run['agent']}")
                    print(f"Status: {run['status']}")
                    print(f"Started: {run['started_at']}")
                    if run['completed_at']:
                        print(f"Completed: {run['completed_at']}")
                        duration = format_duration(run['started_at'], run['completed_at'])
                        print(f"Duration: {duration}")
                    elif run['status'] == 'running':
                        duration = format_duration(run['started_at'])
                        print(f"Running for: {duration}")
                    if run['project']:
                        print(f"Project: {run['project']}")
                    if run['tags']:
                        print(f"Tags: {run['tags']}")
                    if run['outcome']:
                        print(f"Outcome: {run['outcome']}")
                    
                    # Parse and display steps
                    try:
                        steps = json.loads(run['steps'])
                        if steps:
                            print(f"\nSteps ({len(steps)}):")
                            for i, step in enumerate(steps, 1):
                                print(f"  {i}. {step}")
                    except (json.JSONDecodeError, TypeError):
                        if run['steps'] and run['steps'] != '[]':
                            print(f"Steps: {run['steps']}")
                    print()
                else:
                    print(f"Run #{run_id} not found")
            except ValueError:
                print("Invalid run ID")
        else:
            print(f"Unknown run subcommand: {subcmd}")
            print("\nRun commands:")
            print("  memory-tool run start \"task description\" [--agent claw|claude] [--project X] [--tags x,y]")
            print("  memory-tool run step <id> \"step description\"")
            print("  memory-tool run complete <id> \"outcome summary\"")
            print("  memory-tool run fail <id> \"reason\"")
            print("  memory-tool run list [--status running|completed|failed] [--project X] [--limit 10]")
            print("  memory-tool run show <id>")
            print("  memory-tool run cancel <id>")

    elif cmd == "next":
        suggest_next()

    elif cmd in ("help", "--help", "-h"):
        print_help()

    else:
        print(f"Unknown command: {cmd}")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
