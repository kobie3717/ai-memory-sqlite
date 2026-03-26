"""Database connection and schema initialization."""

import sqlite3
from .config import DB_PATH, EMBEDDING_DIM

# Lazy imports for optional dependencies
try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False


def has_vec_support():
    """Check if vector search dependencies are available."""
    global _SQLITE_VEC_AVAILABLE
    if not _SQLITE_VEC_AVAILABLE:
        return False

    # Also check other dependencies
    try:
        import numpy
        import onnxruntime
        from tokenizers import Tokenizer
        return True
    except ImportError:
        return False


def get_db():
    """Get database connection with proper configuration."""
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
        except (sqlite3.OperationalError, AttributeError) as e:
            pass  # Extension loading failed or not available

    return conn


def init_db():
    """Initialize database schema with all tables and indexes."""
    conn = get_db()

    # Whitelist of valid column definitions for ALTER TABLE statements
    VALID_COLUMN_ADDITIONS = {
        "accessed_at": "ALTER TABLE memories ADD COLUMN accessed_at TEXT DEFAULT NULL",
        "access_count": "ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0",
        "stale": "ALTER TABLE memories ADD COLUMN stale INTEGER DEFAULT 0",
        "expires_at": "ALTER TABLE memories ADD COLUMN expires_at TEXT DEFAULT NULL",
        "source": "ALTER TABLE memories ADD COLUMN source TEXT DEFAULT 'manual'",
        "topic_key": "ALTER TABLE memories ADD COLUMN topic_key TEXT DEFAULT NULL",
        "revision_count": "ALTER TABLE memories ADD COLUMN revision_count INTEGER DEFAULT 1",
        "derived_from": "ALTER TABLE memories ADD COLUMN derived_from TEXT DEFAULT NULL",
        "citations": "ALTER TABLE memories ADD COLUMN citations TEXT DEFAULT NULL",
        "reasoning": "ALTER TABLE memories ADD COLUMN reasoning TEXT DEFAULT NULL",
        "fsrs_stability": "ALTER TABLE memories ADD COLUMN fsrs_stability REAL DEFAULT 1.0",
        "fsrs_difficulty": "ALTER TABLE memories ADD COLUMN fsrs_difficulty REAL DEFAULT 5.0",
        "fsrs_interval": "ALTER TABLE memories ADD COLUMN fsrs_interval REAL DEFAULT 1.0",
        "fsrs_reps": "ALTER TABLE memories ADD COLUMN fsrs_reps INTEGER DEFAULT 0",
        "last_accessed_at": "ALTER TABLE memories ADD COLUMN last_accessed_at TEXT DEFAULT NULL",
        "imp_novelty": "ALTER TABLE memories ADD COLUMN imp_novelty REAL DEFAULT 5.0",
        "imp_relevance": "ALTER TABLE memories ADD COLUMN imp_relevance REAL DEFAULT 5.0",
        "imp_frequency": "ALTER TABLE memories ADD COLUMN imp_frequency REAL DEFAULT 0.0",
        "imp_impact": "ALTER TABLE memories ADD COLUMN imp_impact REAL DEFAULT 5.0",
        "imp_score": "ALTER TABLE memories ADD COLUMN imp_score REAL DEFAULT 5.0",
    }

    # Add columns if upgrading (must run BEFORE triggers reference them)
    for col_name, sql_statement in VALID_COLUMN_ADDITIONS.items():
        try:
            conn.execute(sql_statement)
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
            revision_count INTEGER DEFAULT 1,
            fsrs_stability REAL DEFAULT 1.0,
            fsrs_difficulty REAL DEFAULT 5.0,
            fsrs_interval REAL DEFAULT 1.0,
            fsrs_reps INTEGER DEFAULT 0,
            last_accessed_at TEXT DEFAULT NULL,
            imp_novelty REAL DEFAULT 5.0,
            imp_relevance REAL DEFAULT 5.0,
            imp_frequency REAL DEFAULT 0.0,
            imp_impact REAL DEFAULT 5.0,
            imp_score REAL DEFAULT 5.0
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
            # Validate EMBEDDING_DIM is an integer before using in SQL
            if not isinstance(EMBEDDING_DIM, int) or EMBEDDING_DIM <= 0:
                raise ValueError(f"Invalid EMBEDDING_DIM: {EMBEDDING_DIM}")
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

        CREATE TABLE IF NOT EXISTS dream_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_file TEXT UNIQUE NOT NULL,
            processed_at TEXT DEFAULT (datetime('now')),
            insights_found INTEGER DEFAULT 0,
            file_size INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text TEXT NOT NULL,
            correction TEXT NOT NULL,
            category TEXT DEFAULT 'preference',
            status TEXT DEFAULT 'pending',
            source TEXT DEFAULT 'user',
            created_at TEXT DEFAULT (datetime('now')),
            applied_at TEXT,
            memory_id INTEGER
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
        CREATE INDEX IF NOT EXISTS idx_dream_session ON dream_log(session_file);
        CREATE INDEX IF NOT EXISTS idx_corrections_status ON corrections(status);
        CREATE INDEX IF NOT EXISTS idx_corrections_created ON corrections(created_at);
    """)

    conn.commit()
    conn.close()
