"""Database connection and schema initialization."""

import sqlite3
import time
from typing import Optional
from .config import DB_PATH, EMBEDDING_DIM

# Lazy imports for optional dependencies
try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False


def has_vec_support() -> bool:
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


def get_db() -> sqlite3.Connection:
    """Get database connection with proper configuration."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec extension if available
    if has_vec_support():
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (sqlite3.OperationalError, AttributeError) as e:
            import sys
            print(f"[ai-iq WARNING] sqlite-vec extension failed to load: {e}. Semantic search disabled, using text-only fallback.", file=sys.stderr)
            pass  # Extension loading failed or not available

    return conn


def retry_on_busy(func, *args, max_retries: int = 3, backoff_ms: int = 100, **kwargs):
    """Retry a database operation on SQLITE_BUSY errors.

    Provides application-level retry on top of busy_timeout for extra safety
    when multiple agents write concurrently.
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if ("database is locked" in str(e) or "SQLITE_BUSY" in str(e)) and attempt < max_retries:
                time.sleep(backoff_ms / 1000.0)
                continue
            raise


def init_db() -> None:
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
        "confidence": "ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 0.7",
        "content_hash": "ALTER TABLE memories ADD COLUMN content_hash TEXT DEFAULT NULL",
        "proof_count": "ALTER TABLE memories ADD COLUMN proof_count INTEGER DEFAULT 1",
        "source_memory_ids": "ALTER TABLE memories ADD COLUMN source_memory_ids TEXT DEFAULT NULL",
        "wing": "ALTER TABLE memories ADD COLUMN wing TEXT DEFAULT NULL",
        "room": "ALTER TABLE memories ADD COLUMN room TEXT DEFAULT NULL",
        "tier": "ALTER TABLE memories ADD COLUMN tier TEXT DEFAULT 'episodic' CHECK(tier IN ('working', 'episodic', 'semantic'))",
        "last_validated_at": "ALTER TABLE memories ADD COLUMN last_validated_at TEXT DEFAULT NULL",
        "agent_id": "ALTER TABLE memories ADD COLUMN agent_id TEXT DEFAULT NULL",
        "disclosure_condition": "ALTER TABLE memories ADD COLUMN disclosure_condition TEXT DEFAULT NULL",
        "is_pinned": "ALTER TABLE memories ADD COLUMN is_pinned INTEGER DEFAULT 0",
    }

    # Whitelist for beliefs table migrations
    BELIEFS_COLUMN_ADDITIONS = {
        "belief_state": "ALTER TABLE beliefs ADD COLUMN belief_state TEXT DEFAULT 'hypothesis'",
    }

    # Whitelist for runs table migrations (Phase 2: Passport POW)
    RUNS_COLUMN_ADDITIONS = {
        "domain": "ALTER TABLE runs ADD COLUMN domain TEXT DEFAULT NULL",
    }

    # Add columns if upgrading (must run BEFORE triggers reference them)
    for col_name, sql_statement in VALID_COLUMN_ADDITIONS.items():
        try:
            conn.execute(sql_statement)
        except sqlite3.OperationalError:
            pass

    # Migrate beliefs table (must run AFTER beliefs table exists)
    try:
        # Check if beliefs table exists first
        conn.execute("SELECT 1 FROM beliefs LIMIT 1")
        for col_name, sql_statement in BELIEFS_COLUMN_ADDITIONS.items():
            try:
                conn.execute(sql_statement)
            except sqlite3.OperationalError:
                pass
    except sqlite3.OperationalError:
        pass  # beliefs table doesn't exist yet

    # Migrate runs table (must run AFTER runs table exists)
    try:
        conn.execute("SELECT 1 FROM runs LIMIT 1")
        for col_name, sql_statement in RUNS_COLUMN_ADDITIONS.items():
            try:
                conn.execute(sql_statement)
            except sqlite3.OperationalError:
                pass
    except sqlite3.OperationalError:
        pass  # runs table doesn't exist yet

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
            derived_from TEXT DEFAULT NULL,
            citations TEXT DEFAULT NULL,
            reasoning TEXT DEFAULT NULL,
            fsrs_stability REAL DEFAULT 1.0,
            fsrs_difficulty REAL DEFAULT 5.0,
            fsrs_interval REAL DEFAULT 1.0,
            fsrs_reps INTEGER DEFAULT 0,
            last_accessed_at TEXT DEFAULT NULL,
            imp_novelty REAL DEFAULT 5.0,
            imp_relevance REAL DEFAULT 5.0,
            imp_frequency REAL DEFAULT 0.0,
            imp_impact REAL DEFAULT 5.0,
            imp_score REAL DEFAULT 5.0,
            confidence REAL DEFAULT 0.7,
            content_hash TEXT DEFAULT NULL,
            proof_count INTEGER DEFAULT 1,
            source_memory_ids TEXT DEFAULT NULL,
            wing TEXT DEFAULT NULL,
            room TEXT DEFAULT NULL,
            tier TEXT DEFAULT 'episodic' CHECK(tier IN ('working', 'episodic', 'semantic')),
            last_validated_at TEXT DEFAULT NULL,
            agent_id TEXT DEFAULT NULL,
            disclosure_condition TEXT DEFAULT NULL,
            is_pinned INTEGER DEFAULT 0
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
        CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash) WHERE content_hash IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_wing_room ON memories(wing, room) WHERE wing IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier);
        CREATE INDEX IF NOT EXISTS idx_agent_id ON memories(agent_id) WHERE agent_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_pinned ON memories(is_pinned);

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
            import sys
            print(f"[ai-iq WARNING] sqlite-vec extension failed to load: {e}. Semantic search disabled, using text-only fallback.", file=sys.stderr)
            pass  # Silently fail if vec is not available

    # Phase 6: Search feedback tracking table
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            search_type TEXT DEFAULT 'hybrid',
            result_ids TEXT,
            used_ids TEXT,
            result_count INTEGER DEFAULT 0,
            hit_rate REAL,
            latency_ms INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_search_log_query ON search_log(query);
        CREATE INDEX IF NOT EXISTS idx_search_log_created ON search_log(created_at);
    """)

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
            relation_type TEXT NOT NULL CHECK(relation_type IN ('knows','works_on','owns','depends_on','built_by','uses','blocks','related_to','leads_to','prevents','resolves','requires')),
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

        CREATE TABLE IF NOT EXISTS passport_credentials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            credential_json TEXT NOT NULL,
            public_key BLOB NOT NULL,
            issued_at TEXT DEFAULT (datetime('now')),
            expires_at TEXT,
            revoked INTEGER DEFAULT 0
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
        CREATE INDEX IF NOT EXISTS idx_passport_agent ON passport_credentials(agent_id);
        CREATE INDEX IF NOT EXISTS idx_dream_session ON dream_log(session_file);
        CREATE INDEX IF NOT EXISTS idx_corrections_status ON corrections(status);
        CREATE INDEX IF NOT EXISTS idx_corrections_created ON corrections(created_at);
    """)

    # Create index for domain column after ALTER TABLE migrations
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_domain ON runs(domain)")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # domain column doesn't exist yet or index already exists

    # Beliefs system tables (Phase 7)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER REFERENCES memories(id),
            prediction TEXT NOT NULL,
            expected_outcome TEXT,
            confidence REAL DEFAULT 0.5,
            deadline TEXT,
            status TEXT DEFAULT 'open',
            actual_outcome TEXT,
            resolved_at TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS belief_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            prediction_id INTEGER,
            old_confidence REAL,
            new_confidence REAL,
            reason TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);
        CREATE INDEX IF NOT EXISTS idx_predictions_memory ON predictions(memory_id);
        CREATE INDEX IF NOT EXISTS idx_predictions_deadline ON predictions(deadline);
        CREATE INDEX IF NOT EXISTS idx_belief_updates_memory ON belief_updates(memory_id);
        CREATE INDEX IF NOT EXISTS idx_belief_updates_prediction ON belief_updates(prediction_id);
    """)

    # Extended beliefs system tables (explicit beliefs with evidence tracking)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS beliefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER REFERENCES memories(id),
            statement TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            category TEXT DEFAULT 'general',
            evidence_for INTEGER DEFAULT 0,
            evidence_against INTEGER DEFAULT 0,
            source TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            status TEXT DEFAULT 'active',
            belief_state TEXT DEFAULT 'hypothesis' CHECK(belief_state IN ('hypothesis', 'tested', 'validated', 'deprecated', 'refuted'))
        );

        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            belief_id INTEGER REFERENCES beliefs(id),
            memory_id INTEGER REFERENCES memories(id),
            direction TEXT NOT NULL CHECK(direction IN ('supports', 'contradicts')),
            strength REAL DEFAULT 0.5,
            note TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS belief_revisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            belief_id INTEGER REFERENCES beliefs(id),
            old_confidence REAL,
            new_confidence REAL,
            reason TEXT,
            revision_type TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS belief_timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            belief_id INTEGER REFERENCES beliefs(id),
            memory_id INTEGER REFERENCES memories(id),
            old_confidence REAL,
            new_confidence REAL,
            reason TEXT,
            source_type TEXT DEFAULT 'manual',
            timestamp TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_beliefs_category ON beliefs(category);
        CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status);
        CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence);
        CREATE INDEX IF NOT EXISTS idx_beliefs_memory ON beliefs(memory_id);
        CREATE INDEX IF NOT EXISTS idx_beliefs_state ON beliefs(belief_state);
        CREATE INDEX IF NOT EXISTS idx_evidence_belief ON evidence(belief_id);
        CREATE INDEX IF NOT EXISTS idx_evidence_memory ON evidence(memory_id);
        CREATE INDEX IF NOT EXISTS idx_belief_revisions_belief ON belief_revisions(belief_id);
        CREATE INDEX IF NOT EXISTS idx_belief_timeline_belief ON belief_timeline(belief_id);
        CREATE INDEX IF NOT EXISTS idx_belief_timeline_memory ON belief_timeline(memory_id);
        CREATE INDEX IF NOT EXISTS idx_belief_timeline_timestamp ON belief_timeline(timestamp);
    """)

    # User feedback table
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rating TEXT NOT NULL CHECK(rating IN ('good', 'bad', 'meh')),
            reason TEXT,
            session_id TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            linked_memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
        CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_memory ON feedback(linked_memory_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
    """)

    # GDPR/EU AI Act audit log table
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            memory_id INTEGER,
            content_hash TEXT,
            category TEXT,
            project TEXT,
            actor TEXT DEFAULT 'system',
            reason TEXT,
            ip_address TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            retention_until TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type, created_at);
        CREATE INDEX IF NOT EXISTS idx_audit_memory ON audit_log(memory_id);
    """)

    # Drift detection validation table (Phase 7: Reinforced Lies Problem)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS validation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            validator TEXT NOT NULL,
            validation_type TEXT NOT NULL CHECK(validation_type IN ('user', 'external_source', 'llm_check', 'cross_reference')),
            result TEXT NOT NULL CHECK(result IN ('confirmed', 'refuted', 'uncertain')),
            notes TEXT,
            validated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_validation_memory ON validation_log(memory_id);
        CREATE INDEX IF NOT EXISTS idx_validation_date ON validation_log(validated_at);
        CREATE INDEX IF NOT EXISTS idx_validation_result ON validation_log(result);
    """)

    # Contradiction journal table (deferred active-forgetting mechanism)
    from .contradiction import ensure_contradiction_table
    ensure_contradiction_table(conn)

    conn.commit()
    conn.close()
