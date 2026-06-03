# AI-IQ Architecture

**AI-IQ is a biological memory system for AI agents.** It models how human memory works — not as a static database, but as a living system that learns, forgets, consolidates, and adapts over time.

## Philosophy: Memory as Biology, Not Storage

Traditional RAG systems treat memory like a filing cabinet: store documents, retrieve documents. AI-IQ treats memory like a brain: encode experiences, consolidate patterns, prune noise, strengthen useful pathways, and forget what doesn't matter.

This isn't metaphor. Every feature maps to a biological memory mechanism.

---

## Biological Memory Principles

### 1. Hippocampus (Short-Term → Long-Term Transfer)

**Biology**: New experiences enter the hippocampus first. During sleep, important ones transfer to the neocortex for permanent storage. Unimportant ones fade.

**AI-IQ**:
- Session memories (recent additions) start with low `access_count`
- `dream` mode consolidates duplicates, merges similar memories, prunes noise
- Frequently accessed memories (`access_count >= 5`) become immune to decay
- Auto-snapshot on session end captures what happened, stores as consolidated memory

**Commands**: `memory-tool snapshot`, `memory-tool dream`, `memory-tool hot`

---

### 2. REM Sleep (Consolidation & Reconsolidation)

**Biology**: During REM sleep, the brain:
- Replays experiences to find patterns
- Merges similar memories
- Prunes weak connections
- Strengthens important pathways
- Creates new associations (insight)

**AI-IQ**: `dream` mode does the same:
- Scans all memories for 85-95% similarity (reconsolidation threshold)
- Auto-merges duplicates, supersedes outdated info
- Detects contradictions (>80% similarity + negation patterns)
- Normalizes dates, extracts insights from session transcripts
- Runs decay algorithm to flag stale memories

**Implementation**: `memory_tool/dream.py` — processes transcripts, consolidates duplicates, runs semantic similarity checks, applies FSRS decay

**Commands**: `memory-tool dream`

---

### 3. Forgetting Curve (FSRS-Based Decay)

**Biology**: Memories fade over time unless reinforced. The forgetting curve (Ebbinghaus) models exponential decay.

**AI-IQ**: Uses FSRS-6 (Free Spaced Repetition Scheduler) algorithm to predict retention probability:
- Memories have `stability` (how long they last) and `difficulty` (how hard to recall)
- Each access updates stability based on retention probability
- `decay` command flags memories below retention threshold as stale
- Pending items expire after 30 days, general memories after 90 days
- High `access_count` (≥5) grants immunity

**Implementation**: `memory_tool/fsrs.py` — pure implementation of FSRS-6 retention curve

**Commands**: `memory-tool decay`, `memory-tool stale`, `memory-tool gc`

---

### 4. Spreading Activation (Context Discovery)

**Biology**: When you think of "coffee," related concepts like "morning," "caffeine," "mug" activate automatically. This is spreading activation — traversing neural pathways.

**AI-IQ**: Graph intelligence does the same:
- Entities (person, project, feature, concept) form nodes
- Relationships (knows, works_on, depends_on, uses) form edges
- `spread` command traverses graph from seed entity, scores related entities by proximity
- Auto-linking connects memories to relevant entities
- Search can traverse graph to find contextually related memories

**Implementation**: `memory_tool/graph.py` — entity/relationship CRUD, spreading activation algorithm, auto-linking

**Commands**: `memory-tool graph spread <entity> [depth]`, `memory-tool graph auto-link`

---

### 5. Confidence & Belief Updating (Bayesian Learning)

**Biology**: Your brain assigns confidence to beliefs. New evidence updates confidence via Bayesian inference. Strong evidence shifts beliefs; weak evidence nudges them.

**AI-IQ**: Beliefs system with Bayesian updates:
- Beliefs have `confidence` (0.0-1.0, like prior probability)
- Evidence supports or contradicts beliefs
- Evidence has `weight` (0.0-1.0, like likelihood ratio)
- Bayesian formula updates belief confidence when evidence is added
- Predictions track accuracy over time, calibrate future confidence

**Implementation**:
- `memory_tool/beliefs.py` — basic belief tracking (uses `memories.confidence`)
- `memory_tool/beliefs_extended.py` — full Bayesian system with explicit `beliefs` table, evidence tracking, calibration stats

**Commands**: `memory-tool believe`, `memory-tool predict`, `memory-tool resolve`, `memory-tool beliefs --conflicts`

---

### 6. Meta-Cognition (Search Feedback)

**Biology**: Your brain monitors its own performance. If visual memory works better than verbal for you, it strengthens visual pathways. This is meta-learning.

**AI-IQ**: Feedback layer tracks search effectiveness:
- Logs which search mode (keyword/semantic/graph) produced results that were actually used
- Calculates precision/recall/F1 per mode
- RRF fusion combines keyword and semantic results with recency and proof boosting
- Tracks helpful vs unhelpful memories to identify knowledge gaps

**Implementation**: `memory_tool/meta_learning.py` — tracks search feedback, calculates metrics, identifies patterns

**Commands**: `memory-tool feedback-stats`, `memory-tool gaps`, `memory-tool hot`

---

### 7. Passport System (Complete Identity Card)

**Biology**: When you think about a memory, you don't just recall the content — you recall context, emotions, when it happened, who was involved, why it matters. A memory is not just text; it's a web of connections.

**AI-IQ**: Passport System provides complete memory dossier:
- **Core identity**: content, category, project, metadata
- **Graph connections**: linked entities with their relationships (1-hop traversal)
- **Memory relationships**: derived-from chains, related memories, supersedes links
- **Provenance**: citations, reasoning, source memories (full audit trail)
- **Usage patterns**: access count, revisions, FSRS retention state
- **Passport score**: composite 0-10 score from priority (30%), access patterns (20%), proof count (20%), graph connections (15%), recency (15%)
- **Spreading activation**: discovers related entities via graph traversal

Like a traveler's passport that proves identity and history, a memory passport is its complete dossier across all dimensions.

**Implementation**: `memory_tool/passport.py` — aggregates data from memories, graph, relations, calculates composite score, runs spreading activation

**Commands**: `memory-tool passport <id>`

---

### 8. Identity Layer (Self-Model)

**Biology**: Your brain maintains a model of "you" — your preferences, habits, behavioral patterns. This self-model guides decisions.

**AI-IQ**: Identity layer mines memory patterns:
- Scans past decisions, errors, learnings, beliefs for trait patterns
- Example: If you often fix "missing null checks," trait = "tends to forget null checks"
- Builds behavioral profile over time
- Snapshots identity evolution (like developmental psychology milestones)

**Implementation**: `memory_tool/identity.py` — trait discovery from regex patterns, behavioral profiling, identity snapshots

**Commands**: `memory-tool identity discover`, `memory-tool identity snapshot`, `memory-tool identity evolve`

---

### 9. User Model (Honcho-Style Profile)

**Biology**: Your brain maintains a model of others — their preferences, communication style, quirks. This guides how you interact with them.

**AI-IQ**: User model layer builds structured profile:
- Scans memories for identity signals (role, expertise, email)
- Extracts preferences (tools, workflows, communication style)
- Detects patterns (recurring behaviors, common requests)
- Logs corrections (AI errors + correct behavior)
- Captures dialect (shorthand terms, jargon, project vocabulary)

Outputs to `/root/.claude/projects/-root/memory/user-model.md` — a living document that AI agents can read to understand user preferences and adapt behavior.

**Dialectic mode**: Uses Claude API to analyze recent sessions and extract new insights, saving them as memories (category='user_model').

**Implementation**: `memory_tool/user_model.py` — scans memories by category/content patterns, builds structured markdown, optional AI analysis

**Commands**: `memory-tool user-model`, `memory-tool user-model --update [--days N]`

---

### 10. Causal Reasoning (Narrative Construction)

**Biology**: Your brain builds cause-effect stories. "I didn't sleep → I'm tired → I drank coffee → I feel better."

**AI-IQ**: Causal graph edges:
- LEADS_TO (A causes B)
- PREVENTS (A blocks B)
- RESOLVES (A fixes B)
- REQUIRES (A needs B first)

Walk these edges to construct narratives: "Error X → Investigation Y → Decision Z → Fix W."

**Implementation**: `memory_tool/narrative.py` — walks causal edges, constructs chronological stories, generates narrative text

**Commands**: `memory-tool narrative <entity>` (builds cause-effect story about an entity)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CLI (cli.py)                           │
│  Command parser, argument handling, output formatting   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Core (core.py)                              │
│  Central hub — re-exports all operations                │
└───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┘
    │   │   │   │   │   │   │   │   │   │   │   │   │
┌───▼───▼───▼───▼───▼───▼───▼───▼───▼───▼───▼───▼───▼───┐
│  Specialized Modules (26 files)                         │
│                                                          │
│  memory_ops    → CRUD (add/update/delete/get)           │
│  database      → SQLite connection, schema init         │
│  embedding     → Vector embeddings (all-MiniLM-L6-v2)   │
│  graph         → Entities, relationships, spread        │
│  passport      → Complete memory identity card          │
│  beliefs       → Basic belief tracking                  │
│  beliefs_ext   → Bayesian updates, evidence, calib      │
│  dream         → Consolidation, reconsolidation, prune  │
│  fsrs          → FSRS-6 retention curve                 │
│  importance    → Multi-channel scoring (novelty, etc)   │
│  utils         → Auto-tag, similarity, dedup            │
│  relations     → Memory links, conflicts, merge         │
│  snapshots     → Session summaries, project detection   │
│  export        → MEMORY.md generation, topics, backup   │
│  sync          → OpenClaw bidirectional bridge          │
│  runs          → Workflow tracking (steps, timing)      │
│  identity      → Trait discovery, self-model            │
│  user_model    → User profiling, Honcho-style           │
│  meta_learning → Search feedback, metrics tracking      │
│  narrative     → Causal story construction              │
│  feedback      → Search result usage tracking           │
│  corrections   → Captures self-corrections              │
│  display       → Output formatting, help text           │
│  config        → Paths, constants, logger setup         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Storage: SQLite + FTS5 + sqlite-vec + Graph Tables     │
│                                                          │
│  memories          → Main table (content, tags, meta)   │
│  vec_memories      → Vector embeddings (384-dim)        │
│  graph_entities    → Nodes (person, project, etc)       │
│  graph_rels        → Edges (knows, depends_on, etc)     │
│  beliefs           → Explicit beliefs (extended mode)   │
│  predictions       → Testable predictions               │
│  evidence          → Support/contradict beliefs         │
│  identity_traits   → Discovered behavioral patterns     │
│  runs              → Workflow tracking                  │
│  search_feedback   → Usage tracking for meta-learning   │
│  corrections       → Self-correction log                │
└─────────────────────────────────────────────────────────┘
```

---

## Module Reference

### Core Infrastructure

**`config.py`** (61 lines)
- Paths (DB, MEMORY.md, backups, topics)
- Constants (embedding dim, RRF K, similarity thresholds)
- Auto-tag rules (pm2, nginx, auth, etc → tags)
- Logging setup (file + console with rotation)

**`database.py`** (530 lines)
- SQLite connection with optional sqlite-vec extension
- Schema initialization (12 tables: memories, graph, beliefs, etc)
- FTS5 virtual table for keyword search
- Index creation for performance
- Migration support

**`core.py`** (45 lines)
- Re-exports all operations from specialized modules
- Central import point for CLI and tests

**`cli.py`** (1,851 lines)
- Command parser (argparse)
- Routes 50+ commands to appropriate modules
- Output formatting (compact vs verbose)
- Error handling and logging

---

### Memory Operations

**`memory_ops.py`** (783 lines)
- `add_memory()` — Create with auto-tag, embed, smart ingest (detect duplicates)
- `update_memory()` — Modify content, re-tag, re-embed
- `delete_memory()` — Soft delete (sets deleted_at)
- `get_memory()` — Fetch single memory with full detail
- `list_memories()` — Filter by project/category/tag/stale/expired
- `search_memories()` — Hybrid FTS+vector+graph search with RRF fusion
- Smart ingest: SKIP (exact match), UPDATE (topic-key), SUPERSEDE (85%+ similar), CREATE (new)

**`utils.py`** (98 lines)
- `auto_tag()` — Keyword detection (pm2, docker, auth → tags)
- `normalize()` — Lowercase, strip, collapse whitespace
- `word_set()` — Tokenize for overlap calculation
- `similarity()` — SequenceMatcher ratio (0.0-1.0)
- `find_similar()` — Scans all memories for duplicates

**`relations.py`** (198 lines)
- `relate_memories()` — Link two memories (type: related_to, blocks, caused_by)
- `get_conflicts()` — Find 50-85% similar pairs (potential duplicates)
- `merge_memories()` — Combine two into one, delete old
- `supersede_memory()` — Mark old memory as replaced by new one

---

### Search & Retrieval

**`embedding.py`** (277 lines)
- `get_embedding_model()` — Load all-MiniLM-L6-v2 (ONNX runtime)
- `embed_text()` — Single text → 384-dim vector
- `embed_texts_batch()` — Batch processing for efficiency
- `semantic_search()` — Cosine similarity search via sqlite-vec
- `reindex_embeddings()` — Bulk embed all memories

**`importance.py`** (105 lines)
- Multi-channel scoring: novelty (age decay), relevance (tag match), confirmation (access count)
- Each channel scores 0.0-1.0, weighted sum determines final importance
- Used to rank search results and prioritize MEMORY.md export

**`meta_learning.py`** (346 lines)
- Tracks which search mode (keyword/semantic/graph) produced results that were used
- Calculates precision, recall, F1 per mode
- Auto-tunes RRF fusion weights to favor effective modes
- Self-improving search over time

**`feedback.py`** (622 lines)
- `log_search_feedback()` — Records which results were actually used
- `get_search_quality_stats()` — Precision/recall per mode
- Powers meta-learning weight adjustments

---

### Graph Intelligence

**`graph.py`** (500 lines)
- `add_entity()` — Create node (person, project, org, feature, concept, tool, service)
- `add_relationship()` — Create edge (knows, works_on, owns, depends_on, uses, etc)
- `set_fact()` — Key-value metadata on entities (tracks history)
- `get_entity()` — Full entity with facts, relationships, linked memories
- `spreading_activation()` — Traverse graph from seed, score related entities by proximity
- `auto_link_memories()` — Scans content for entity mentions, creates links

**`narrative.py`** (344 lines)
- Walks causal edges (LEADS_TO, PREVENTS, RESOLVES, REQUIRES)
- Constructs chronological cause-effect stories
- Generates narrative text about entities
- Example: "Error X led to Investigation Y, which resolved Bug Z"

---

### Belief & Prediction System

**`beliefs.py`** (576 lines)
- Basic belief tracking using `memories.confidence` field
- `add_belief()` — Create belief with confidence (0.0-1.0)
- `add_prediction()` — Testable prediction with deadline
- `resolve_prediction()` — Mark confirmed/refuted, update belief confidence
- `get_conflicting_beliefs()` — Detects contradictory beliefs (semantic similarity + negation)

**`beliefs_extended.py`** (1,207 lines)
- Advanced Bayesian system with separate `beliefs` table
- Evidence tracking (support/contradict) with weights
- Bayesian confidence updates: `P(belief|evidence) ∝ P(evidence|belief) × P(belief)`
- Calibration stats: expected confidence vs actual accuracy
- Sophisticated contradiction detection

---

### Maintenance & Housekeeping

**`dream.py`** (685 lines)
- **Reconsolidation**: Finds 85-95% similar memories, merges, supersedes old
- **Contradiction detection**: >80% similarity + negation patterns → warns
- **Date normalization**: Extracts dates from content, sets `expires_at`
- **Insight extraction**: Scans session transcripts for learnings
- **Duplicate consolidation**: Merges exact/near-exact duplicates
- Runs decay algorithm after processing
- Like REM sleep — consolidates, prunes, strengthens

**`export.py`** (700 lines)
- `export_memory_md()` — Generates MEMORY.md with 5KB hard cap
- `export_topics()` — Per-project markdown files in `topics/` dir
- `backup_database()` — Timestamped snapshots to `~/backups/memory/`
- `restore_database()` — Restore from backup
- `decay_memories()` — FSRS-based staleness flagging
- `garbage_collect()` — Purge old deleted memories (default 180 days)

**`snapshots.py`** (260 lines)
- `create_snapshot()` — Manual session summary
- `auto_snapshot()` — Detects git changes, file edits, generates summary
- `detect_active_projects()` — Infers project from file paths
- Runs on Stop hook after each session

**`fsrs.py`** (63 lines)
- Pure implementation of FSRS-6 algorithm
- `fsrs_retention()` — Calculate retention probability given stability and elapsed days
- `fsrs_new_stability()` — Update stability after review/access (rating 1-4)
- No dependencies, no side effects — pure functions

---

### Identity & Self-Model

**`identity.py`** (495 lines)
- Trait discovery from memory patterns (regex-based)
- Example patterns: "always forget X", "prefer Y over Z", "struggle with W"
- Behavioral profiling: categorizes decisions, errors, learnings
- Identity snapshots: captures trait evolution over time
- Narrative generation: "I tend to X, but I'm getting better at Y"

---

### Integration & Sync

**`sync.py`** (524 lines)
- Bidirectional sync with OpenClaw (file-based memory system)
- `sync_to_openclaw()` — Export memories to OpenClaw format
- `sync_from_openclaw()` — Import OpenClaw memories
- `sync_bidirectional()` — Two-way sync with conflict detection
- Maintains bridge table to track sync state

**`runs.py`** (167 lines)
- Workflow tracking: start run, add steps, complete run
- Captures timing, outcomes, final state
- Enables pattern analysis: "Deployments usually fail at step 3"

**`corrections.py`** (111 lines)
- Detects when AI corrects itself ("Actually, X should be Y")
- Logs self-corrections with confidence hit
- Used for meta-learning about error patterns

---

### UI & Output

**`display.py`** (435 lines)
- `show_help()` — Full command reference
- `format_memory()` — Compact vs verbose output
- `print_table()` — Pretty-printed tables for lists
- `render_stats()` — Memory system health metrics
- ANSI color support for readability

---

## Data Flow Examples

### 1. Adding a Memory

```
User: memory-tool add decision "Use PostgreSQL for main DB" --project MyApp --tags database,postgres

CLI (cli.py)
  └─> memory_ops.add_memory()
       ├─> utils.auto_tag() → detects "postgres" keyword → adds tag
       ├─> utils.find_similar() → checks for duplicates
       │    └─> If 85%+ match → SUPERSEDE old memory
       ├─> embedding.embed_and_store() → generates 384-dim vector
       │    └─> database.get_db() → stores in vec_memories table
       ├─> graph.auto_link_memories() → scans for entity mentions
       └─> database INSERT → stores in memories table

Result: Memory stored, tagged, embedded, linked to graph
```

---

### 2. Searching for Context

```
User: memory-tool search "auth flow"

CLI (cli.py)
  └─> memory_ops.search_memories(query="auth flow", mode="hybrid")
       ├─> FTS5 keyword search → "auth" OR "flow" → ranked results
       ├─> embedding.semantic_search() → cosine similarity → ranked results
       ├─> graph.spreading_activation() → find related entities → linked memories
       ├─> RRF fusion → combine 3 result sets → final ranking
       │    └─> meta_learning weights adjust fusion based on past success
       ├─> importance scoring → novelty + relevance + confirmation
       └─> feedback.log_search() → track query for meta-learning

Result: Top 10 results, ranked by hybrid relevance
```

---

### 3. Dream Mode Consolidation

```
User: memory-tool dream

CLI (cli.py)
  └─> dream.dream_mode()
       ├─> Scan all memories for 85-95% similarity
       │    └─> embedding.semantic_search() for each memory
       ├─> Merge duplicates → relations.merge_memories()
       ├─> Detect contradictions (>80% + negation patterns) → warn
       ├─> Normalize dates → extract from content → set expires_at
       ├─> Extract insights from session transcripts → add as learnings
       ├─> Run decay → export.decay_memories()
       │    └─> fsrs.fsrs_retention() → flag stale if retention < threshold
       └─> Prune low-value memories → export.garbage_collect()

Result: Database consolidated, duplicates merged, stale flagged
```

---

### 4. Belief Update from Evidence

```
User: memory-tool believe "Redis is faster than PostgreSQL for caching" --confidence 0.6
      memory-tool evidence <belief_id> support "Benchmark: Redis 10x faster" --weight 0.8

CLI (cli.py)
  └─> beliefs_extended.add_evidence()
       ├─> Fetch current belief → confidence = 0.6
       ├─> Apply Bayesian update:
       │    Prior = 0.6
       │    Likelihood ratio = 0.8 (evidence weight)
       │    Posterior = 0.6 * 0.8 / (0.6 * 0.8 + 0.4 * 0.2) = 0.857
       ├─> Update belief confidence → 0.857
       └─> Store evidence in evidence table

Result: Belief confidence increased from 0.6 → 0.857
```

---

### 5. Graph-Based Context Discovery

```
User: memory-tool graph spread FlashVault 2

CLI (cli.py)
  └─> graph.spreading_activation(seed="FlashVault", depth=2)
       ├─> Start at FlashVault entity → score = 1.0
       ├─> Depth 1: Traverse edges (works_on, depends_on, uses)
       │    └─> Find: WireGuard, Overwatch, PostgreSQL → score = 0.7
       ├─> Depth 2: Traverse from depth-1 entities
       │    └─> Find: Unbound DNS, PayFast, React Native → score = 0.5
       ├─> Fetch memories linked to all entities
       └─> Rank by entity score + importance

Result: FlashVault (1.0), WireGuard (0.7), Overwatch (0.7), PostgreSQL (0.7),
        Unbound (0.5), PayFast (0.5), React Native (0.5) + linked memories
```

---

## Integration Points

### Claude Code Hooks

AI-IQ integrates with Claude Code via hooks in `~/.claude/hooks/`:

**PostToolUse Hook** (`error-hook.sh`)
- Triggers after every Bash command
- If exit code ≠ 0 → auto-adds error memory
- Example: `curl fails → "Error: curl timeout on api.example.com" stored as error category`

**Stop Hook** (`session-hook.sh`)
- Triggers when session ends
- Runs `memory-tool auto-snapshot` → detects changes, generates summary
- Runs `memory-tool decay` → flags stale memories
- Runs `memory-tool export` → re-generates MEMORY.md
- Runs `memory-tool backup` → creates timestamped snapshot

**UserPromptSubmit Hook** (optional)
- Detects task type (coding/testing/review) from user input
- Suggests routing to appropriate agent
- Tracks session metrics (edits, commands, reads)

---

### OpenClaw Bridge

AI-IQ syncs with OpenClaw (file-based memory system) via `sync.py`:

- OpenClaw stores memories as markdown files
- `sync-from` imports OpenClaw files → SQLite memories
- `sync-to` exports SQLite memories → OpenClaw files
- `sync` runs bidirectional sync with conflict detection
- Bridge table tracks sync state (last_sync_id)

**Use case**: Share memories between Claude Code (AI-IQ) and other tools (OpenClaw)

---

## Schema Overview

### Core Tables

**memories** (main storage)
- `id`, `content`, `category`, `project`, `tags`, `priority`
- `confidence`, `stability`, `difficulty` (FSRS)
- `access_count`, `last_accessed_at` (usage tracking)
- `created_at`, `updated_at`, `deleted_at` (lifecycle)
- `topic_key` (upsert identifier), `derived_from`, `citations`, `reasoning` (provenance)
- `expires_at`, `stale` (decay system)

**vec_memories** (vector embeddings)
- `memory_id`, `embedding` (384-dim float vector via sqlite-vec)

**graph_entities** (knowledge graph nodes)
- `id`, `type` (person, project, org, feature, concept, tool, service)
- `name`, `summary`, `created_at`

**graph_relationships** (knowledge graph edges)
- `id`, `from_entity`, `to_entity`, `relationship_type`
- Types: knows, works_on, owns, depends_on, built_by, uses, related_to
- Causal types: PREVENTS, RESOLVES, LEADS_TO, REQUIRES

**graph_facts** (entity metadata with history)
- `entity_id`, `key`, `value`, `valid_from`, `valid_to`
- Example: `FlashVault → domain → flashvault.co.za (2024-01-01 → NULL)`

**entity_memory_bridge** (links memories to entities)
- `entity_id`, `memory_id`, `created_at`

---

### Belief System Tables

**beliefs** (explicit beliefs with confidence)
- `id`, `statement`, `confidence`, `project`, `tags`, `created_at`

**predictions** (testable predictions)
- `id`, `statement`, `based_on_belief_id`, `expected_outcome`, `confidence`
- `deadline`, `resolved_at`, `confirmed`, `actual_outcome`

**evidence** (supports/contradicts beliefs)
- `id`, `belief_id`, `direction` (support/contradict), `content`, `weight`, `source`

---

### Auxiliary Tables

**identity_traits** (discovered behavioral patterns)
- `id`, `trait`, `confidence`, `first_seen`, `last_seen`, `occurrences`

**runs** (workflow tracking)
- `id`, `name`, `started_at`, `completed_at`, `outcome`, `final_state`

**run_steps** (workflow steps)
- `id`, `run_id`, `step_number`, `description`, `started_at`, `completed_at`, `outcome`

**search_feedback** (tracks search result usage)
- `id`, `search_id`, `memory_id`, `used` (boolean), `timestamp`

**corrections** (self-correction log)
- `id`, `original_memory_id`, `corrected_content`, `confidence_before`, `confidence_after`, `timestamp`

**openclaw_sync_state** (bridge to OpenClaw)
- `id`, `memory_id`, `openclaw_id`, `last_synced_at`, `sync_direction`

---

## Performance Characteristics

### Storage

- **SQLite file size**: ~500KB for 1000 memories (without vectors)
- **With vectors**: +150KB per 1000 memories (384-dim float32)
- **Indexes**: 3 B-tree indexes (project, category, tags) + 1 FTS5 virtual table
- **Backup size**: Gzip compressed backups ~50KB per 1000 memories

### Search Speed

- **Keyword search (FTS5)**: <10ms for 10,000 memories
- **Semantic search (sqlite-vec)**: 50-100ms for 10,000 memories (cosine similarity)
- **Hybrid search (RRF fusion)**: 100-150ms (both modes + merge)
- **Graph traversal (spreading activation)**: 20-50ms per depth level

### Memory Limits

- **Practical limit**: 100,000 memories (tested to 10,000 in production)
- **MEMORY.md cap**: 5KB hard limit (progressive trimming)
- **Session snapshot**: Auto-generated, max 50 lines
- **Daily backup**: Compressed, retained for 90 days

---

## Testing

- **24 test files** in `tests/` directory
- **268 test functions** covering all modules
- **GitHub Actions CI**: Matrix testing on Python 3.10, 3.11, 3.12
- **Coverage**: 100% on core modules (fsrs, importance, utils, config)
- **Test DB**: Uses `:memory:` SQLite database (no file I/O)

Run tests: `python -m pytest`

---

## Future Directions

**Planned features** (not yet implemented):

1. **Obsidian vault import** — Parse markdown files with frontmatter
2. **CSV export** — Bulk export for analysis in spreadsheets
3. **Multi-user support** — Shared memories with access control
4. **Temporal reasoning** — "What did I know on 2024-01-15?"
5. **Visualization** — Graph viewer, memory timeline, belief network
6. **Audio/video memories** — Transcribe and store with timestamps
7. **Differential backups** — Only store changes since last backup

**Experimental ideas** (research):

- Emotional tagging (positive/negative/neutral sentiment)
- Memory importance learned from LLM feedback ("Was this useful?")
- Cross-project pattern detection ("Same bug in 3 projects")
- Automatic skill assessment ("How good am I at React?")

---

## Design Decisions

### Why SQLite?

- Zero configuration, no server
- ACID transactions (safe concurrent access)
- FTS5 full-text search built-in
- sqlite-vec extension for vectors
- Fast enough for 100K+ memories
- Single file, easy to backup
- Python stdlib support (no dependencies)

### Why FSRS-6?

- Scientific basis (spaced repetition research)
- Proven in Anki (millions of users)
- Pure algorithm, no training needed
- Models forgetting curve accurately
- Allows predictive staleness detection

### Why Hybrid Search?

- Keyword search: fast, precise, exact matches
- Semantic search: handles synonyms, conceptual matches
- Graph traversal: discovers related context
- RRF fusion: no parameter tuning, provably good
- Meta-learning: adapts weights to user behavior

### Why Beliefs?

- Captures uncertainty (not all info is equally certain)
- Bayesian updates allow evidence accumulation
- Predictions enable testing and calibration
- Detects contradictions before they cause bugs
- Models how humans actually reason

### Why Identity Layer?

- AI assistants need self-awareness ("How do I usually solve X?")
- Behavioral patterns guide decisions ("I tend to forget null checks")
- Identity evolves over time (like human personality)
- Narrative construction creates explanations

---

## Comparison to Alternatives

**vs Traditional RAG**
- RAG: Retrieve documents, stuff into context
- AI-IQ: Retrieve relevant context, ranked by importance + freshness + usage

**vs Vector Databases (Pinecone, Weaviate)**
- They: Cloud-hosted, API keys, embeddings only
- AI-IQ: Local, no keys, hybrid FTS+vector+graph

**vs `.claude/memory`**
- Claude: Project-specific, no search, no decay
- AI-IQ: Cross-project, hybrid search, auto-decay, beliefs, graph

**vs Cursor Rules**
- Cursor: Static rules, no learning
- AI-IQ: Dynamic memories, self-learning, predictions

**vs OpenClaw**
- OpenClaw: File-based, manual organization
- AI-IQ: Database-backed, auto-organization, search

**vs Notion/Obsidian**
- Them: Manual note-taking
- AI-IQ: Automated capture (hooks), AI-optimized (embeddings, graph)

---

## Summary

AI-IQ is not a database. It's a **cognitive architecture** for AI agents, modeled on biological memory:

- **Encode** experiences (add memories with auto-tagging, embedding)
- **Consolidate** patterns (dream mode merges, prunes, strengthens)
- **Retrieve** context (hybrid search finds what's relevant now)
- **Learn** from usage (meta-learning tunes search, beliefs update from evidence)
- **Forget** noise (decay flags stale, GC purges old)
- **Reason** causally (graph edges model cause-effect, narrative constructs stories)
- **Self-model** (identity layer discovers behavioral traits)

This creates a memory system that **gets smarter over time** — just like you.
