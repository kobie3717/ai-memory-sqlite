# ai-memory-sqlite

**Persistent memory for AI coding assistants. Stop repeating yourself every session.**

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![SQLite 3.37+](https://img.shields.io/badge/sqlite-3.37%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Zero Dependencies](https://img.shields.io/badge/core-zero%20dependencies-green)

## What's New in v5

**Major upgrades** from previous versions:

- **Run Tracking (Phase 5)** - Structured workflow tracking with steps, timing, and outcomes
- **Provenance System (Phase 6)** - Track memory derivation with `--derived-from`, `--citations`, `--reasoning`
- **Next Actions** - Smart suggestions for what needs attention (expiring memories, conflicts, stale items)
- **Enhanced Graph** - Spreading activation, auto-linking, OpenClaw import
- **Improved Hybrid Search** - Better RRF fusion, graph-aware semantic search
- **Configurable Paths** - Environment variables for all paths, no hardcoded values
- **Production Tested** - Managing 100+ memories across 7 real-world projects

## The Problem

Every time you start a new session with Claude Code, Cursor, or any AI assistant, it forgets your project context. You waste time re-explaining architecture decisions, project conventions, and lessons learned. Your AI starts from zero, every single time.

## The Solution

`ai-memory-sqlite` gives your AI assistant a persistent, searchable memory that survives across sessions:

- **Persistent memory across sessions** - SQLite-backed storage that never forgets
- **Hybrid search** - Combines keyword (FTS5), semantic embeddings, and graph traversal with RRF fusion
- **Smart deduplication** - Blocks duplicates automatically, warns on similar entries
- **Graph intelligence** - Entities, relationships, facts, and spreading activation for context discovery
- **Automated hooks** - Auto-captures errors and generates session snapshots
- **Cross-tool sync** - Share memory between Claude Code, OpenClaw, and other AI tools
- **Zero config, single file** - No API keys, no cloud services, 100% local and private

## Quick Start

```bash
# Clone and install
git clone https://github.com/kobie3717/ai-memory-sqlite.git
cd ai-memory-sqlite
./scripts/install.sh

# Add your first memory
memory-tool add learning "Our API uses Express + PostgreSQL" --project MyApp

# Search memories
memory-tool search "database"

# View all commands
memory-tool --help
```

## Features

### 1. Hybrid Search (FTS + Semantic + RRF)

Combines three search strategies with Reciprocal Rank Fusion:

- **Keyword search** - Fast FTS5 full-text search
- **Semantic search** - all-MiniLM-L6-v2 embeddings (384-dim) via sqlite-vec
- **Graph traversal** - Spreading activation across entity relationships

```bash
memory-tool search "authentication flow"        # Hybrid (default)
memory-tool search "auth" --semantic            # Semantic-only
memory-tool search "jwt token" --keyword        # FTS-only
```

### 2. Graph Intelligence

Build a knowledge graph of entities (people, projects, tools, concepts) with relationships and facts.

```bash
# Add entities
memory-tool graph add project WhatsAuction "Real-time auction platform"
memory-tool graph add tool PostgreSQL "Primary database"

# Create relationships
memory-tool graph rel WhatsAuction uses PostgreSQL

# Set facts
memory-tool graph fact PostgreSQL version "14.5"

# Spreading activation (find related context)
memory-tool graph spread WhatsAuction --depth 2
```

### 3. Smart Ingestion with Conflict Detection

Automatically detects duplicates and conflicts using similarity scoring:

- **> 85% similar** - Blocks as duplicate
- **65-85% similar** - Warns and suggests merge/update
- **< 65% similar** - Adds as new memory

```bash
memory-tool conflicts                    # Find potential duplicates
memory-tool merge <id1> <id2>           # Merge similar memories
memory-tool supersede <old_id> <new_id> # Mark as superseded
```

### 4. Automated Hooks (Claude Code Integration)

- **PostToolUse hook** - Auto-captures failed Bash commands as error memories
- **Stop hook** - Generates session snapshot from git/file changes, runs decay, exports MEMORY.md
- **Daily cron** - Maintenance at 3:17 AM (decay, garbage collection, backup)

### 5. Cross-Tool Sync (OpenClaw Bridge)

Bidirectional sync with OpenClaw's workspace format:

```bash
memory-tool sync          # Two-way sync
memory-tool sync-to       # Claude Code → OpenClaw
memory-tool sync-from     # OpenClaw → Claude Code
```

Share context seamlessly between AI assistants.

## Command Reference

### Core Operations

```bash
memory-tool add <category> "<content>" [options]
  --project <name>         # Associate with project
  --tags <tag1,tag2>       # Add tags (auto-tags also applied)
  --priority <0-10>        # Priority (default: 0)
  --related <id>           # Link to related memory
  --expires <YYYY-MM-DD>   # Auto-expire date
  --key <topic-key>        # Upsert key for topics

memory-tool update <id> "<new content>"
memory-tool delete <id>
memory-tool get <id>                      # Full detail view
memory-tool tag <id> <tag1,tag2>
memory-tool relate <id1> <id2> [type]
```

**Categories**: `project`, `decision`, `preference`, `error`, `learning`, `pending`, `architecture`, `workflow`, `contact`

### Search & Discovery

```bash
memory-tool search "<query>" [--full] [--semantic] [--keyword]
memory-tool list [--category X] [--project X] [--tag X] [--stale] [--expired]
memory-tool projects                      # Project summary
memory-tool topics                        # Generate topic .md files
memory-tool pending                       # Show TODO items
memory-tool conflicts                     # Find duplicates
```

### Graph Operations

```bash
memory-tool graph                         # Show summary
memory-tool graph add <type> <name> [summary]
memory-tool graph rel <from> <rel_type> <to> [note]
memory-tool graph fact <entity> <key> <value>
memory-tool graph get <name>
memory-tool graph list [type]
memory-tool graph delete <name>
memory-tool graph spread <name> [depth]   # Spreading activation
memory-tool graph link <memory_id> <entity>
memory-tool graph auto-link               # Auto-link all memories
memory-tool graph import-openclaw
memory-tool graph stats
```

**Entity types**: `person`, `project`, `org`, `feature`, `concept`, `tool`, `service`

**Relationship types**: `knows`, `works_on`, `owns`, `depends_on`, `built_by`, `uses`, `blocks`, `related_to`

### Session Management

```bash
memory-tool snapshot "<summary>" [--project X]
memory-tool auto-snapshot                 # Auto-detect from git/file changes
memory-tool snapshots [--limit N]
memory-tool detect-project                # Auto-detect from cwd
```

### Maintenance

```bash
memory-tool decay                         # Flag stale, deprioritize, expire
memory-tool stale                         # Review stale memories
memory-tool gc [days]                     # Garbage collect (default: 180 days)
memory-tool reindex                       # Rebuild vector embeddings
memory-tool stats                         # Full statistics
memory-tool backup                        # Manual backup
memory-tool restore <file>
```

### Cross-Tool Sync

```bash
memory-tool sync                          # Bidirectional sync
memory-tool sync-to                       # Export to OpenClaw
memory-tool sync-from                     # Import from OpenClaw
```

### Import/Export

```bash
memory-tool export [--project X]          # Regenerate MEMORY.md
memory-tool import-md <file>              # Import session summary markdown
memory-tool log-error <command> <error>   # Log failed command
```

### Run Tracking

Track structured workflows and agent runs with steps, outcomes, and timing:

```bash
# Start a new run
memory-tool run start "Fix user authentication bug" --agent claude --project MyApp

# Add steps as you work
memory-tool run step 1 "Identified issue in JWT validation"
memory-tool run step 1 "Updated auth middleware"  
memory-tool run step 1 "Added unit tests"

# Complete the run
memory-tool run complete 1 "Fixed auth bug, all tests passing"

# List runs
memory-tool run list                      # Recent runs
memory-tool run list --status running    # Active runs only
memory-tool run list --project MyApp --limit 20

# View detailed run information
memory-tool run show 1                   # Full run details with all steps

# Manage runs
memory-tool run cancel 2                 # Cancel a run
memory-tool run fail 3 "Unable to reproduce bug" # Mark as failed
```

**Run statuses**: `running`, `completed`, `failed`, `cancelled`

**Use cases**:
- Track multi-step debugging sessions
- Document feature implementation workflows  
- Monitor agent task progress
- Capture development decision trails
- Generate workflow reports for team reviews

## Integration

### Claude Code

See [INSTALLATION.md](INSTALLATION.md) for full setup instructions.

1. Install hooks in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": "~/.claude/projects/-root/memory/error-hook.sh",
    "Stop": "~/.claude/projects/-root/memory/session-hook.sh"
  }
}
```

2. Add cron job for daily maintenance:

```bash
17 3 * * * /root/.claude/projects/-root/memory/daily-maintenance.sh >> /root/.claude/projects/-root/memory/cron.log 2>&1
```

3. The system auto-loads `MEMORY.md` at session start via project instructions.

### Other AI Tools

For generic integration:

1. Export memories to markdown: `memory-tool export --project YourProject`
2. Include `MEMORY.md` in your AI tool's context
3. Use `memory-tool add` to capture learnings during sessions
4. Use `memory-tool auto-snapshot` to summarize sessions

## Architecture

**Database**: SQLite 3.37+ with extensions
- **memories** - Core memory storage
- **memories_fts** - FTS5 full-text search index
- **memory_vec** - sqlite-vec vector embeddings (384-dim)
- **memory_relations** - Bidirectional memory links
- **session_snapshots** - Session summaries
- **runs** - Structured workflow/task tracking with steps and timing
- **graph_entities** - Knowledge graph nodes
- **graph_relationships** - Knowledge graph edges
- **graph_facts** - Entity key-value metadata

**Hybrid Search**: RRF fusion (k=60) of keyword + semantic + graph results

**Embedding Model**: all-MiniLM-L6-v2 (ONNX, 384 dimensions)

**Auto-Tagging**: Content analysis detects keywords (pm2, nginx, react, auth, etc.)

**Decay System**:
- Pending items stale after 30 days
- General memories stale after 90 days
- Priority decreases after 60 days without access
- Expired memories auto-deactivate

See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details.

## Comparison

| Feature | ai-memory-sqlite | Mem0 | Engram | LedgerMind |
|---------|-----------------|------|---------|------------|
| Self-hosted | ✅ | ❌ (cloud) | ✅ | ✅ |
| Zero config | ✅ | ❌ | ✅ | ❌ |
| Hybrid search | ✅ | ✅ | ❌ | ❌ |
| Knowledge graph | ✅ | ❌ | ❌ | ❌ |
| Vector search | ✅ (optional) | ✅ | ✅ | ❌ |
| Auto-hooks | ✅ | ❌ | ❌ | ❌ |
| Cross-tool sync | ✅ | ❌ | ❌ | ❌ |
| Privacy | 100% local | Cloud API | 100% local | 100% local |

## Requirements

**Core** (zero dependencies):
- Python 3.8+
- SQLite 3.37+ (with FTS5)

**Optional** (for semantic search):
```bash
pip install -r optional-requirements.txt
```
- numpy >= 1.21.0
- onnxruntime >= 1.14.0
- tokenizers >= 0.13.0
- sqlite-vec >= 0.1.0
- huggingface-hub >= 0.14.0

Semantic search downloads all-MiniLM-L6-v2 model (~90MB) on first use.

## Project Status

**Active development.** Used in production for multi-project AI development workflows.

Real-world stats: 100+ memories, 7 projects, 17 entities, 21 relationships, 109 embeddings, zero data loss over 6 months of daily use.

## Credits

Inspired by research on 25+ open-source memory systems:

- **Engram** - Temporal decay and graph-based memory
- **LedgerMind** - Sequential memory with branching
- **Vestige** - Semantic clustering and decay
- **OpenClaw** - Multi-tool workspace sync
- **Sediment** - Layered memory architecture

Built for real-world use with Claude Code in production environments.

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Issues and pull requests welcome. See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details.

---

**Stop repeating yourself. Give your AI a memory.**
