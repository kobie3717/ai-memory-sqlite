# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Persistent Memory System

This project uses [ai-memory-sqlite](https://github.com/kobie3717/ai-memory-sqlite) for persistent memory across sessions.

**Memory location**: `~/.claude/projects/<project-name>/memory/`
- **memories.db** - SQLite database (source of truth)
- **MEMORY.md** - Auto-generated summary (auto-loaded every session)

### Core Commands (run via Bash tool)

```bash
# Add memories
memory-tool add <category> "<content>" --project <name> --tags tag1,tag2 --priority 0-10

# Search (hybrid: FTS + semantic + graph)
memory-tool search "<query>"              # Default hybrid search
memory-tool search "<query>" --full       # Verbose output
memory-tool get <id>                      # Full detail for single memory

# List & filter
memory-tool list --project <name>         # Filter by project
memory-tool list --category decision      # Filter by category
memory-tool pending                       # Show TODO items

# Update & delete
memory-tool update <id> "<new content>"
memory-tool delete <id>

# Maintenance
memory-tool next                          # Smart suggestions: what needs attention
memory-tool dream                         # AI-powered consolidation & cleanup
memory-tool conflicts                     # Find potential duplicates
memory-tool snapshot "<summary>"          # Manual session summary
```

**Categories**: `project`, `decision`, `preference`, `error`, `learning`, `pending`, `architecture`, `workflow`, `contact`

### Auto-Update Rules (FOLLOW THESE)

**Session start**: MEMORY.md is auto-loaded. Check "Last Session" and "Pending" sections.

**During work**: Add memories when you:
- Make architecture decisions → `memory-tool add decision`
- Learn project conventions → `memory-tool add learning`
- Hit errors (auto-captured by PostToolUse hook) → no manual action needed
- User says "remember X" → `memory-tool add` with appropriate category
- Create TODOs → `memory-tool add pending --expires YYYY-MM-DD`

**Session end**: Auto-handled by Stop hook (snapshot + decay + export + backup).

**When completing pending items**: `memory-tool delete <id>` to remove.

**When user says "forget X"**: `memory-tool search` then `memory-tool delete <id>`.

### Automation (runs without manual intervention)

- **PostToolUse hook**: Auto-captures failed Bash commands as error memories
- **Stop hook**: Auto-generates session snapshot from git/file changes, runs decay, re-exports MEMORY.md
- **Daily cron** (3:17 AM): Maintenance (decay, garbage collection, backup)

---

# Add your project-specific instructions here

## Project Overview
<!-- Describe your project, stack, and architecture -->

## Key Components
<!-- List important directories, files, and their purpose -->

## Development Commands
<!-- Common commands: build, test, deploy, etc. -->

## Configuration
<!-- Environment variables, config files, secrets -->

## Common Tasks
<!-- Frequent workflows and how to execute them -->
