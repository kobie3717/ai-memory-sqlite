#!/usr/bin/env python3
"""
Claude Code Persistent Memory System v5 + FSRS-6 Spaced Repetition
SQLite-backed with FTS, dedup, relationships, FSRS decay, smart context, auto-snapshots,
auto-tagging, expiry, error capture hook, backup/restore, progressive disclosure,
topic-key upserts, conflict detection, smart ingest, topic file export.
Phase 2: Hybrid search with semantic embeddings (sqlite-vec) + RRF fusion.
Phase 3: Graph intelligence with entities, relationships, facts, and spreading activation.
Phase 6: FSRS-6 spaced repetition model for intelligent memory decay and retention tracking.

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
  memory-tool dream                             # Review transcripts, consolidate memories, normalize dates (AI memory REM sleep)
  memory-tool capture-correction "<text>"       # Extract and store corrections from user feedback
  memory-tool correct "<text>"                  # Queue a correction manually
  memory-tool corrections                       # Show pending corrections
  memory-tool apply-correction <id>             # Apply correction as memory
  memory-tool dismiss-correction <id>           # Dismiss a correction
  memory-tool detect "<text>"                   # Detect correction in text
  memory-tool stale                             # Review stale memories
  memory-tool decay                             # Flag stale, deprioritize, expire (FSRS-6)
  memory-tool consolidate                       # Cross-memory consolidation (merge, patterns, prune)
  memory-tool retention                         # Show memories by retention (lowest first)
  memory-tool importance                        # Show memories ranked by importance score
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

if __name__ == "__main__":
    # Import and run the CLI
    from memory_tool.cli import main
    main()
