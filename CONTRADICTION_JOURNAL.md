# Contradiction Journal

## Overview

The Contradiction Journal is a deferred active-forgetting mechanism that prevents premature deletion of memories when the AI encounters contradictions. Instead of immediately deleting memories on first contradiction (like `validate refute`), it logs contradictions and reconciles them at task boundaries.

## Why This Matters

**Problem**: Claude might misunderstand a correct memory and mark it as wrong, destroying valid knowledge.

**Solution**: Multiple independent contradictions (from different execution contexts) = proof the memory is actually wrong.

## How It Works

### 1. Log Contradictions (Mid-Execution Safe)

When the AI finds a memory is wrong during task execution:

```bash
memory-tool contradict <memory_id> \
  --expected "what the memory claimed" \
  --found "what was actually observed" \
  [--context-hash <hash>]
```

- Safe to call mid-execution
- Does NOT delete the memory
- Logs the contradiction with a context hash
- Context hash auto-generated if not provided (from session/environment)

### 2. Reconcile (Task Boundary)

Run at task boundaries (automatically via Stop hook):

```bash
memory-tool reconcile [--dry-run]
```

**Reconciliation Logic**:

- **2+ distinct context hashes** → Soft-delete memory (set `active=0`)
- **1 context hash** → Flag for review (add `contradiction-flagged` tag)
- **Already inactive/deleted** → Skip and mark as reconciled

### 3. View Contradictions

```bash
memory-tool contradictions [--limit N]
```

Shows unreconciled contradictions with:
- Memory content
- Expected vs Found values
- Context hash
- Session ID
- Summary statistics by memory

## Database Schema

```sql
CREATE TABLE contradiction_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    expected TEXT,
    found TEXT,
    context_hash TEXT NOT NULL,
    session_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    reconciled INTEGER DEFAULT 0
);
```

## Integration

### Stop Hook

The reconcile command is automatically called in the Stop hook:

```bash
# /root/.claude/projects/-root/memory/session-hook.sh
memory-tool reconcile 2>/dev/null
```

This ensures contradictions are reconciled at natural task boundaries.

### Context Hash Generation

Context hashes identify independent execution contexts:

1. **Explicit**: Pass `--context-hash` for manual control
2. **Session ID**: Uses `CLAUDE_SESSION_ID` or `TERM_SESSION_ID` env var
3. **Fallback**: Hash of `hostname + current_hour`

The fallback groups contradictions within the same hour, which is sufficient for detecting independent sessions.

## Commands Reference

### contradict

Log a contradiction without deleting:

```bash
memory-tool contradict <memory_id> [options]

Options:
  --expected TEXT       What the memory claimed
  --found TEXT          What was actually observed
  --context-hash HASH   Unique identifier for this execution context
```

### reconcile

Reconcile contradictions at task boundary:

```bash
memory-tool reconcile [--dry-run]

Options:
  --dry-run   Show what would happen without making changes
```

### contradictions

View unreconciled contradictions:

```bash
memory-tool contradictions [--limit N]

Options:
  --limit N   Maximum number of entries to show (default: 20)
```

## Example Workflow

```bash
# 1. AI encounters wrong memory during task execution
memory-tool contradict 123 \
  --expected "Python uses semicolons" \
  --found "Python uses newlines/indentation"

# 2. View logged contradictions
memory-tool contradictions

# 3. At task boundary, reconcile (automatic via Stop hook)
memory-tool reconcile

# Result: Memory #123 flagged for review (only 1 context)

# 4. Later, different context contradicts same memory
memory-tool contradict 123 \
  --expected "semicolons" \
  --found "indentation" \
  --context-hash "different_session_456"

# 5. Reconcile again
memory-tool reconcile

# Result: Memory #123 soft-deleted (2+ distinct contexts)
```

## Implementation Details

### Soft Delete

Memories are NEVER hard-deleted. `reconcile` sets `active=0`:

```sql
UPDATE memories SET active = 0 WHERE id = ?
```

This preserves audit trails and allows recovery if needed.

### Contradiction Journal Retention

Contradiction journal entries are NEVER hard-deleted. They're marked as `reconciled=1` when processed:

```sql
UPDATE contradiction_journal SET reconciled = 1 WHERE memory_id = ?
```

This maintains a complete history of why memories were deleted.

### Flag for Review

When a memory has contradictions from only 1 context, it's flagged:

```sql
UPDATE memories SET tags = tags || ',contradiction-flagged' WHERE id = ?
```

You can then manually review:

```bash
memory-tool list --tag contradiction-flagged
```

## Comparison with `validate refute`

| Feature | `validate refute` | `contradict + reconcile` |
|---------|-------------------|--------------------------|
| **Timing** | Immediate demotion | Deferred to task boundary |
| **Protection** | None | Requires 2+ independent contradictions |
| **Recovery** | Manual | Automatic flagging for review |
| **Audit** | Limited | Full contradiction history |
| **Use Case** | User-confirmed wrong | AI-detected potential wrong |

## Testing

Run the test suite:

```bash
bash /tmp/test_contradiction_journal.sh
```

This creates a test memory, logs contradictions from multiple contexts, and verifies the reconciliation logic works correctly.

## Files

- `/root/ai-iq/memory_tool/contradiction.py` - Core implementation
- `/root/ai-iq/memory_tool/database.py` - Table creation
- `/root/ai-iq/memory_tool/cli.py` - Command handlers
- `/root/.claude/projects/-root/memory/session-hook.sh` - Stop hook integration
