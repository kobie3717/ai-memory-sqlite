# Installation Guide

Complete installation instructions for ai-memory-sqlite.

## Quick Install

For Claude Code users on Linux/macOS:

```bash
git clone https://github.com/kobie3717/ai-memory-sqlite.git
cd ai-memory-sqlite
./scripts/install.sh
```

This will:
1. Copy files to `~/.claude/projects/-root/memory/`
2. Install `memory-tool` to `/usr/local/bin/`
3. Set up hooks in `~/.claude/settings.json`
4. Add cron job for daily maintenance
5. Create backup directory

Verify installation:

```bash
memory-tool stats
memory-tool add learning "Test memory" --project Test
memory-tool search "test"
```

## Manual Install

### Step 1: Clone Repository

```bash
git clone https://github.com/kobie3717/ai-memory-sqlite.git
cd ai-memory-sqlite
```

### Step 2: Choose Installation Location

For Claude Code:
```bash
INSTALL_DIR="$HOME/.claude/projects/-root/memory"
```

For other tools:
```bash
INSTALL_DIR="$HOME/.config/ai-memory"
```

### Step 3: Copy Files

```bash
mkdir -p "$INSTALL_DIR"
cp src/memory-tool.py "$INSTALL_DIR/"
cp scripts/*.sh "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR"/*.sh
chmod +x "$INSTALL_DIR/memory-tool.py"
```

### Step 4: Install CLI Command

```bash
sudo tee /usr/local/bin/memory-tool > /dev/null <<'EOF'
#!/bin/bash
python3 $HOME/.claude/projects/-root/memory/memory-tool.py "$@"
EOF

sudo chmod +x /usr/local/bin/memory-tool
```

Adjust path if you used a different `INSTALL_DIR`.

### Step 5: Verify Installation

```bash
memory-tool stats
# Should show: Memories: 0 total
```

## Optional: Semantic Search

Semantic search requires additional Python dependencies.

### Install Dependencies

```bash
pip install -r optional-requirements.txt
```

Or manually:

```bash
pip install numpy onnxruntime tokenizers sqlite-vec huggingface-hub
```

### Download Embedding Model

On first use with `--semantic`, the model downloads automatically (~90MB):

```bash
memory-tool search "test" --semantic
# Downloads all-MiniLM-L6-v2 to ~/.cache/models/all-MiniLM-L6-v2/
```

Or pre-download:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    allow_patterns=["tokenizer.json", "onnx/model.onnx"],
    local_dir="~/.cache/models/all-MiniLM-L6-v2"
)
```

### Build Embeddings Index

Generate embeddings for existing memories:

```bash
memory-tool reindex
```

This can take a few seconds for 100+ memories.

### Verify Semantic Search

```bash
memory-tool search "authentication" --semantic
memory-tool stats
# Should show: Vector index: X/Y embeddings
```

## Claude Code Integration

### Hook Configuration

Edit `~/.claude/settings.json` to add hooks:

```json
{
  "hooks": {
    "PostToolUse": "~/.claude/projects/-root/memory/error-hook.sh",
    "Stop": "~/.claude/projects/-root/memory/session-hook.sh"
  }
}
```

If `settings.json` doesn't exist, create it:

```bash
mkdir -p ~/.claude
cat > ~/.claude/settings.json <<'EOF'
{
  "hooks": {
    "PostToolUse": "~/.claude/projects/-root/memory/error-hook.sh",
    "Stop": "~/.claude/projects/-root/memory/session-hook.sh"
  }
}
EOF
```

### What Each Hook Does

**PostToolUse** (`error-hook.sh`):
- Runs after every Bash command
- Captures failed commands (exit code != 0)
- Auto-creates error memories

**Stop** (`session-hook.sh`):
- Runs when session ends
- Generates session snapshot from git/file changes
- Runs decay algorithm
- Regenerates MEMORY.md
- Creates daily backup

### Project Instructions (CLAUDE.md)

For auto-loading MEMORY.md at session start, add to your project's `CLAUDE.md`:

```markdown
## Persistent Memory System

SQLite-backed memory at `~/.claude/projects/-root/memory/`:
- **memories.db** - Source of truth (SQLite + FTS5)
- **MEMORY.md** - Auto-loaded context (5KB budget)
- **memory-tool** - CLI for managing memories

### Usage
- Session start: Review MEMORY.md "Last Session" and "Pending"
- During work: `memory-tool add <category> "<content>"` for important info
- Errors auto-captured by PostToolUse hook
- Session end auto-handled by Stop hook

### Commands
```bash
memory-tool search "<query>"           # Hybrid search
memory-tool add learning "<content>"   # Add memory
memory-tool pending                    # Show TODOs
memory-tool stats                      # View statistics
```
```

Claude Code will include `MEMORY.md` in context automatically.

### Daily Maintenance Cron

Add cron job for daily maintenance (decay + backup + garbage collection):

```bash
crontab -e
```

Add this line:

```cron
17 3 * * * $HOME/.claude/projects/-root/memory/daily-maintenance.sh >> $HOME/.claude/projects/-root/memory/cron.log 2>&1
```

This runs at 3:17 AM daily.

**What it does** (`daily-maintenance.sh`):

```bash
#!/bin/bash
memory-tool decay 2>/dev/null
memory-tool gc 180 2>/dev/null  # Delete memories inactive >180 days
memory-tool backup 2>/dev/null
memory-tool export 2>/dev/null
```

## Generic AI Tool Integration

For tools without hook support (Cursor, Continue, etc.):

### Manual Context Loading

1. Generate MEMORY.md:
   ```bash
   memory-tool export --project YourProject
   ```

2. Include in your tool's context/instructions file

3. Reference memories during work:
   ```bash
   memory-tool search "relevant topic"
   ```

### Session Workflow

**Session start**:
```bash
memory-tool export --project YourProject
memory-tool pending
# Review MEMORY.md and pending items
```

**During work**:
```bash
memory-tool add decision "Decided to use PostgreSQL for persistence"
memory-tool add error "npm build fails with ESM import error"
```

**Session end**:
```bash
memory-tool snapshot "Implemented user authentication with JWT" --project YourProject
memory-tool decay
memory-tool backup
```

### Automation with Aliases

Add to `~/.bashrc` or `~/.zshrc`:

```bash
alias mem='memory-tool'
alias mem-start='memory-tool export && memory-tool pending'
alias mem-end='memory-tool auto-snapshot && memory-tool decay && memory-tool backup'
```

Usage:
```bash
mem-start             # Start session
mem add learning "..." # During work
mem search "auth"     # Find memories
mem-end               # End session
```

## OpenClaw Bridge Setup

For cross-tool sync with OpenClaw:

### Prerequisites

- OpenClaw installed at `/root/.openclaw/`
- Workspace directory at `/root/.openclaw/workspace/`

### Enable Sync

Create OpenClaw memory directory:

```bash
mkdir -p /root/.openclaw/workspace/memory
```

Run initial sync:

```bash
memory-tool sync
```

This creates:
- `/root/.openclaw/workspace/memory/claude-code-bridge.md`
- `/root/.openclaw/workspace/memory/graph.json`
- `/root/.openclaw/workspace/memory/topics/*.json`

### Sync Workflow

**Export to OpenClaw**:
```bash
memory-tool sync-to
```

**Import from OpenClaw**:
```bash
memory-tool sync-from
```

**Bidirectional sync**:
```bash
memory-tool sync
```

### Automatic Sync

Add to Stop hook (`session-hook.sh`):

```bash
# After existing commands
memory-tool sync-to 2>/dev/null
```

Now every session end exports to OpenClaw automatically.

## Troubleshooting

### memory-tool: command not found

Check installation:

```bash
which memory-tool
# Should show: /usr/local/bin/memory-tool

cat /usr/local/bin/memory-tool
# Should contain: python3 $HOME/.claude/projects/-root/memory/memory-tool.py "$@"
```

Fix permissions:

```bash
sudo chmod +x /usr/local/bin/memory-tool
```

### sqlite3.OperationalError: no such module: fts5

Your SQLite is too old. Rebuild with FTS5:

**Ubuntu/Debian**:
```bash
sudo apt-get install sqlite3 libsqlite3-dev
```

**macOS**:
```bash
brew upgrade sqlite3
```

Verify:
```bash
sqlite3 --version
# Need: 3.37.0 or higher
```

### ImportError: No module named 'sqlite_vec'

Install optional dependencies:

```bash
pip install sqlite-vec
```

Or disable semantic search (use keyword + graph only):

```bash
memory-tool search "query"  # Uses FTS, no --semantic
```

### Hooks not firing

Check Claude Code version:

```bash
claude --version
# Need: v1.0.0+ for hook support
```

Check hook paths in `~/.claude/settings.json`:

```bash
cat ~/.claude/settings.json | grep -A 4 hooks
```

Paths must be absolute. Use `~/.claude/...` or `/home/user/.claude/...`

Test hook manually:

```bash
echo '{"tool_name":"Bash","tool_result":"Exit code: 1"}' | ~/.claude/projects/-root/memory/error-hook.sh
```

### Semantic search slow

First search loads model into memory (~200MB). Subsequent searches are faster.

For faster startup, use `--keyword` for simple queries:

```bash
memory-tool search "nginx" --keyword
```

### Database locked

Another process is accessing the DB. Wait and retry.

If persistent, check for zombie processes:

```bash
lsof ~/.claude/projects/-root/memory/memories.db
```

Kill if needed:

```bash
kill <PID>
```

### Out of disk space for backups

Backups accumulate in `/root/backups/memory/`. Clean old backups:

```bash
ls -lah /root/backups/memory/
rm /root/backups/memory/memories_2026030*.db  # Delete old backups
```

Change retention in daily-maintenance.sh:

```bash
# Keep only last 3 backups
ls -t /root/backups/memory/memories_*.db | tail -n +4 | xargs rm -f
```

## Uninstall

Remove all files:

```bash
# Remove CLI command
sudo rm /usr/local/bin/memory-tool

# Remove memory directory (WARNING: deletes all memories)
rm -rf ~/.claude/projects/-root/memory/

# Remove backups
rm -rf /root/backups/memory/

# Remove hooks from settings.json
# Manually edit ~/.claude/settings.json and remove "hooks" section

# Remove cron job
crontab -e
# Delete the daily-maintenance.sh line
```

To preserve memories before uninstall:

```bash
# Export to markdown
memory-tool export > ~/memory-backup.md

# Or copy database
cp ~/.claude/projects/-root/memory/memories.db ~/memory-backup.db
```

## Upgrade

To upgrade to latest version:

```bash
cd ai-memory-sqlite
git pull
./scripts/install.sh
```

The installer preserves your existing `memories.db`.

**Database migrations** are automatic. Schema changes apply on first run after upgrade.

Check for breaking changes in CHANGELOG.md before upgrading.

## Verification Checklist

After installation, verify all features:

```bash
# 1. Basic commands
memory-tool stats
memory-tool add learning "Test memory" --project Test
memory-tool list --project Test
memory-tool search "test"

# 2. Semantic search (if optional deps installed)
memory-tool search "test" --semantic
memory-tool reindex

# 3. Graph features
memory-tool graph add project TestProject
memory-tool graph rel TestProject uses PostgreSQL
memory-tool graph spread TestProject

# 4. Session features
memory-tool snapshot "Test snapshot" --project Test
memory-tool snapshots --limit 5

# 5. Maintenance
memory-tool decay
memory-tool backup
memory-tool stale

# 6. OpenClaw sync (if applicable)
memory-tool sync

# 7. Hooks (requires Claude Code session)
# Trigger error: run failing command in Claude Code
# Check: memory-tool list --category error

# 8. Topics export
memory-tool topics
ls -la ~/.claude/projects/-root/memory/topics/
```

If all commands succeed, installation is complete.

## Support

For issues, see:
- [Troubleshooting](#troubleshooting)
- [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- GitHub Issues: https://github.com/yourusername/ai-memory-sqlite/issues
