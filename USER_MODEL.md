# User Model Feature

Honcho-style user modeling for AI-IQ. Builds a structured user profile from memory patterns.

## What It Does

The user model scans your existing memories to build a living document that captures:

1. **Identity** - Your role, expertise, email, projects
2. **Preferences** - Communication style, code style, tools you prefer/avoid
3. **Patterns** - Recurring workflows, common requests, behavioral patterns
4. **Corrections** - Things the AI got wrong + correct behavior
5. **Dialect** - Shorthand terms, jargon, tags, and projects you use

## Usage

### Basic Mode (Scan Existing Memories)

Generate user profile from existing memories:

```bash
memory-tool user-model
```

Output: `/root/.claude/projects/-root/memory/user-model.md`

Preview without writing:

```bash
memory-tool user-model --dry-run
```

Custom output path:

```bash
memory-tool user-model --output /path/to/custom-file.md
```

### Dialectic Mode (AI Analysis)

Use Claude API to analyze recent session data and extract user model signals:

```bash
memory-tool user-model --update
```

This mode:
- Scans last 7 days of memories (default)
- Calls Claude API to extract user model insights
- Saves extracted signals as new memories (category='user_model')
- Updates the user-model.md document

Analyze last 14 days:

```bash
memory-tool user-model --update --days 14
```

Preview without saving:

```bash
memory-tool user-model --update --dry-run
```

**Requirements for --update mode:**
- `anthropic` package installed (`pip install anthropic`)
- `ANTHROPIC_API_KEY` environment variable set

## How It Works

### 1. Memory Scanning

The module queries the SQLite database for memories matching user model patterns:

- **Identity**: Memories with category 'profile', 'expertise', 'role', or containing "I am", "my role", etc.
- **Preferences**: Memories with category 'preference', 'style', or containing "prefer", "always use", "I like", etc.
- **Patterns**: Memories that occur 2+ times (grouped by content)
- **Corrections**: Records from the `corrections` table + feedback memories
- **Dialect**: Unique tags and project names from all memories

### 2. Document Generation

Builds a markdown document with:
- Header with timestamp
- 5 main sections (Identity, Preferences, Patterns, Corrections, Dialect)
- Top 15 items per section (to keep document size manageable)
- Project context where relevant
- Active projects ranked by memory count
- Common tags sorted alphabetically

### 3. Dialectic Update (Optional)

When `--update` is used:
1. Gathers recent memories from last N days
2. Sends to Claude API with analysis prompt
3. Claude extracts structured insights (JSON format)
4. Saves insights as new memories (category='user_model')
5. Regenerates full user model document

## Output Format

```markdown
# User Model

*Last updated: 2026-05-24 10:49*

## Identity

- User email: jiwentzel@icloud.com
- Role: Full-stack developer, AI integrator
  *Project: WhatsAuction*

## Preferences

- Always use TypeScript over JavaScript for type safety
- Prefer Docker for deployment, not PM2
- Communication: Direct, action-oriented, skip pleasantries

## Patterns

- **Deploy with backup + rollback script** (5x)
  *Project: FlashVault*
- **Run tests before merging** (3x)

## Corrections

- **Original**: Use npm for dependencies
  **Corrected**: Use pnpm for monorepo workspace support

## Dialect

**Active Projects:**
- WhatsAuction (245 memories)
- FlashVault (189 memories)
- AI-IQ (67 memories)

**Common Tags:**
baileys, docker, nginx, typescript, postgresql, auth, api, deployment
```

## Integration with AI Agents

The user model is designed to be:

1. **Read by AI agents** at session start to understand user preferences
2. **Updated automatically** via hooks (e.g., session end hook runs `memory-tool user-model`)
3. **Referenced in prompts** to customize agent behavior

Example integration in CLAUDE.md:

```markdown
# User Profile

See /root/.claude/projects/-root/memory/user-model.md for:
- My communication preferences
- Tools I prefer
- Common workflows
- Past corrections
```

## API Usage

Programmatic access:

```python
from memory_tool.database import get_db
from memory_tool.user_model import scan_user_memories, generate_user_model

db = get_db()

# Scan memories
user_data = scan_user_memories(db)
print(f"Found {len(user_data['identity'])} identity signals")

# Generate markdown
from pathlib import Path
markdown = generate_user_model(db, output_path=Path("/tmp/user-model.md"))

db.close()
```

## File Location

Default output: `/root/.claude/projects/-root/memory/user-model.md`

This path is chosen because:
- It's in the Claude Code memory directory (persists across sessions)
- It's at the project root level (accessible to all projects)
- It's near MEMORY.md (existing memory context file)

## Maintenance

The user model should be refreshed periodically:

1. **After major work sessions** - Run `memory-tool user-model --update`
2. **Weekly** - Automated via cron: `memory-tool user-model`
3. **When preferences change** - Manual refresh after adding preference memories

## Design Philosophy

Inspired by [Honcho](https://www.honcho.dev/), this feature treats user modeling as:

- **Progressive** - Builds over time as you use the system
- **Structured** - Clear categories, not just unstructured notes
- **Living** - Updates with new insights, supersedes old patterns
- **Actionable** - Designed to guide AI behavior, not just document history

## Comparison to Alternatives

**vs Cursor Rules:**
- Cursor: Static rules file, manually maintained
- User Model: Dynamic, auto-generated from actual behavior

**vs .claude/memory:**
- .claude/memory: Project-specific, free-form
- User Model: Cross-project, structured categories

**vs Notion/Obsidian notes:**
- Notion: Manual note-taking
- User Model: Automated extraction from AI session logs

## Future Enhancements

Potential improvements:

1. **Confidence scores** - Track how certain each insight is
2. **Temporal tracking** - "Used to prefer X, now prefers Y"
3. **Contradiction detection** - Flag conflicting preferences
4. **Multi-agent coordination** - Share user model across agent teams
5. **Privacy controls** - Exclude sensitive categories from sharing

## License

Part of AI-IQ, MIT licensed.
