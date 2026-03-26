"""Configuration constants for memory-tool."""

import os
from pathlib import Path

# Paths
MEMORY_DIR = Path(__file__).parent.parent
DB_PATH = MEMORY_DIR / "memories.db"
MEMORY_MD_PATH = MEMORY_DIR / "MEMORY.md"
TOPICS_DIR = MEMORY_DIR / "topics"
BACKUP_DIR = Path(os.getenv("MEMORY_BACKUP_DIR", str(Path.home() / "backups/memory")))
MAX_MEMORY_MD_BYTES = 5120  # 5KB hard cap

# OpenClaw Bridge paths (configurable via environment variables)
OPENCLAW_MEMORY_DIR = Path(os.getenv("OPENCLAW_MEMORY_DIR", str(Path.home() / ".openclaw/workspace/memory")))
OPENCLAW_GRAPH_DB = Path(os.getenv("OPENCLAW_GRAPH_DB", str(Path.home() / ".openclaw/memory-graph.db")))
SYNC_STATE_FILE = MEMORY_DIR / ".sync-state.json"

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

# Project detection paths (example - customize for your projects)
PROJECT_PATHS = {
    # Example format:
    # "/path/to/project": "ProjectName",
    # "/home/user/myapp": "MyApp",
}

# Auto-Tag Keywords
AUTO_TAG_RULES = {
    "pm2": ["pm2"],
    "whatsapp": ["whatsapp", "baileys", "webhook", "meta dashboard", "waba"],
    "baileys": ["baileys", "qrcode-terminal"],
    "database": ["postgresql", "psql", "prisma", "sequelize", "migration", "schema"],
    "auth": ["jwt", "login", "password", "token", "auth", "bcrypt"],
    "nginx": ["nginx", "reverse proxy", "ssl", "certbot", "letsencrypt"],
    "docker": ["docker", "container", "dockerfile"],
    "payfast": ["payfast", "payment", "merchant"],
    "wireguard": ["wireguard", "wg0", "wg-quick", "wg show"],
    "dns": ["unbound", "dns", "resolve", "dig"],
    "esm": ["esm", "commonjs", "import", "require", "module"],
    "react": ["react", "vite", "tailwind", "frontend", "tsx", "jsx"],
    "api": ["endpoint", "route", "controller", "middleware", "express"],
}
