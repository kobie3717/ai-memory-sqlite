#!/bin/bash
set -e

# AI Memory SQLite Installer
# One-command installation script

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}AI Memory SQLite Installer${NC}"
echo "================================"

# Check Python version
echo -n "Checking Python version... "
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}FAILED${NC}"
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}FAILED${NC}"
    echo "Python $PYTHON_VERSION detected. Python 3.8 or higher is required."
    exit 1
fi
echo -e "${GREEN}OK${NC} (Python $PYTHON_VERSION)"

# Check SQLite version
echo -n "Checking SQLite version... "
SQLITE_VERSION=$(python3 -c 'import sqlite3; print(sqlite3.sqlite_version)' 2>/dev/null || echo "0.0.0")
SQLITE_MAJOR=$(echo $SQLITE_VERSION | cut -d. -f1)
SQLITE_MINOR=$(echo $SQLITE_VERSION | cut -d. -f2)

if [ "$SQLITE_MAJOR" -lt 3 ] || ([ "$SQLITE_MAJOR" -eq 3 ] && [ "$SQLITE_MINOR" -lt 37 ]); then
    echo -e "${YELLOW}WARNING${NC} (SQLite $SQLITE_VERSION)"
    echo "  SQLite 3.37+ recommended for optimal performance."
    echo "  Continuing anyway..."
else
    echo -e "${GREEN}OK${NC} (SQLite $SQLITE_VERSION)"
fi

# Determine installation directory
INSTALL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/ai-memory"
BIN_DIR="$HOME/.local/bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="$SCRIPT_DIR/../memory-tool.py"

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo -e "${RED}ERROR:${NC} memory-tool.py not found at $SOURCE_FILE"
    echo "Please run this script from the ai-memory-sqlite repository."
    exit 1
fi

# Create directories
echo -n "Creating installation directory... "
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"
echo -e "${GREEN}OK${NC}"

# Copy memory-tool.py
echo -n "Installing memory-tool.py... "
cp "$SOURCE_FILE" "$INSTALL_DIR/memory-tool.py"
chmod +x "$INSTALL_DIR/memory-tool.py"
echo -e "${GREEN}OK${NC}"

# Create symlink
echo -n "Creating symlink... "
if [ -L "$BIN_DIR/memory-tool" ] || [ -f "$BIN_DIR/memory-tool" ]; then
    rm -f "$BIN_DIR/memory-tool"
    echo -e "${YELLOW}REPLACED${NC} (existing symlink removed)"
else
    echo -e "${GREEN}OK${NC}"
fi
ln -s "$INSTALL_DIR/memory-tool.py" "$BIN_DIR/memory-tool"

# Check if ~/.local/bin is in PATH
echo -n "Checking PATH configuration... "
if [[ ":$PATH:" == *":$BIN_DIR:"* ]]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}ACTION REQUIRED${NC}"
    echo ""
    echo "  Add ~/.local/bin to your PATH by adding this line to your shell config:"
    echo ""

    # Detect shell
    if [ -n "$BASH_VERSION" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.profile"
    fi

    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "  Add to: $SHELL_RC"
    echo ""

    # Offer to add it automatically
    read -p "  Add to $SHELL_RC automatically? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "" >> "$SHELL_RC"
        echo "# Added by ai-memory-sqlite installer" >> "$SHELL_RC"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_RC"
        echo -e "  ${GREEN}Added!${NC} Restart your shell or run: source $SHELL_RC"
    else
        echo -e "  ${YELLOW}Skipped${NC}. Add manually to use 'memory-tool' command globally."
    fi
fi

# Initialize database
echo -n "Initializing database... "
export PATH="$BIN_DIR:$PATH"
if "$BIN_DIR/memory-tool" --init &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}SKIPPED${NC} (already initialized)"
fi

# Success message
echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
echo "  2. Test the installation: memory-tool stats"
echo "  3. Add your first memory: memory-tool add learning \"Your first memory\""
echo "  4. (Optional) Set up semantic search: bash scripts/setup-embedding-model.sh"
echo "  5. (Optional) Configure Claude Code hooks: see hooks/claude-code/README.md"
echo ""
echo "Documentation: https://github.com/kobie3717/ai-memory-sqlite"
echo ""
