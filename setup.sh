#!/usr/bin/env bash
# Dream Agent — One-command setup
# Creates venv, installs dependencies, sets up inbox, installs LaunchAgents.
#
# Usage:
#   ./setup.sh              # Full setup
#   ./setup.sh --no-launch  # Skip LaunchAgent installation

set -e

INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${INSTALL_DIR}/.venv"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
INBOX_DIR="${HOME}/dream/inbox"
PROCESSED_DIR="${HOME}/dream/processed"

NO_LAUNCH=false
for arg in "$@"; do
    case "$arg" in
        --no-launch) NO_LAUNCH=true ;;
    esac
done

echo ""
echo "================================================"
echo "  DREAM AGENT — Setup"
echo "================================================"
echo ""
echo "  Install dir: ${INSTALL_DIR}"
echo "  Venv dir:    ${VENV_DIR}"
echo ""

# 1. Create Python venv
echo "[1/5] Creating Python virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created ${VENV_DIR}"
else
    echo "  Venv already exists, skipping"
fi

# 2. Install dependencies
echo "[2/5] Installing dependencies..."
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -e "${INSTALL_DIR}[menubar]"
echo "  Dependencies installed"

# 3. Create inbox directories
echo "[3/5] Creating dream inbox directories..."
mkdir -p "${INBOX_DIR}" "${PROCESSED_DIR}"
mkdir -p "${INSTALL_DIR}/data"
echo "  ${INBOX_DIR}"
echo "  ${PROCESSED_DIR}"

# 4. Make hook executable
echo "[4/5] Setting up git hook..."
chmod +x "${INSTALL_DIR}/hooks/post-commit.sh"
echo "  Hook ready at: ${INSTALL_DIR}/hooks/post-commit.sh"
echo "  To install globally:"
echo "    mkdir -p ~/.config/git/hooks"
echo "    cp ${INSTALL_DIR}/hooks/post-commit.sh ~/.config/git/hooks/post-commit"
echo "    git config --global core.hooksPath ~/.config/git/hooks"

# 5. Install LaunchAgents
if [ "$NO_LAUNCH" = true ]; then
    echo "[5/5] Skipping LaunchAgent installation (--no-launch)"
else
    echo "[5/5] Installing LaunchAgents..."
    mkdir -p "${LAUNCH_AGENTS_DIR}"

    for plist in com.dream.service.plist com.dream.menubar.plist com.dream.sync.plist; do
        src="${INSTALL_DIR}/launchd/${plist}"
        dst="${LAUNCH_AGENTS_DIR}/${plist}"

        # Stop existing service if running
        launchctl bootout "gui/$(id -u)/${plist%.plist}" 2>/dev/null || true

        # Replace placeholders and install
        sed \
            -e "s|__INSTALL_DIR__|${INSTALL_DIR}|g" \
            -e "s|__VENV_DIR__|${VENV_DIR}|g" \
            "${src}" > "${dst}"

        echo "  Installed ${plist}"
    done

    # Load services
    for plist in com.dream.service.plist com.dream.menubar.plist com.dream.sync.plist; do
        launchctl bootstrap "gui/$(id -u)" "${LAUNCH_AGENTS_DIR}/${plist}" 2>/dev/null || \
        launchctl load "${LAUNCH_AGENTS_DIR}/${plist}" 2>/dev/null || true
        echo "  Loaded ${plist%.plist}"
    done
fi

echo ""
echo "================================================"
echo "  DREAM AGENT — Setup Complete"
echo "================================================"
echo ""
echo "  Services:"
echo "    API:     http://localhost:8000/api/health"
echo "    Docs:    http://localhost:8000/docs"
echo "    Menubar: Look for the mushroom icon"
echo "    Sync:    Runs every hour (engram -> substrate)"
echo ""
echo "  Directories:"
echo "    Inbox:     ${INBOX_DIR}"
echo "    Processed: ${PROCESSED_DIR}"
echo "    Data:      ${INSTALL_DIR}/data/"
echo ""
echo "  Next steps:"
echo "    1. Install the git hook in your repos (see step 4 above)"
echo "    2. Check the API: curl http://localhost:8000/api/health"
echo "    3. Start coding — Dream watches your commits"
echo ""
echo "  The substrate dreams."
echo ""
