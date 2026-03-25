#!/usr/bin/env bash
# Dream Agent — Git post-commit hook
# Captures commit data and writes to ~/dream/inbox/ for async processing.
#
# Install in a specific repo:
#   cp hooks/post-commit.sh .git/hooks/post-commit && chmod +x .git/hooks/post-commit
#
# Install globally:
#   git config --global core.hooksPath ~/.config/git/hooks
#   cp hooks/post-commit.sh ~/.config/git/hooks/post-commit

set -e

INBOX_DIR="${HOME}/dream/inbox"

# Skip if inbox doesn't exist (Dream not set up)
[ -d "$INBOX_DIR" ] || exit 0

# Skip if .no-dream file exists in repo root
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
[ -f "${REPO_ROOT}/.no-dream" ] && exit 0

# Gather commit data
REPO_NAME="$(basename "$REPO_ROOT")"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
MESSAGE="$(git log -1 --format='%s' 2>/dev/null || echo '')"
BODY="$(git log -1 --format='%b' 2>/dev/null || echo '')"
AUTHOR="$(git log -1 --format='%an' 2>/dev/null || echo '')"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%S)"

# Get diff (cap at 50KB to avoid huge commits)
DIFF="$(git diff HEAD~1 HEAD 2>/dev/null | head -c 51200 || echo '')"

# Skip if diff is empty (e.g., merge commits with no changes)
[ -z "$DIFF" ] && [ -z "$MESSAGE" ] && exit 0

# Write JSON to inbox
FILENAME="commit-${REPO_NAME}-${TIMESTAMP}-$$.json"

# Use python3 for safe JSON encoding (available on macOS)
python3 -c "
import json, sys
data = {
    'type': 'commit',
    'repo': sys.argv[1],
    'branch': sys.argv[2],
    'message': sys.argv[3],
    'body': sys.argv[4],
    'author': sys.argv[5],
    'timestamp': sys.argv[6],
    'diff': sys.stdin.read()
}
with open(sys.argv[7], 'w') as f:
    json.dump(data, f, ensure_ascii=False)
" "$REPO_NAME" "$BRANCH" "$MESSAGE" "$BODY" "$AUTHOR" "$TIMESTAMP" "${INBOX_DIR}/${FILENAME}" <<< "$DIFF"
