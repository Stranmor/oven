#!/usr/bin/env bash

set -euo pipefail

echo "Fetching upstream..."
git fetch upstream

# Check if there are uncommitted changes
STASHED=0
if ! git diff-index --quiet HEAD --; then
    echo "Stashing uncommitted changes..."
    git stash push -m "Auto-stash before upstream sync"
    STASHED=1
fi

echo "Rebasing on upstream/main..."
if ! git rebase upstream/main; then
    echo "ERROR: Rebase conflict detected! Aborting rebase..."
    git rebase --abort
    
    if [ "$STASHED" -eq 1 ]; then
        echo "Restoring stashed changes..."
        git stash pop || true
    fi
    
    echo "Error: Rebase failed. Please resolve conflicts manually via agent."
    exit 1
fi

if [ "$STASHED" -eq 1 ]; then
    echo "Restoring stashed changes..."
    git stash pop || echo "Warning: Conflicts occurred while popping stash."
fi

echo "Running cargo check..."
cargo check --workspace --all-targets

echo "Upstream sync completed successfully!"
