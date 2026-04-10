#!/usr/bin/env bash
set -e

# Setup logging
mkdir -p "$HOME/.local/share/forge"
LOG_FILE="$HOME/.local/share/forge/sync.log"

exec >> >(awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' >> "$LOG_FILE") 2>&1

echo "--- Starting Forge sync ---"

# Prevent concurrent runs
exec 200>/tmp/forge-sync.lock
if ! flock -n 200; then
    echo "Another sync instance is running. Exiting."
    exit 0
fi

# Ensure correct environment
cd "$HOME/Documents/project/_mycelium/oven"
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Try to get Telegram token from environment or ~/.zshenv if not set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    if grep -q "TELEGRAM_BOT_TOKEN=" "$HOME/.zshenv" 2>/dev/null; then
        export TELEGRAM_BOT_TOKEN=$(grep "TELEGRAM_BOT_TOKEN=" "$HOME/.zshenv" | cut -d'=' -f2 | tr -d "\"\'")
    fi
fi

CHAT_ID="432567587"

send_telegram() {
    local message="$1"
    if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="$CHAT_ID" \
            --data-urlencode text="$message" > /dev/null
    else
        echo "Telegram token not configured. Skipping notification: $message"
    fi
}

handle_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Sync failed with exit code $exit_code."
        send_telegram "❌ Forge sync failed! Check logs at $LOG_FILE for details."
    fi
    exit $exit_code
}

trap handle_error EXIT

echo "Fetching upstream..."
git fetch upstream

NEW_COMMITS=$(git rev-list HEAD..upstream/main --count)

if [ "$NEW_COMMITS" -eq 0 ]; then
    echo "Already up-to-date with upstream/main."
    trap - EXIT
    exit 0
fi

echo "Found $NEW_COMMITS new commit(s) from upstream."

PRE_REBASE_SHA=$(git rev-parse HEAD)

echo "Attempting rebase on upstream/main..."
if ! git rebase upstream/main; then
    echo "Rebase failed due to conflicts. Aborting rebase."
    git rebase --abort
    exit 1
fi

echo "Rebase successful! Building release..."

if ! cargo build --release; then
    echo "Compilation failed after rebase! Undoing rebase..."
    git reset --hard "$PRE_REBASE_SHA"
    exit 1
fi

echo "Copying binary to ~/.local/bin/forge..."
mkdir -p "$HOME/.local/bin"
cp target/release/forge "$HOME/.local/bin/forge"

echo "Sync complete and binary installed!"

LATEST_COMMIT=$(git log -1 --format='%h - %s' upstream/main)
echo "Upstream commit: $LATEST_COMMIT"

send_telegram "✅ Forge synced successfully!
$NEW_COMMITS new commit(s) pulled.
Latest: $LATEST_COMMIT"

trap - EXIT
exit 0