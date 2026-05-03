#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

branch="$(git rev-parse --abbrev-ref HEAD)"
origin_url="$(git remote get-url origin)"

ensure_github_known_host() {
  if [[ "$origin_url" == git@github.com:* || "$origin_url" == ssh://git@github.com/* ]]; then
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    touch "$HOME/.ssh/known_hosts"
    chmod 600 "$HOME/.ssh/known_hosts"

    if ! ssh-keygen -F github.com -f "$HOME/.ssh/known_hosts" >/dev/null 2>&1; then
      echo "[start_wrapper] Adding github.com host key to known_hosts..."
      ssh-keyscan -t ed25519 github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
    fi
  fi
}

# Download latest scripts from S3/R2 — avoids Docker rebuild on code changes.
# Falls back silently if S3 is not configured or unavailable.
try_s3_update() {
  echo "[start_wrapper] Checking S3/R2 for script updates..."
  python3 "$REPO_DIR/self_update.py" || echo "[start_wrapper] WARNING: S3 script update failed, continuing"
}

update_repo() {
  echo "[start_wrapper] Syncing branch '$branch' from origin..."
  if ! git fetch --prune origin 2>&1; then
    echo "[start_wrapper] WARNING: git fetch failed, continuing with current version"
    return 0
  fi

  if git show-ref --quiet "refs/remotes/origin/$branch"; then
    git pull --ff-only origin "$branch" || echo "[start_wrapper] WARNING: git pull failed, continuing"
  else
    echo "[start_wrapper] No origin/$branch found; skipping pull."
  fi
}

start_app() {
  echo "[start_wrapper] Starting app..."
  exec python3 app.py "$@"
}

ensure_github_known_host
try_s3_update
update_repo
start_app "$@"
