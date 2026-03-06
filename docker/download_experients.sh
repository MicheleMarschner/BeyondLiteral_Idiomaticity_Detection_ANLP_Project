#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENTS_URL:?Set EXPERIMENTS_URL (Release asset download URL)}"
: "${APP_ROOT:=/app}"
: "${EXPERIMENTS_DIR:=/app/experiments}"

mkdir -p "$APP_ROOT"
cd "$APP_ROOT"

if [ ! -d "$EXPERIMENTS_DIR" ] || [ -z "$(ls -A "$EXPERIMENTS_DIR" 2>/dev/null)" ]; then
  echo "[download] $EXPERIMENTS_URL"
  curl -L --fail --retry 5 --retry-delay 2 -o /tmp/experiments.tar.gz "$EXPERIMENTS_URL"
  tar -xzf /tmp/experiments.tar.gz
  rm -f /tmp/experiments.tar.gz
  test -d "$EXPERIMENTS_DIR" || (echo "[error] experiments/ missing after extract" && exit 1)
else
  echo "[skip] experiments already present"
fi

exec "$@"