#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENTS_URL:?Set EXPERIMENTS_URL (Release asset download URL)}"
: "${APP_ROOT:=/app}"
: "${EXPERIMENTS_DIR:=/app/experiments}"

mkdir -p "$APP_ROOT"
cd "$APP_ROOT"

if [ ! -d "$EXPERIMENTS_DIR" ] || [ -z "$(ls -A "$EXPERIMENTS_DIR" 2>/dev/null)" ]; then
  echo "[download] $EXPERIMENTS_URL"
  gdown --fuzzy "$EXPERIMENTS_URL" -O /tmp/experiments.tar.gz
  tar -xzf /tmp/experiments.tar.gz
  rm -f /tmp/experiments.tar.gz
  test -d "$EXPERIMENTS_DIR" || (echo "[error] experiments/ missing after extract" && exit 1)
else
  echo "[skip] experiments already present"
fi

exec "$@"