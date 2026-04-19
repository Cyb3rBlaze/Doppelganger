#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHANNEL="${1:-}"

resolve_python() {
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    printf '%s\n' "${CONDA_PREFIX}/bin/python"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    printf '%s\n' "${VIRTUAL_ENV}/bin/python"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  echo "No Python interpreter found on PATH." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python)"

print_usage() {
  cat <<'EOF'
Usage: ./start.sh <target>

Targets:
  api       Start the FastAPI server with uvicorn reload enabled
  server    Alias for api
  terminal  Start the local terminal adapter
  telegram  Start the Telegram long-polling adapter
  gmail-auth  Run Gmail OAuth bootstrap and store the token locally
  unified-memory-migrate  Create unified memory tables and backfill message/doc memory
  dream-mode  Run one dream-mode pass over unified memory and add dream edges

Examples:
  ./start.sh api
  ./start.sh terminal
  ./start.sh telegram
  ./start.sh gmail-auth
  ./start.sh unified-memory-migrate
  ./start.sh dream-mode
EOF
}

if [[ -z "$CHANNEL" ]]; then
  print_usage
  exit 1
fi

case "$CHANNEL" in
  api|server)
    exec "$PYTHON_BIN" -m uvicorn app.main:app --reload
    ;;
  terminal)
    exec "$PYTHON_BIN" -m app.channels.terminal
    ;;
  telegram)
    exec "$PYTHON_BIN" -m app.channels.telegram
    ;;
  gmail-auth)
    exec "$PYTHON_BIN" -m app.tools.gmail_client
    ;;
  unified-memory-migrate)
    exec "$PYTHON_BIN" -m app.services.unified_memory
    ;;
  dream-mode)
    exec "$PYTHON_BIN" -m app.services.dream_mode
    ;;
  *)
    echo "Unknown target: $CHANNEL" >&2
    print_usage
    exit 1
    ;;
esac
