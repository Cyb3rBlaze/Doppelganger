#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="python"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  ./ingest.sh ingest [SOURCE_DIR]"
  echo "  ./ingest.sh search QUERY [LIMIT]"
  exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
  ingest)
    SOURCE_DIR="${1:-${INTERNAL_DOCUMENTS_SOURCE_DIR:-}}"
    if [[ -n "$SOURCE_DIR" ]]; then
      exec "$PYTHON_BIN" -m core.ingest ingest --source-dir "$SOURCE_DIR"
    fi
    exec "$PYTHON_BIN" -m core.ingest ingest
    ;;
  search)
    if [[ $# -lt 1 ]]; then
      echo "Usage: ./ingest.sh search QUERY [LIMIT]"
      exit 1
    fi
    QUERY="$1"
    LIMIT="${2:-5}"
    exec "$PYTHON_BIN" -m core.ingest search "$QUERY" --limit "$LIMIT"
    ;;
  *)
    echo "Unknown command: $COMMAND"
    echo "Usage:"
    echo "  ./ingest.sh ingest [SOURCE_DIR]"
    echo "  ./ingest.sh search QUERY [LIMIT]"
    exit 1
    ;;
esac
