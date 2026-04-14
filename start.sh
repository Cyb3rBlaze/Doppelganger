#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHANNEL="${1:-}"

print_usage() {
  cat <<'EOF'
Usage: ./start.sh <target>

Targets:
  api       Start the FastAPI server with uvicorn reload enabled
  server    Alias for api
  terminal  Start the local terminal adapter
  telegram  Start the Telegram long-polling adapter
  gmail-auth  Run Gmail OAuth bootstrap and store the token locally

Examples:
  ./start.sh api
  ./start.sh terminal
  ./start.sh telegram
  ./start.sh gmail-auth
EOF
}

if [[ -z "$CHANNEL" ]]; then
  print_usage
  exit 1
fi

case "$CHANNEL" in
  api|server)
    exec uvicorn app.main:app --reload
    ;;
  terminal)
    exec python -m app.channels.terminal
    ;;
  telegram)
    exec python -m app.channels.telegram
    ;;
  gmail-auth)
    exec python -m app.tools.gmail_client
    ;;
  *)
    echo "Unknown target: $CHANNEL" >&2
    print_usage
    exit 1
    ;;
esac
