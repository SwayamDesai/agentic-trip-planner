#!/usr/bin/env bash
# Start the web UI. Open http://127.0.0.1:8000 once it boots.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn api:app --reload --port "${PORT:-8000}"
