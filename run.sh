#!/usr/bin/env bash
# Run the system for development: API and worker together.
#
# Two processes now, not one. Starting only the API leaves plans sitting in the
# queue forever, which looks like a hang and is the first thing to check.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PY=.venv/bin/python

cleanup() { echo; echo "stopping…"; kill 0; }
trap cleanup EXIT INT TERM

echo "worker  → planning jobs"
$PY worker.py &

echo "api     → http://127.0.0.1:${PORT}"
$PY -m uvicorn api:app --reload --host 0.0.0.0 --port "$PORT" &

wait
