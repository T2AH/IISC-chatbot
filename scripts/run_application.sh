#!/usr/bin/env bash
# Run the API (Uvicorn) and Streamlit UI after ensuring deps are installed and ports are free.
# Usage: ./scripts/run_application.sh [API_PORT] [UI_PORT]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

API_PORT="${1:-${API_PORT:-8000}}"
UI_PORT="${2:-${UI_PORT:-8501}}"

mkdir -p logs

kill_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN || true)
    if [ -n "$pids" ]; then
      echo "Killing processes on port $port: $pids"
      kill -9 $pids || true
    fi
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -n tcp "$port" >/dev/null 2>&1 || true
  fi
}

stop_if_running() {
  for f in .api.pid .ui.pid; do
    if [ -f "$f" ]; then
      pid=$(cat "$f" || true)
      if [ -n "${pid}" ] && kill -0 "$pid" >/dev/null 2>&1; then
        echo "Stopping existing process $pid from $f"
        kill "$pid" || true
        sleep 1
        kill -9 "$pid" >/dev/null 2>&1 || true
      fi
      rm -f "$f"
    fi
  done
}

# 1) Stop existing processes and free ports
stop_if_running
kill_port "$API_PORT"
kill_port "$UI_PORT"

# 2) Python env and dependencies (prefer .venv if present)
# Determine VENV_DIR: $VENV_DIR > .venv > venv
VENV_DIR="${VENV_DIR:-}"
if [ -z "$VENV_DIR" ]; then
  if [ -d .venv ]; then
    VENV_DIR=".venv"
  elif [ -d venv ]; then
    VENV_DIR="venv"
  else
    VENV_DIR=".venv"
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
if [ ! -x "$PY" ]; then
  echo "Virtualenv at $VENV_DIR seems broken (no bin/python). Recreating..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
  PY="$VENV_DIR/bin/python"
fi

"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install -r requirements.txt

# 3) Load .env (export all)
set -a
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
# Allow overriding via CLI args
export API_PORT UI_PORT
set +a

# 4) Start services
nohup "$PY" -m uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" > logs/api.out 2>&1 &
echo $! > .api.pid
nohup "$PY" -m streamlit run ui/streamlit_app.py --server.port "$UI_PORT" --server.headless true > logs/ui.out 2>&1 &
echo $! > .ui.pid

echo "API started:    http://localhost:$API_PORT (PID $(cat .api.pid))"
echo "Streamlit UI:   http://localhost:$UI_PORT (PID $(cat .ui.pid))"
echo "Logs: logs/api.out, logs/ui.out"
