#!/usr/bin/env bash
# Stop the API and Streamlit UI started by run_application.sh
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

stop_by_pid_file() {
  local f="$1"
  if [ -f "$f" ]; then
    local pid
    pid=$(cat "$f" || true)
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "Stopping PID $pid from $f"
      kill "$pid" || true
      sleep 1
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$f"
  fi
}

stop_by_pid_file .api.pid
stop_by_pid_file .ui.pid

echo "Stopped."
