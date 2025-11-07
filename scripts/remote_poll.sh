#!/usr/bin/env bash
set -u
LOG_OUT="remote_build_poll.log"
INTERVAL=120
{
  echo "[INFO] Starting poller (interval=${INTERVAL}s) at $(date)"
  echo "[INFO] Writing to $(pwd)/$LOG_OUT"
} >> "$LOG_OUT"

while true; do
  TS=$(date '+%F %T')
  {
    echo "[$TS] == POLL =="
    ssh iisc-remote '
      set -e
      cd ~/iisc-index
      LOG=build.log
      PIDFILE=index_build.pid
      echo "== Process =="
      if [ -f "$PIDFILE" ] && ps -p $(cat "$PIDFILE") >/dev/null; then
        ps -p $(cat "$PIDFILE") -o pid,etime,%cpu,%mem,cmd --no-headers || true
        STATUS=running
      else
        echo "stopped"
        STATUS=stopped
      fi
      echo
      echo "== Last log =="
      tail -n 15 "$LOG" 2>/dev/null || echo "no log yet"
      echo
      echo "== Index size =="
      du -sh data/index/fastembed_bge_small_iisc_full 2>/dev/null || echo "not yet"
      echo
      echo "__STATUS__:$STATUS"
    '
  } >> "$LOG_OUT" 2>&1

  # Determine if done
  if tail -n 1 "$LOG_OUT" | grep -q "__STATUS__:stopped"; then
    echo "[$TS] Build appears done; stopping poller." >> "$LOG_OUT"
    exit 0
  fi

  sleep "$INTERVAL"

done
