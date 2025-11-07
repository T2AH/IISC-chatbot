"""Unified app manager: kill ports, (re)start API and UI, and perform health checks.

Usage examples:
  - python scripts/app_manager.py restart --api-port 8000 --ui-port 8501
  - python scripts/app_manager.py status
  - python scripts/app_manager.py stop

This script assumes the virtual environment is already active if needed.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import requests


def kill_port(port: int) -> None:
    """Kill any process listening on the given TCP port (Linux)."""
    try:
        out = subprocess.check_output(["bash", "-lc", f"lsof -ti tcp:{port}"]).decode().strip()
        if not out:
            return
        for pid in out.splitlines():
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    except subprocess.CalledProcessError:
        # no process found
        return


def start_api(port: int = 8000) -> subprocess.Popen:
    """Start FastAPI (uvicorn) in reload mode."""
    env = os.environ.copy()
    # Ensure project root on PYTHONPATH
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", str(port), "--reload"]
    return subprocess.Popen(cmd, env=env)


def start_ui(port: int = 8501) -> subprocess.Popen:
    """Start Streamlit UI headless on the given port."""
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.headless", "true", "--server.port", str(port)]
    return subprocess.Popen(cmd, env=env)


def wait_for_health(url: str, timeout: float = 30.0) -> bool:
    """Wait for /health to become healthy within timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                status = data.get("status")
                if status in {"healthy", "empty"}:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def restart(api_port: int, ui_port: int) -> None:
    # Kill existing servers
    kill_port(api_port)
    kill_port(ui_port)

    # Start API
    api_proc = start_api(api_port)
    api_ok = wait_for_health(f"http://localhost:{api_port}/health", timeout=45)
    print(f"API health: {'OK' if api_ok else 'FAILED'} at http://localhost:{api_port}/health")

    # Start UI
    ui_proc = start_ui(ui_port)
    print(f"UI started at http://localhost:{ui_port}")

    # Keep parent alive to pass-through Ctrl+C and terminate children
    try:
        while True:
            time.sleep(2)
            # Optionally, check if children exited unexpectedly
            if api_proc.poll() is not None:
                print("API process exited; terminating UI...")
                ui_proc.terminate()
                break
            if ui_proc.poll() is not None:
                print("UI process exited; terminating API...")
                api_proc.terminate()
                break
    except KeyboardInterrupt:
        print("Stopping...")
        api_proc.terminate()
        ui_proc.terminate()


def status(api_port: int, ui_port: int) -> None:
    def port_pids(p: int) -> list[int]:
        try:
            out = subprocess.check_output(["bash", "-lc", f"lsof -ti tcp:{p}"]).decode().strip()
            return [int(x) for x in out.splitlines() if x]
        except subprocess.CalledProcessError:
            return []

    api_pids = port_pids(api_port)
    ui_pids = port_pids(ui_port)
    print(f"API (port {api_port}): {'RUNNING pids=' + ','.join(map(str, api_pids)) if api_pids else 'STOPPED'}")
    print(f"UI  (port {ui_port}): {'RUNNING pids=' + ','.join(map(str, ui_pids)) if ui_pids else 'STOPPED'}")
    try:
        r = requests.get(f"http://localhost:{api_port}/health", timeout=5)
        print(f"API /health: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"API /health: not reachable ({e})")


def stop(api_port: int, ui_port: int) -> None:
    kill_port(api_port)
    kill_port(ui_port)
    print("Stopped API and UI (if running).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage API and UI processes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_restart = sub.add_parser("restart", help="Kill ports and start API+UI")
    p_restart.add_argument("--api-port", type=int, default=8000)
    p_restart.add_argument("--ui-port", type=int, default=8501)

    p_status = sub.add_parser("status", help="Show running status and API health")
    p_status.add_argument("--api-port", type=int, default=8000)
    p_status.add_argument("--ui-port", type=int, default=8501)

    p_stop = sub.add_parser("stop", help="Stop API and UI")
    p_stop.add_argument("--api-port", type=int, default=8000)
    p_stop.add_argument("--ui-port", type=int, default=8501)

    args = parser.parse_args()

    if args.cmd == "restart":
        restart(args.api_port, args.ui_port)
    elif args.cmd == "status":
        status(args.api_port, args.ui_port)
    elif args.cmd == "stop":
        stop(args.api_port, args.ui_port)


if __name__ == "__main__":
    main()
