#!/usr/bin/env python3
"""
🫀 CardioSense Full-Stack Orchestrator
======================================
Launches both Backend (Flask REST API on :5001) and Frontend (Web UI on :8000)
with unified logging, health monitoring, port management, and graceful shutdown.

Supported execution commands:
    python start
    python start.py
    python3 start
    python3 start.py
    ./start
    ./start.py
"""

import os
import sys
import time
import signal
import socket
import atexit
import argparse
import threading
import subprocess
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
BACKEND_SCRIPT = PROJECT_ROOT / "backend" / "app.py"
if not BACKEND_SCRIPT.exists():
    BACKEND_SCRIPT = PROJECT_ROOT / "app.py"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Default configuration
DEFAULT_BACKEND_PORT = 5001
DEFAULT_FRONTEND_PORT = 8000
HEALTH_CHECK_TIMEOUT = 20

# Colors for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Track processes for cleanup
backend_proc = None
frontend_proc = None
stop_event = threading.Event()


def print_banner():
    print(f"\n{CYAN}{BOLD}{'=' * 65}{RESET}")
    print(f"{CYAN}{BOLD}   🫀  CardioSense AI — Full-Stack Dual-Engine Launcher{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 65}{RESET}")
    print(f"  {DIM}Root directory:{RESET} {PROJECT_ROOT}")
    print(f"  {DIM}Python engine :{RESET} {sys.executable}")
    print(f"{CYAN}{BOLD}{'-' * 65}{RESET}\n")


def is_port_in_use(port):
    """Check if a port is bound on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def get_pids_on_port(port):
    """Find PIDs listening on a specific port using lsof."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        if not out:
            return []
        pids = [int(p) for p in out.splitlines() if p.strip().isdigit()]
        return pids
    except Exception:
        return []


def kill_pids_on_port(port, label="Server"):
    """Gracefully terminate any stale process holding a port."""
    pids = get_pids_on_port(port)
    if not pids:
        return

    print(f"{YELLOW}⚠️  Port {port} is occupied by existing PID(s): {pids}. Cleaning up...{RESET}")
    for pid in pids:
        # Don't kill our own process
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"{RED}Permission denied terminating PID {pid}.{RESET}")

    time.sleep(0.8)

    # Force kill if still holding
    remaining = get_pids_on_port(port)
    for pid in remaining:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    time.sleep(0.3)
    if not is_port_in_use(port):
        print(f"{GREEN}✓ Cleaned port {port} successfully.{RESET}")


def stream_output(process, prefix, color):
    """Continuously stream process stdout/stderr with a colored prefix."""
    try:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            line_str = line.rstrip()
            if line_str:
                print(f"{color}[{prefix}]{RESET} {line_str}", flush=True)
    except Exception:
        pass


def check_url(url, timeout=1.0):
    """Test if a URL responds with 200 OK."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CardioSense-HealthChecker/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 304)
    except Exception:
        return False


def wait_for_services(backend_port, frontend_port, timeout=HEALTH_CHECK_TIMEOUT):
    """Poll both backend and frontend until they are verified responsive."""
    backend_url = f"http://127.0.0.1:{backend_port}/api/health"
    frontend_url = f"http://127.0.0.1:{frontend_port}/"

    print(f"{CYAN}⏳ Initializing Machine Learning models and verifying endpoints...{RESET}")
    start_time = time.time()
    backend_ready = False
    frontend_ready = False

    while time.time() - start_time < timeout:
        if not backend_ready and check_url(backend_url):
            backend_ready = True
            print(f"{GREEN}✓ Backend REST API is online{RESET}  -> {backend_url}")

        if not frontend_ready and check_url(frontend_url):
            frontend_ready = True
            print(f"{GREEN}✓ Frontend Web UI is online{RESET}   -> http://localhost:{frontend_port}")

        if backend_ready and frontend_ready:
            elapsed = time.time() - start_time
            print(f"\n{GREEN}{BOLD}✨ All CardioSense services are healthy! ({elapsed:.1f}s){RESET}\n")
            return True

        time.sleep(0.4)

    # If timeout occurred
    print(f"{YELLOW}⚠️  Health check timeout reached ({timeout}s).{RESET}")
    if not backend_ready:
        print(f"{RED}✗ Backend did not respond on port {backend_port} within timeout.{RESET}")
    if not frontend_ready:
        print(f"{RED}✗ Frontend did not respond on port {frontend_port} within timeout.{RESET}")
    return backend_ready and frontend_ready


def shutdown():
    """Cleanly shut down both child processes."""
    global backend_proc, frontend_proc
    if stop_event.is_set():
        return
    stop_event.set()

    print(f"\n\n{YELLOW}{BOLD}🛑 Shutting down CardioSense services...{RESET}")

    for name, proc in [("Backend REST API", backend_proc), ("Frontend Web Server", frontend_proc)]:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
                print(f"{GREEN}✓ {name} stopped cleanly.{RESET}")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"{YELLOW}⚠️ {name} forcefully terminated.{RESET}")
            except Exception as e:
                print(f"{RED}Error stopping {name}: {e}{RESET}")

    print(f"{CYAN}Goodbye! 👋{RESET}\n")


def signal_handler(signum, frame):
    shutdown()
    sys.exit(0)


def check_status(backend_port=DEFAULT_BACKEND_PORT, frontend_port=DEFAULT_FRONTEND_PORT):
    """Check status of services."""
    backend_online = check_url(f"http://127.0.0.1:{backend_port}/api/health")
    frontend_online = check_url(f"http://127.0.0.1:{frontend_port}/")

    print("\n" + "=" * 50)
    print(" 🫀 CardioSense System Status")
    print("=" * 50)
    print(f"Backend  (: {backend_port}): {'🟢 ONLINE' if backend_online else '🔴 OFFLINE'}")
    print(f"Frontend (: {frontend_port}): {'🟢 ONLINE' if frontend_online else '🔴 OFFLINE'}")
    print("=" * 50 + "\n")
    return backend_online and frontend_online


def stop_all(backend_port=DEFAULT_BACKEND_PORT, frontend_port=DEFAULT_FRONTEND_PORT):
    """Stop all running services on the ports."""
    print(f"\n{YELLOW}Stopping any running CardioSense services...{RESET}")
    kill_pids_on_port(backend_port, "Backend")
    kill_pids_on_port(frontend_port, "Frontend")
    print(f"{GREEN}✓ Stopped all instances.{RESET}\n")


def main():
    global backend_proc, frontend_proc

    parser = argparse.ArgumentParser(
        description="🫀 CardioSense Full-Stack Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start                 # Start both frontend & backend, open browser
    python start --no-browser    # Start without opening browser
    python start --status        # Check if services are currently running
    python start --stop          # Stop any running services on ports 5001 & 8000
        """
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    parser.add_argument("--status", action="store_true", help="Check status of running backend and frontend services")
    parser.add_argument("--stop", action="store_true", help="Stop running services on backend and frontend ports")
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT, help=f"Backend port (default: {DEFAULT_BACKEND_PORT})")
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT, help=f"Frontend port (default: {DEFAULT_FRONTEND_PORT})")
    args = parser.parse_args()

    if args.status:
        check_status(args.backend_port, args.frontend_port)
        return

    if args.stop:
        stop_all(args.backend_port, args.frontend_port)
        return

    # Print welcome banner
    print_banner()

    # Ensure clean ports before starting
    kill_pids_on_port(args.backend_port, "Backend")
    kill_pids_on_port(args.frontend_port, "Frontend")

    # Register exit and signal handlers
    atexit.register(shutdown)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 1. Launch Backend REST API
    backend_env = dict(os.environ)
    backend_env["PORT"] = str(args.backend_port)
    backend_env["PYTHONUNBUFFERED"] = "1"

    print(f"{CYAN}🚀 Starting Backend REST API on port {args.backend_port}...{RESET}")
    backend_proc = subprocess.Popen(
        [sys.executable, str(BACKEND_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 2. Launch Frontend Static Web Server
    frontend_env = dict(os.environ)
    frontend_env["PYTHONUNBUFFERED"] = "1"

    print(f"{MAGENTA}🌐 Starting Frontend Web Server on port {args.frontend_port}...{RESET}")
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(args.frontend_port), "--directory", str(FRONTEND_DIR)],
        cwd=str(PROJECT_ROOT),
        env=frontend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Start streaming logs in background threads
    t_backend = threading.Thread(
        target=stream_output,
        args=(backend_proc, "BACKEND", CYAN),
        daemon=True
    )
    t_frontend = threading.Thread(
        target=stream_output,
        args=(frontend_proc, "FRONTEND", MAGENTA),
        daemon=True
    )
    t_backend.start()
    t_frontend.start()

    # Wait for services to respond
    all_healthy = wait_for_services(args.backend_port, args.frontend_port)

    # Display Live Dashboard
    print(f"{GREEN}{BOLD}{'=' * 65}{RESET}")
    print(f"{GREEN}{BOLD}   🎉 CardioSense System is LIVE & READY!{RESET}")
    print(f"{GREEN}{BOLD}{'=' * 65}{RESET}")
    print(f"  {BOLD}🖥️  Web UI Client:{RESET}     {CYAN}http://localhost:{args.frontend_port}{RESET}")
    print(f"  {BOLD}🔌 Backend REST API:{RESET}  {CYAN}http://localhost:{args.backend_port}{RESET}")
    print(f"  {BOLD}💓 Health Endpoint:{RESET}   {CYAN}http://localhost:{args.backend_port}/api/health{RESET}")
    print(f"  {BOLD}📊 Model Metrics:{RESET}     {CYAN}http://localhost:{args.backend_port}/api/metrics{RESET}")
    print(f"  {BOLD}🎯 Prediction API:{RESET}    {CYAN}http://localhost:{args.backend_port}/api/predict (POST){RESET}")
    print(f"{GREEN}{BOLD}{'-' * 65}{RESET}")
    print(f"  {YELLOW}👉 Press {BOLD}Ctrl + C{RESET}{YELLOW} at any time to cleanly stop both servers.{RESET}")
    print(f"{GREEN}{BOLD}{'=' * 65}{RESET}\n")

    # Open in default browser
    if not args.no_browser and all_healthy:
        try:
            webbrowser.open(f"http://localhost:{args.frontend_port}")
        except Exception:
            pass

    # Keep main thread alive monitoring children
    try:
        while not stop_event.is_set():
            # Check if any process terminated prematurely
            b_poll = backend_proc.poll()
            f_poll = frontend_proc.poll()

            if stop_event.is_set():
                break

            if b_poll is not None:
                if not stop_event.is_set():
                    print(f"{RED}Backend process exited unexpectedly with code {b_poll}.{RESET}")
                break

            if f_poll is not None:
                if not stop_event.is_set():
                    print(f"{RED}Frontend process exited unexpectedly with code {f_poll}.{RESET}")
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
