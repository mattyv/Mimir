#!/usr/bin/env python
"""Integration test watcher.

Re-runs the phase-gated pytest suite whenever a .py file in src/ or tests/
is saved.  Run with:

    python watch.py           # fast mode: no coverage
    python watch.py --cov     # with coverage gate (slower)
    python watch.py --phases phase2 phase3   # restrict markers
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

_PHASES_DEFAULT = ["phase0", "phase1", "phase2", "phase3"]
_WATCH_DIRS = ["src", "tests"]


def _build_cmd(phases: list[str], cov: bool) -> list[str]:
    marker = " or ".join(phases)
    cmd = [sys.executable, "-m", "pytest", "-m", marker, "-q", "--tb=short"]
    if cov:
        cmd += ["--cov=src/mimir", "--cov-fail-under=95"]
    return cmd


def _run(cmd: list[str]) -> None:
    print("\n" + "─" * 60, flush=True)
    subprocess.run(cmd)
    print("─" * 60 + "\n", flush=True)


class _Handler(FileSystemEventHandler):
    def __init__(self, cmd: list[str]) -> None:
        self._cmd = cmd
        self._last = 0.0

    def on_modified(self, event: FileSystemEvent) -> None:
        if not str(event.src_path).endswith(".py"):
            return
        now = time.monotonic()
        if now - self._last < 1.0:  # debounce
            return
        self._last = now
        _run(self._cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cov", action="store_true", help="Enable coverage gate")
    parser.add_argument("--phases", nargs="+", default=_PHASES_DEFAULT)
    args = parser.parse_args()

    cmd = _build_cmd(args.phases, args.cov)
    root = Path(__file__).parent

    print(f"Watching {_WATCH_DIRS} — running: {' '.join(cmd)}")
    _run(cmd)  # initial run

    observer = Observer()
    handler = _Handler(cmd)
    for d in _WATCH_DIRS:
        observer.schedule(handler, str(root / d), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
