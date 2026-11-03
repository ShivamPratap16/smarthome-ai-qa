"""Run every test suite that exists so far, one after another.

Each suite writes its own ``reports/<suite>/latest.json``. This script does NOT
decide pass/fail of the bot — that is the quality gate's job (``python -m qa.gate``).
It only fails (exit 1) when a suite *crashed*: it ran but produced no report.
Suites whose entry file does not exist yet are skipped.

Usage::

    python scripts/run_all.py                 # all suites
    python scripts/run_all.py --only evals api
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


@dataclass
class Suite:
    """One suite: its report name, the command to run it, and the file that must exist."""

    name: str
    command: list[str]
    needs: str  # path (relative to repo root) that must exist, otherwise the suite is skipped


SUITES = [
    Suite("evals", [PY, "-m", "qa.evals.run", "--mode", "replay"], "qa/evals/run.py"),
    Suite("redteam", [PY, "-m", "qa.redteam.run", "--mode", "replay"], "qa/redteam/run.py"),
    Suite("tool_calling", [PY, "-m", "pytest", "tests/conversation", "-q"], "tests/conversation"),
    Suite("api", [PY, "-m", "pytest", "tests/api", "-q"], "tests/api"),
    Suite("web", [PY, "-m", "pytest", "tests/web", "-q"], "tests/web"),
    Suite("data", [PY, "-m", "qa.data_checks.run"], "qa/data_checks/run.py"),
    Suite("load", [PY, "load/run_load.py"], "load/run_load.py"),
]


def latest_mtime(suite: str) -> float | None:
    """Modification time of reports/<suite>/latest.json, or None if missing."""
    path = ROOT / "reports" / suite / "latest.json"
    return path.stat().st_mtime if path.exists() else None


def run_suite(suite: Suite) -> str:
    """Run one suite and return its outcome: skipped | ok | tests-failed | no-report."""
    if not (ROOT / suite.needs).exists():
        print(f"--- {suite.name}: skipped ({suite.needs} not found)")
        return "skipped"
    print(f"--- {suite.name}: {' '.join(suite.command[1:])}", flush=True)
    before = latest_mtime(suite.name)
    env = {**os.environ, "LLM_MODE": os.environ.get("LLM_MODE", "replay")}
    started = time.time()
    code = subprocess.run(suite.command, check=False, cwd=ROOT, env=env).returncode
    after = latest_mtime(suite.name)
    took = time.time() - started
    if after is None or after == before:
        print(f"--- {suite.name}: exit {code}, NO new report written ({took:.0f}s)")
        return "no-report"
    outcome = "ok" if code == 0 else "tests-failed"
    print(f"--- {suite.name}: exit {code}, report written ({took:.0f}s)")
    return outcome


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit 1 only if a suite produced no report."""
    parser = argparse.ArgumentParser(description="Run all available QA suites")
    parser.add_argument("--only", nargs="*", help="run only these suites")
    args = parser.parse_args(argv)

    selected = [s for s in SUITES if not args.only or s.name in args.only]
    results = {s.name: run_suite(s) for s in selected}

    print("\nSuite results:")
    for name, outcome in results.items():
        print(f"  {name:<13} {outcome}")
    crashed = [n for n, o in results.items() if o == "no-report"]
    if crashed:
        print(f"\nThese suites ran but wrote no report: {', '.join(crashed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
