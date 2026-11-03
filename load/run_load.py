"""Run the Locust load test headless and write reports/load/latest.json (Contract 5).

    .venv/Scripts/python load/run_load.py                       # fake LLM, 20 users, 30 s
    .venv/Scripts/python load/run_load.py --users 50 --duration 60s
    .venv/Scripts/python load/run_load.py --host http://127.0.0.1:8000   # an already-running server
                                                                          # (e.g. LLM_MODE=live/replay)

Steps: start a fake-mode server (unless --host) -> run locust -> read its stats CSV ->
compute p50/p95/p99, requests/s, error rate -> write the report -> exit 1 if over budget.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa.report import write_report
from tests.api.live_server import LiveServer

P95_BUDGET_MS = 3000          # same limit as qa/thresholds.yaml (load.p95_latency_ms)
MAX_ERROR_RATE = 0.01


def run_locust(host: str, users: int, spawn_rate: int, duration: str, csv_prefix: Path) -> int:
    """Run locust headless; it writes <prefix>_stats.csv. Returns locust's exit code."""
    cmd = [sys.executable, "-m", "locust", "-f", str(ROOT / "load" / "locustfile.py"),
           "--headless", "--host", host, "-u", str(users), "-r", str(spawn_rate),
           "-t", duration, "--csv", str(csv_prefix), "--only-summary", "--loglevel", "WARNING"]
    print("running:", " ".join(cmd[2:]))
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def read_stats(csv_path: Path) -> list[dict]:
    """Rows of Locust's *_stats.csv (one per endpoint + an 'Aggregated' row)."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(row: dict, key: str) -> float:
    value = row.get(key, "") or "0"
    return 0.0 if value == "N/A" else float(value)


def summarise(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Turn Locust rows into (summary, cases) for write_report."""
    cases = []
    for row in rows:
        if row["Name"] == "Aggregated":
            continue
        count, fails = _num(row, "Request Count"), _num(row, "Failure Count")
        p95 = _num(row, "95%")
        error_rate = fails / count if count else 0.0
        passed = count > 0 and error_rate <= MAX_ERROR_RATE and p95 <= P95_BUDGET_MS
        cases.append({"id": row["Name"], "category": row["Type"], "passed": passed,
                      "reason": f"{int(count)} reqs, {int(fails)} failed, p95 {p95:.0f} ms",
                      "latency_ms": _num(row, "50%"),
                      "scores": {"p50_ms": _num(row, "50%"), "p95_ms": p95, "p99_ms": _num(row, "99%"),
                                 "rps": round(_num(row, "Requests/s"), 2),
                                 "error_rate": round(error_rate, 4)}})
    agg = next(r for r in rows if r["Name"] == "Aggregated")
    total_reqs, total_fails = _num(agg, "Request Count"), _num(agg, "Failure Count")
    metrics = {"p50_latency_ms": _num(agg, "50%"), "p95_latency_ms": _num(agg, "95%"),
               "p99_latency_ms": _num(agg, "99%"), "avg_latency_ms": round(_num(agg, "Average Response Time"), 1),
               "rps": round(_num(agg, "Requests/s"), 2), "requests": int(total_reqs),
               "error_rate": round(total_fails / total_reqs, 4) if total_reqs else 1.0}
    passed = sum(c["passed"] for c in cases)
    summary = {"total": len(cases), "passed": passed, "failed": len(cases) - passed,
               "pass_rate": round(passed / len(cases), 4) if cases else 0.0, "metrics": metrics}
    return summary, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--spawn-rate", type=int, default=5)
    parser.add_argument("--duration", default="30s")
    parser.add_argument("--host", help="test this server instead of starting a fake-mode one")
    parser.add_argument("--llm-mode", default="fake", help="label for the report when using --host")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="locust-", ignore_cleanup_errors=True) as tmp:
        prefix = Path(tmp) / "load"
        if args.host:
            host, llm_mode = args.host, args.llm_mode
            run_locust(host, args.users, args.spawn_rate, args.duration, prefix)
        else:
            llm_mode = "fake"
            with LiveServer(llm_mode="fake") as server:
                host = server.base_url
                run_locust(host, args.users, args.spawn_rate, args.duration, prefix)
        rows = read_stats(Path(f"{prefix}_stats.csv"))

    summary, cases = summarise(rows)
    path = write_report("load", summary, cases, llm_mode=llm_mode, users=args.users,
                        duration=args.duration, host=host)
    m = summary["metrics"]
    print(f"\nrequests={m['requests']} rps={m['rps']} error_rate={m['error_rate']:.2%} "
          f"p50={m['p50_latency_ms']:.0f}ms p95={m['p95_latency_ms']:.0f}ms p99={m['p99_latency_ms']:.0f}ms")
    for c in cases:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['id']:<28} {c['reason']}")
    print(f"report: {path}")
    ok = m["p95_latency_ms"] <= P95_BUDGET_MS and m["error_rate"] <= MAX_ERROR_RATE
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
