"""Run all SQL data checks + quality monitoring on the chat logs and write the "data" report.

    python -m qa.data_checks.run                       # demo DB (created if missing), SQLite
    python -m qa.data_checks.run --engine duckdb       # same checks through DuckDB ("local Athena")
    python -m qa.data_checks.run --db data/chatlogs.db # the app's real log file

Exit code 1 if any check fails (so CI can block a release that leaks PII into logs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qa.data_checks.checks import load_checks, run_checks
from qa.data_checks.engines import make_engine
from qa.data_checks.monitor import collect_metrics
from qa.data_checks.seed import DEFAULT_DEMO_DB, seed_demo_db
from qa.report import write_report


def run(db_path: str | Path = DEFAULT_DEMO_DB, engine_kind: str = "sqlite",
        reports_dir: str | Path | None = None, write: bool = True) -> dict:
    """Seed the DB if needed, run checks + monitoring, write the report. Returns the summary."""
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"{db_path} not found -> seeding a demo database (fake LLM, offline)")
        seed_demo_db(db_path)
    engine = make_engine(engine_kind, db_path)
    try:
        results = run_checks(engine, load_checks())
        metrics, details = collect_metrics(engine)
    finally:
        engine.close()

    cases = [r.as_case() for r in results]
    passed = sum(1 for c in cases if c["passed"])
    critical_failed = [r.name for r in results if not r.passed and r.severity == "critical"]
    summary = {"total": len(cases), "passed": passed, "failed": len(cases) - passed,
               "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
               "metrics": {**metrics, "critical_failures": len(critical_failed)}}
    if write:
        kwargs = {"reports_dir": reports_dir} if reports_dir else {}
        path = write_report("data", summary, cases, engine=engine_kind, database=str(db_path),
                            monitoring=details, **kwargs)
        print(f"report: {path}")
    _print_table(results, metrics)
    return summary


def _print_table(results, metrics: dict) -> None:
    """Small console summary."""
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {status}  [{r.severity:8}] {r.name}" + ("" if r.passed else f"  ({r.violations} rows)"))
    print("  metrics: " + ", ".join(f"{k}={v}" for k, v in metrics.items()))


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SQL data-quality and data-protection checks.")
    parser.add_argument("--db", default=str(DEFAULT_DEMO_DB), help="SQLite chat-log file")
    parser.add_argument("--engine", choices=["sqlite", "duckdb"], default="sqlite")
    parser.add_argument("--reseed", action="store_true", help="recreate the demo DB first")
    args = parser.parse_args()
    if args.reseed:
        seed_demo_db(args.db)
    summary = run(args.db, args.engine)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
