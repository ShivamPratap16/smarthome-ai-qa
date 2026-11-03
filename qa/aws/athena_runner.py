"""Run the Athena versions of the data checks - on real AWS Athena, or offline with DuckDB.

Modes:
  * ``offline`` (default): DuckDB reads the local Parquet lake written by ``qa.aws.export``.
    DuckDB's SQL is very close to Athena's (Presto/Trino), so the same .sql files run unchanged.
  * ``athena``: boto3 sends each query to AWS Athena, waits for it, and reads the result rows.
    Needs AWS credentials, the tables from ``athena_queries/ddl/create_tables.sql`` and an S3
    location for query results.

    python -m qa.aws.athena_runner                                  # offline
    python -m qa.aws.athena_runner --mode athena --database hawa_chatlogs \
        --output s3://my-hawa-lake/athena-results/                  # real AWS
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import boto3

from qa.aws.export import DEFAULT_LAKE
from qa.data_checks.checks import CheckResult, load_checks, run_checks
from qa.data_checks.engines import DuckDBEngine

ATHENA_DIR = Path(__file__).resolve().parent / "athena_queries"
DDL_FILE = ATHENA_DIR / "ddl" / "create_tables.sql"


class AthenaQueryError(RuntimeError):
    """An Athena query ended in FAILED or CANCELLED."""


class AthenaEngine:
    """Engine (same ``query(sql)`` method as the local ones) backed by AWS Athena via boto3."""

    name = "athena"

    def __init__(self, database: str, output_location: str, workgroup: str = "primary",
                 client=None, poll_seconds: float = 1.0, timeout_seconds: float = 120.0):
        self.client = client or boto3.client("athena")
        self.database = database
        self.output_location = output_location
        self.workgroup = workgroup
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def query(self, sql: str) -> list[dict]:
        """Start the query, wait until it finishes, return rows as dicts (values are strings)."""
        qid = self.client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.output_location},
            WorkGroup=self.workgroup,
        )["QueryExecutionId"]
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            status = self.client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
            state = status["State"]
            if state == "SUCCEEDED":
                break
            if state in ("FAILED", "CANCELLED"):
                raise AthenaQueryError(f"{qid} {state}: {status.get('StateChangeReason', '')}")
            if time.monotonic() > deadline:
                raise AthenaQueryError(f"{qid} timed out after {self.timeout_seconds}s")
            time.sleep(self.poll_seconds)
        return self._results(qid)

    def _results(self, qid: str) -> list[dict]:
        """Read all result pages. The first row of the first page is the column header."""
        rows: list[list[str | None]] = []
        for page in self.client.get_paginator("get_query_results").paginate(QueryExecutionId=qid):
            for row in page["ResultSet"]["Rows"]:
                rows.append([cell.get("VarCharValue") for cell in row["Data"]])
        if not rows:
            return []
        header, data = rows[0], rows[1:]
        return [dict(zip(header, r)) for r in data]

    def close(self) -> None:
        """Nothing to close (kept so all engines look the same)."""


def run_athena_checks(mode: str = "offline", lake_dir: str | Path = DEFAULT_LAKE,
                      engine=None, **athena_kwargs) -> list[CheckResult]:
    """Run every check in ``athena_queries/`` and return the results."""
    if engine is None:
        engine = (DuckDBEngine.from_parquet(lake_dir) if mode == "offline"
                  else AthenaEngine(**athena_kwargs))
    try:
        return run_checks(engine, load_checks(ATHENA_DIR))
    finally:
        engine.close()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run data checks on Athena (or DuckDB offline).")
    parser.add_argument("--mode", choices=["offline", "athena"], default="offline")
    parser.add_argument("--lake", default=str(DEFAULT_LAKE))
    parser.add_argument("--database", default=os.getenv("ATHENA_DATABASE", "hawa_chatlogs"))
    parser.add_argument("--output", default=os.getenv("ATHENA_OUTPUT"),
                        help="s3://bucket/prefix/ for Athena query results")
    parser.add_argument("--workgroup", default=os.getenv("ATHENA_WORKGROUP", "primary"))
    args = parser.parse_args()
    kwargs = {}
    if args.mode == "athena":
        if not args.output:
            parser.error("--output s3://... is required in athena mode")
        kwargs = {"database": args.database, "output_location": args.output,
                  "workgroup": args.workgroup}
    elif not Path(args.lake).exists():
        parser.error(f"{args.lake} not found - run `python -m qa.aws.export` first")
    results = run_athena_checks(args.mode, args.lake, **kwargs)
    for r in results:
        print(f"  {'PASS' if r.passed else 'FAIL'}  [{r.severity:8}] {r.name}"
              + (f"  error: {r.error}" if r.error else ""))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
