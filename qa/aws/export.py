"""Export the chat logs to a Parquet "data lake" and (optionally) upload it to S3.

Layout (Hive-style partitions, what Athena/Glue expect)::

    <lake>/messages/dt=2026-10-28/data_0.parquet
    <lake>/tool_calls/dt=2026-10-28/data_0.parquet
    <lake>/conversations/dt=.../  <lake>/tickets/dt=.../

``dt`` = the day of the row, so Athena can read only the days a query needs (cheaper, faster).

Data-protection control: before anything is exported we run the *critical* PII checks on the
source DB. If raw phone numbers / emails / Aadhaar numbers are found, the export is refused -
unmasked logs must never reach the shared data lake.

    python -m qa.aws.export --db data/demo_chatlogs.db --lake data/lake
    python -m qa.aws.export --bucket my-hawa-lake            # + upload (real AWS, env credentials)
    python -m qa.aws.export --bucket my-hawa-lake --reports  # also upload reports/*/latest.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import boto3

from qa.data_checks.checks import load_checks, run_checks
from qa.data_checks.engines import DuckDBEngine, SqliteEngine
from qa.data_checks.seed import DEFAULT_DEMO_DB, seed_demo_db

DEFAULT_LAKE = Path("data/lake")
DEFAULT_PREFIX = "chatlogs"

# How each table gets its partition day.
PARTITION_SQL = {
    "conversations": "SELECT *, substr(started_at, 1, 10) AS dt FROM conversations",
    "messages": "SELECT *, substr(created_at, 1, 10) AS dt FROM messages",
    # Orphan tool calls (no parent message) land in today's partition so checks still see them.
    "tool_calls": ("SELECT tc.*, coalesce(substr(m.created_at, 1, 10), CAST(current_date AS VARCHAR)) "
                   "AS dt FROM tool_calls tc LEFT JOIN messages m ON m.id = tc.message_id"),
    "tickets": "SELECT *, substr(created_at, 1, 10) AS dt FROM tickets",
}


class UnmaskedPIIError(RuntimeError):
    """Raised when the source logs still contain personal data."""


def pii_gate(db_path: str | Path) -> None:
    """Refuse to export if any critical (PII) check fails on the source DB."""
    engine = SqliteEngine(db_path)
    try:
        critical = [c for c in load_checks() if c.severity == "critical"]
        failed = [r for r in run_checks(engine, critical) if not r.passed]
    finally:
        engine.close()
    if failed:
        names = ", ".join(f"{r.name} ({r.violations} rows)" for r in failed)
        raise UnmaskedPIIError(f"export blocked, unmasked PII found: {names}")


def export_to_parquet(db_path: str | Path, lake_dir: str | Path = DEFAULT_LAKE,
                      check_pii: bool = True) -> list[Path]:
    """Write every table as date-partitioned Parquet under `lake_dir`. Returns the files written."""
    if check_pii:
        pii_gate(db_path)
    lake = Path(lake_dir)
    engine = DuckDBEngine.from_sqlite(db_path)
    try:
        for table, select in PARTITION_SQL.items():
            target = lake / table
            if target.exists():
                shutil.rmtree(target)        # full refresh: the lake mirrors the DB
            if not engine.query(f"SELECT count(*) AS n FROM {table}")[0]["n"]:
                continue
            target.mkdir(parents=True, exist_ok=True)
            engine.conn.execute(f"COPY ({select}) TO '{target.as_posix()}' "
                                "(FORMAT parquet, PARTITION_BY (dt), OVERWRITE_OR_IGNORE true)")
    finally:
        engine.close()
    return sorted(lake.glob("*/dt=*/*.parquet"))


def upload_dir(local_dir: str | Path, bucket: str, prefix: str = DEFAULT_PREFIX,
               s3=None) -> list[str]:
    """Upload every file under `local_dir` to s3://bucket/prefix/<same relative path>.

    Objects are encrypted at rest (SSE-S3). Returns the uploaded keys.
    """
    s3 = s3 or boto3.client("s3")
    local_dir = Path(local_dir)
    keys = []
    for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
        key = f"{prefix.strip('/')}/{path.relative_to(local_dir).as_posix()}"
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
        keys.append(key)
    return keys


def upload_reports(reports_dir: str | Path, bucket: str, prefix: str = "reports", s3=None) -> list[str]:
    """Upload reports/<suite>/latest.json files (what the quality-monitor Lambda reads)."""
    s3 = s3 or boto3.client("s3")
    keys = []
    for path in sorted(Path(reports_dir).glob("*/latest.json")):
        key = f"{prefix}/{path.parent.name}/latest.json"
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ServerSideEncryption": "AES256",
                                                          "ContentType": "application/json"})
        keys.append(key)
    return keys


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export chat logs to Parquet (+ upload to S3).")
    parser.add_argument("--db", default=str(DEFAULT_DEMO_DB))
    parser.add_argument("--lake", default=str(DEFAULT_LAKE))
    parser.add_argument("--bucket", default=os.getenv("HAWA_LAKE_BUCKET"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--reports", action="store_true", help="also upload reports/*/latest.json")
    args = parser.parse_args()

    if not Path(args.db).exists():
        seed_demo_db(args.db)
    try:
        files = export_to_parquet(args.db, args.lake)
    except UnmaskedPIIError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"wrote {len(files)} parquet files under {args.lake}")
    if args.bucket:
        keys = upload_dir(args.lake, args.bucket, args.prefix)
        print(f"uploaded {len(keys)} files to s3://{args.bucket}/{args.prefix}/")
        if args.reports:
            print(f"uploaded {len(upload_reports('reports', args.bucket))} reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
