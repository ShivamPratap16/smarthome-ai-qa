"""Query engines that run the same SQL checks.

* ``SqliteEngine``  - runs directly on the app's SQLite chat-log file (fast, no copy).
* ``DuckDBEngine``  - loads the same tables into DuckDB. DuckDB speaks almost the same SQL as
  AWS Athena (Presto/Trino), so it is our free, offline "local Athena".

Every engine has one method, ``query(sql) -> list[dict]``. SQLite has no regex function, so
we register ``regexp_like(text, pattern)`` (Athena's name) as a small Python function. In DuckDB
we add a macro with the same name. That way one ``.sql`` file works in SQLite, DuckDB and Athena.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Protocol

import duckdb

# Column names + types of the 4 chat-log tables (Contract 2). Used for DuckDB and Parquet.
TABLE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "conversations": [("id", "VARCHAR"), ("user_id", "VARCHAR"), ("started_at", "VARCHAR")],
    "messages": [("id", "BIGINT"), ("conversation_id", "VARCHAR"), ("role", "VARCHAR"),
                 ("content", "VARCHAR"), ("created_at", "VARCHAR"), ("prompt_version", "VARCHAR"),
                 ("latency_ms", "BIGINT"), ("tokens_in", "BIGINT"), ("tokens_out", "BIGINT"),
                 ("cost_usd", "DOUBLE"), ("refused", "INTEGER"), ("blocked_by", "VARCHAR")],
    "tool_calls": [("id", "BIGINT"), ("message_id", "BIGINT"), ("name", "VARCHAR"),
                   ("args_json", "VARCHAR"), ("result_json", "VARCHAR"), ("error", "VARCHAR")],
    "tickets": [("id", "VARCHAR"), ("device_id", "VARCHAR"), ("issue", "VARCHAR"),
                ("phone_masked", "VARCHAR"), ("created_at", "VARCHAR")],
}


class Engine(Protocol):
    """Anything that can run a SELECT and return rows as dicts."""

    name: str

    def query(self, sql: str) -> list[dict]: ...


def _regexp_like(text: str | None, pattern: str) -> int | None:
    """SQLite version of Athena's regexp_like (search anywhere in the text)."""
    if text is None:
        return None
    return 1 if re.search(pattern, text) else 0


class SqliteEngine:
    """Run checks directly on a SQLite chat-log database."""

    name = "sqlite"

    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.create_function("regexp_like", 2, _regexp_like, deterministic=True)

    def query(self, sql: str) -> list[dict]:
        """Run one query and return all rows as dicts."""
        return [dict(r) for r in self.conn.execute(sql).fetchall()]

    def close(self) -> None:
        """Close the connection."""
        self.conn.close()


class DuckDBEngine:
    """DuckDB in memory. Create with ``from_sqlite(path)`` or ``from_parquet(lake_dir)``."""

    name = "duckdb"

    def __init__(self) -> None:
        self.conn = duckdb.connect(":memory:")
        # Athena name -> DuckDB function (both search for the pattern anywhere in the text).
        self.conn.execute("CREATE MACRO regexp_like(s, p) AS regexp_matches(s, p)")

    @classmethod
    def from_sqlite(cls, db_path: str | Path) -> DuckDBEngine:
        """Copy the 4 tables from a SQLite file into DuckDB tables with fixed types."""
        engine = cls()
        src = sqlite3.connect(str(db_path))
        try:
            for table, columns in TABLE_COLUMNS.items():
                engine.create_table(table)
                names = ", ".join(c for c, _ in columns)
                rows = src.execute(f"SELECT {names} FROM {table}").fetchall()
                if rows:
                    marks = ", ".join("?" for _ in columns)
                    engine.conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
        finally:
            src.close()
        return engine

    @classmethod
    def from_parquet(cls, lake_dir: str | Path) -> DuckDBEngine:
        """Expose ``<lake>/<table>/dt=YYYY-MM-DD/*.parquet`` as views (like Athena external tables)."""
        engine = cls()
        lake = Path(lake_dir)
        for table in TABLE_COLUMNS:
            files = sorted((lake / table).glob("dt=*/*.parquet"))
            if files:
                pattern = (lake / table).as_posix() + "/dt=*/*.parquet"
                engine.conn.execute(f"CREATE VIEW {table} AS SELECT * FROM "
                                    f"read_parquet('{pattern}', hive_partitioning = true)")
            else:   # no data exported yet -> empty table with the right columns
                engine.create_table(table)
        return engine

    def create_table(self, table: str) -> None:
        """Create an empty table with the Contract 2 columns."""
        cols = ", ".join(f"{c} {t}" for c, t in TABLE_COLUMNS[table])
        self.conn.execute(f"CREATE TABLE {table} ({cols})")

    def query(self, sql: str) -> list[dict]:
        """Run one query and return all rows as dicts."""
        cur = self.conn.execute(sql)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the connection."""
        self.conn.close()


def make_engine(kind: str, db_path: str | Path) -> SqliteEngine | DuckDBEngine:
    """Build an engine by name: 'sqlite' or 'duckdb'."""
    if kind == "sqlite":
        return SqliteEngine(db_path)
    if kind == "duckdb":
        return DuckDBEngine.from_sqlite(db_path)
    raise ValueError(f"unknown engine {kind!r}; use sqlite|duckdb")
