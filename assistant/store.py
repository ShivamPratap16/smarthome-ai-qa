"""SQLite chat log store. Everything is PII-masked BEFORE it is written.

Tables (Contract 2):
  conversations(id, user_id, started_at)
  messages(id, conversation_id, role, content, created_at, prompt_version, latency_ms,
           tokens_in, tokens_out, cost_usd, refused, blocked_by)
  tool_calls(id, message_id, name, args_json, result_json, error)
  tickets(id, device_id, issue, phone_masked, created_at)

Use ``ChatStore(":memory:")`` in tests.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from assistant.guardrails import mask_pii

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, started_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL, prompt_version TEXT,
    latency_ms INTEGER DEFAULT 0, tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0, refused INTEGER DEFAULT 0, blocked_by TEXT);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER NOT NULL REFERENCES messages(id),
    name TEXT NOT NULL, args_json TEXT, result_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY, device_id TEXT NOT NULL, issue TEXT, phone_masked TEXT, created_at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChatStore:
    """Small wrapper around one SQLite connection (thread-safe via a lock)."""

    def __init__(self, path: str | Path = "data/chatlogs.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def ensure_conversation(self, conversation_id: str, user_id: str) -> None:
        """Create the conversation row if it does not exist yet."""
        self._execute("INSERT OR IGNORE INTO conversations (id, user_id, started_at) VALUES (?, ?, ?)",
                      (conversation_id, user_id, _now()))

    def add_message(self, conversation_id: str, role: str, content: str, prompt_version: str = "",
                    latency_ms: int = 0, tokens_in: int = 0, tokens_out: int = 0,
                    cost_usd: float = 0.0, refused: bool = False, blocked_by: str | None = None) -> int:
        """Insert a (masked) message and return its id."""
        cur = self._execute(
            "INSERT INTO messages (conversation_id, role, content, created_at, prompt_version, latency_ms,"
            " tokens_in, tokens_out, cost_usd, refused, blocked_by) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (conversation_id, role, mask_pii(content), _now(), prompt_version, latency_ms,
             tokens_in, tokens_out, cost_usd, int(refused), blocked_by))
        return int(cur.lastrowid)

    def add_tool_call(self, message_id: int, name: str, args: dict, result: dict | None,
                      error: str | None) -> int:
        """Insert a (masked) tool call linked to the bot message."""
        cur = self._execute(
            "INSERT INTO tool_calls (message_id, name, args_json, result_json, error) VALUES (?,?,?,?,?)",
            (message_id, name, mask_pii(json.dumps(args, ensure_ascii=False)),
             mask_pii(json.dumps(result, ensure_ascii=False)) if result is not None else None,
             mask_pii(error) if error else None))
        return int(cur.lastrowid)

    def add_ticket(self, ticket_id: str, device_id: str, issue: str, phone: str) -> None:
        """Save a service ticket (plain INSERT: an existing ticket id is never overwritten).

        The phone number is masked first.
        """
        self._execute("INSERT INTO tickets (id, device_id, issue, phone_masked, created_at)"
                      " VALUES (?,?,?,?,?)", (ticket_id, device_id, mask_pii(issue), mask_pii(phone), _now()))

    def last_ticket_number(self) -> int:
        """Highest numeric ticket id stored so far (HW-100007 -> 100007), 0 if none."""
        rows = self.query("SELECT id FROM tickets WHERE id LIKE 'HW-%'")
        numbers = [int(r["id"][3:]) for r in rows if r["id"][3:].isdigit()]
        return max(numbers, default=0)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run a read query and return rows as dicts (handy for tests and data checks)."""
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def messages(self, conversation_id: str) -> list[dict]:
        """All messages of one conversation, oldest first."""
        return self.query("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,))

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
