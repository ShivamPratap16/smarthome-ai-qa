"""Turn a real (production) conversation that went wrong into a golden regression test.

The story: a customer conversation fails in production -> we find it in the SQLite chat log ->
this script copies the user turns into qa/golden/golden.jsonl as a new case with
`needs_review: true` -> a human writes the correct `expected_answer` (and context ids / tools)
and removes `needs_review` -> from then on every eval run checks that this bug stays fixed.

    python -m qa.regressions.from_failure --latest-failed                 # newest failed chat
    python -m qa.regressions.from_failure --conversation demo-123         # a chosen chat
    python -m qa.regressions.from_failure --report reports/evals/latest.json --case aero-beep-01
    python -m qa.regressions.from_failure --list                          # show failed chats

"Failed" in the chat log means one of: the model refused although no guardrail fired (likely
over-refusal), a tool call returned an error, or the bot said it doesn't know.
Messages in the log are already PII-masked, so phone numbers appear as [PHONE].
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from qa.evals.checks import abstained, is_hinglish
from qa.golden.validate import GOLDEN_PATH, load_golden

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "chatlogs.db"
PLACEHOLDER_ANSWER = "TODO: write the correct answer from the manuals, then remove needs_review."

PRODUCT_WORDS = {
    "aero": ("fan", "aero", "pankha", "blade", "remote"),
    "lock": ("lock", "fingerprint", "door", "pin"),
    "pure": ("purifier", "pure ro", "water", "filter", "tds", "ro "),
}


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the chat log read-only-ish (rows as dicts)."""
    if str(db_path) != ":memory:" and not Path(db_path).exists():
        raise FileNotFoundError(f"chat log not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_conversation(conn: sqlite3.Connection, conversation_id: str) -> dict:
    """Messages and tool calls of one conversation, oldest first."""
    messages = [dict(r) for r in conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,))]
    if not messages:
        raise KeyError(f"conversation {conversation_id!r} not found")
    ids = [m["id"] for m in messages]
    marks = ",".join("?" * len(ids))
    tools = [dict(r) for r in conn.execute(
        f"SELECT * FROM tool_calls WHERE message_id IN ({marks}) ORDER BY id", ids)]
    return {"id": conversation_id, "messages": messages, "tool_calls": tools}


def failure_reason(convo: dict) -> str | None:
    """Why this conversation counts as failed, or None if it looks fine."""
    bot = [m for m in convo["messages"] if m["role"] == "assistant"]
    for m in bot:
        if m["refused"] and not m["blocked_by"]:
            return "model refused without a guardrail block (possible over-refusal)"
    errors = [t for t in convo["tool_calls"] if t["error"]]
    if errors:
        return f"tool error: {errors[0]['name']} -> {errors[0]['error']}"
    if any(abstained(m["content"]) for m in bot):
        return "bot said it doesn't know (knowledge gap or retrieval miss)"
    return None


def failed_conversations(conn: sqlite3.Connection, limit: int = 20) -> list[tuple[str, str]]:
    """(conversation_id, reason) of recent failed conversations, newest first."""
    rows = conn.execute("SELECT conversation_id, MAX(id) AS last FROM messages "
                        "GROUP BY conversation_id ORDER BY last DESC").fetchall()
    found = []
    for row in rows:
        reason = failure_reason(load_conversation(conn, row["conversation_id"]))
        if reason:
            found.append((row["conversation_id"], reason))
            if len(found) >= limit:
                break
    return found


def guess_product(text: str) -> str:
    """aero | lock | pure | general, from keywords in the user's messages."""
    low = f" {text.lower()} "
    for product, words in PRODUCT_WORDS.items():
        if any(w in low for w in words):
            return product
    return "general"


def guess_category(turns: list[str], tool_calls: list[dict]) -> str:
    """Best guess; the reviewer can change it."""
    if len(turns) > 1:
        return "multi_turn"
    if tool_calls:
        return "tool_use"
    if is_hinglish(" ".join(turns)):
        return "hinglish"
    return "manual_qa"


def make_case(convo: dict, reason: str, existing_ids: set[str], db_label: str,
              category: str | None = None) -> dict:
    """Build the new golden case (flagged needs_review)."""
    turns = [m["content"] for m in convo["messages"] if m["role"] == "user"]
    bot = [m for m in convo["messages"] if m["role"] == "assistant"]
    slug = re.sub(r"[^a-z0-9]+", "-", convo["id"].lower()).strip("-")[:30] or "chat"
    case_id, n = f"reg-{slug}", 2
    while case_id in existing_ids:
        case_id, n = f"reg-{slug}-{n}", n + 1
    observed = [f"{t['name']}({t['args_json']})" + (f" -> ERROR {t['error']}" if t["error"] else "")
                for t in convo["tool_calls"]]
    return {
        "id": case_id,
        "category": category or guess_category(turns, convo["tool_calls"]),
        "product": guess_product(" ".join(turns)),
        "turns": turns,
        "expected_answer": PLACEHOLDER_ANSWER,
        "expected_context_ids": [],
        "expected_tools": [],
        "must_refuse": category == "refusal",
        "tags": ["regression"],
        "needs_review": True,
        "notes": f"Observed tools: {observed or 'none'}. Phone numbers in turns are masked; "
                 "replace [PHONE] with a test number like 9876543210 if a ticket is expected.",
        "source": {"type": "chatlog", "db": db_label, "conversation_id": convo["id"],
                   "failure": reason, "bot_reply": (bot[-1]["content"] if bot else "")[:300],
                   "prompt_version": bot[-1]["prompt_version"] if bot else None,
                   "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }


def conversation_from_report(report_path: Path, case_id: str | None) -> tuple[str, str]:
    """(session/conversation id, reason) of a failed case in a report (given id, or lowest score)."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = [c for c in report.get("cases", []) if not c.get("passed")]
    if case_id:
        cases = [c for c in report.get("cases", []) if c.get("id") == case_id]
    if not cases:
        raise KeyError("no matching failed case in report")

    def lowest(c: dict) -> float:
        nums = [v for v in (c.get("scores") or {}).values() if isinstance(v, (int, float))]
        return min(nums) if nums else 0.0

    case = min(cases, key=lowest)
    if not case.get("session_id"):
        raise KeyError(f"case {case.get('id')} has no session_id (run evals with --log-db)")
    return case["session_id"], f"{report.get('suite')} case {case.get('id')} failed: {case.get('reason', '')}"


def append_case(case: dict, golden: Path = GOLDEN_PATH) -> None:
    """Append one JSON line to the golden file."""
    text = golden.read_text(encoding="utf-8") if golden.exists() else ""
    with open(golden, "a", encoding="utf-8") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write(json.dumps(case, ensure_ascii=False) + "\n")


def create_regression(db: str | Path, conversation_id: str | None = None, latest_failed: bool = False,
                      golden: Path = GOLDEN_PATH, category: str | None = None,
                      reason: str | None = None, dry_run: bool = False) -> dict | None:
    """Main logic, usable from tests. Returns the new case, or None if it already exists."""
    conn = connect(db)
    try:
        if latest_failed:
            failed = failed_conversations(conn, limit=1)
            if not failed:
                raise LookupError("no failed conversations in the chat log")
            conversation_id, reason = failed[0]
        if not conversation_id:
            raise ValueError("give a conversation id or latest_failed=True")
        convo = load_conversation(conn, conversation_id)
    finally:
        conn.close()
    existing = load_golden(golden) if golden.exists() else []
    if any((c.get("source") or {}).get("conversation_id") == conversation_id for c in existing):
        return None
    reason = reason or failure_reason(convo) or "marked as failed by a reviewer"
    case = make_case(convo, reason, {c["id"] for c in existing}, Path(str(db)).name, category)
    if not dry_run:
        append_case(case, golden)
    return case


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Production failure -> golden regression case")
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite chat log (default data/chatlogs.db)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--conversation", help="conversation id")
    src.add_argument("--latest-failed", action="store_true", help="newest failed conversation")
    src.add_argument("--report", type=Path, help="report JSON; uses the case's session_id")
    src.add_argument("--list", action="store_true", help="list recent failed conversations")
    p.add_argument("--case", help="case id inside --report (default: lowest-scoring failed case)")
    p.add_argument("--category", help="override the guessed category")
    p.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    p.add_argument("--dry-run", action="store_true", help="print the case, do not write it")
    args = p.parse_args(argv)

    try:
        if args.list:
            conn = connect(args.db)
            for cid, why in failed_conversations(conn):
                print(f"{cid:<40} {why}")
            conn.close()
            return 0
        reason = None
        conversation = args.conversation
        if args.report:
            conversation, reason = conversation_from_report(args.report, args.case)
        case = create_regression(args.db, conversation, args.latest_failed, args.golden,
                                 args.category, reason, args.dry_run)
    except (FileNotFoundError, KeyError, LookupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if case is None:
        print("this conversation is already in the golden set - nothing added")
        return 0
    print(json.dumps(case, indent=2, ensure_ascii=False))
    print("dry run - not written" if args.dry_run else
          f"added {case['id']} to {args.golden} (needs_review: write expected_answer, then remove the flag)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
