"""Production failure (chat log) -> new golden regression case."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from assistant.bot import SupportAssistant
from assistant.devices import DeviceCloud
from assistant.llm import FakeLLM, LLMResponse, ToolCall
from assistant.store import ChatStore
from qa.golden.validate import GOLDEN_PATH, load_golden, validate_file
from qa.regressions import from_failure


@pytest.fixture
def chat_db(tmp_path: Path) -> Path:
    """A chat log with one good chat, one over-refusal and one tool error (newest)."""
    db = tmp_path / "chatlogs.db"
    store = ChatStore(db)
    SupportAssistant(llm=FakeLLM(["Boost runs at 380 RPM for 2 hours."]), store=store) \
        .chat("good-1", "How long does Boost run?")
    SupportAssistant(llm=FakeLLM(["Sorry, I can't help with that."]), store=store) \
        .chat("overrefusal-1", "How do I clean the fan blades?")
    bot = SupportAssistant(llm=FakeLLM([
        LLMResponse("", [ToolCall("set_fan_speed", {"device_id": "fan-bedroom", "speed": 2})]),
        "The bedroom fan is offline."]), devices=DeviceCloud(), store=store)
    bot.chat("toolerr-1", "Is the bedroom fan on?")
    bot.chat("toolerr-1", "Set it to speed 2, call me on 9876543210")
    store.close()
    return db


@pytest.fixture
def golden_copy(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    shutil.copy(GOLDEN_PATH, path)
    return path


def test_lists_failed_conversations_newest_first(chat_db: Path):
    conn = from_failure.connect(chat_db)
    failed = from_failure.failed_conversations(conn)
    conn.close()
    assert [cid for cid, _ in failed] == ["toolerr-1", "overrefusal-1"]
    assert "tool error" in failed[0][1] and "over-refusal" in failed[1][1]


def test_latest_failed_becomes_reviewable_golden_case(chat_db: Path, golden_copy: Path):
    before = len(load_golden(golden_copy))
    case = from_failure.create_regression(chat_db, latest_failed=True, golden=golden_copy)
    cases = load_golden(golden_copy)
    assert len(cases) == before + 1 and cases[-1] == case
    assert case["needs_review"] is True and case["category"] == "multi_turn"
    assert case["product"] == "aero" and case["tags"] == ["regression"]
    assert case["turns"][0] == "Is the bedroom fan on?"
    assert "9876543210" not in " ".join(case["turns"]) and "[PHONE]" in case["turns"][1]  # log is masked
    assert case["source"]["conversation_id"] == "toolerr-1"
    assert validate_file(golden_copy) == []                 # still a valid golden file


def test_same_conversation_is_not_added_twice(chat_db: Path, golden_copy: Path):
    assert from_failure.create_regression(chat_db, "overrefusal-1", golden=golden_copy) is not None
    assert from_failure.create_regression(chat_db, "overrefusal-1", golden=golden_copy) is None


def test_dry_run_does_not_write(chat_db: Path, golden_copy: Path):
    text = golden_copy.read_text(encoding="utf-8")
    case = from_failure.create_regression(chat_db, "overrefusal-1", golden=golden_copy, dry_run=True)
    assert case["category"] == "manual_qa" and golden_copy.read_text(encoding="utf-8") == text


def test_from_report_uses_session_id(chat_db: Path, golden_copy: Path, tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"suite": "evals", "cases": [
        {"id": "a", "passed": True, "scores": {"x": 1.0}, "session_id": "good-1"},
        {"id": "b", "passed": False, "scores": {"faithfulness": 0.1}, "session_id": "overrefusal-1",
         "reason": "low faithfulness"}]}), encoding="utf-8")
    code = from_failure.main(["--db", str(chat_db), "--report", str(report), "--golden", str(golden_copy)])
    assert code == 0
    case = load_golden(golden_copy)[-1]
    assert case["source"]["conversation_id"] == "overrefusal-1" and "low faithfulness" in case["source"]["failure"]


def test_cli_errors_are_friendly(tmp_path: Path, golden_copy: Path):
    assert from_failure.main(["--db", str(tmp_path / "missing.db"), "--latest-failed",
                              "--golden", str(golden_copy)]) == 1


def test_needs_review_cases_are_skipped_by_eval_runner(chat_db: Path, golden_copy: Path):
    from qa.evals.run import select_cases
    from_failure.create_regression(chat_db, latest_failed=True, golden=golden_copy)
    assert all(not c.get("needs_review") for c in select_cases(load_golden(golden_copy)))
