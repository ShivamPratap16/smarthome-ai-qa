"""Fixtures, the `live` marker and the tool_calling report for conversation tests.

* Default: everything runs offline with scripted FakeLLM answers.
* `pytest tests/conversation -m live` (or env RUN_LIVE=1) runs the tests marked `@pytest.mark.live`,
  which talk to the real model (LLM_MODE=live, or replay/record if you set LLM_MODE yourself). They use loose,
  semantic assertions because real model wording changes between runs.
* After the session, a pass/fail case per test is written to reports/tool_calling/ (Contract 5).
  Set CONVERSATION_REPORT=0 to switch that off.
"""

from __future__ import annotations

import os
import time

import pytest

_RESULTS: dict[str, dict] = {}


def live_enabled(config: pytest.Config) -> bool:
    """Live tests run only on purpose: `-m live` or RUN_LIVE=1 (they spend Gemini quota)."""
    markexpr = config.getoption("markexpr", "") or ""
    return os.getenv("RUN_LIVE") == "1" or ("live" in markexpr and "not live" not in markexpr)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: talks to the real LLM; skipped unless -m live / RUN_LIVE=1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if live_enabled(config):
        return
    skip = pytest.mark.skip(reason="live LLM test: run with -m live (uses Gemini quota)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def offline_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline tests can never reach Gemini: fake mode and no key."""
    if "live" not in request.keywords:
        monkeypatch.setenv("LLM_MODE", "fake")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def live_llm():
    """The real model client for live tests (LLM_MODE live by default, or record/replay)."""
    from assistant.llm import get_llm
    mode = os.getenv("LLM_MODE", "fake")
    return get_llm(mode if mode in ("live", "record", "replay") else "live")


# --------------------------------------------------------------------------- report
def _is_conversation_test(item_path: str) -> bool:
    return "conversation" in item_path.replace("\\", "/").split("/")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Remember the outcome of each test (setup errors and call failures both count)."""
    path = report.nodeid.split("::")[0]
    if not _is_conversation_test(path):
        return
    category = path.rsplit("/", 1)[-1].removeprefix("test_").removesuffix(".py")
    entry = _RESULTS.setdefault(report.nodeid, {"id": report.nodeid.split("::", 1)[1],
                                                "category": category, "passed": True,
                                                "reason": "", "latency_ms": 0})
    if report.skipped and not hasattr(report, "wasxfail"):
        entry["skipped"] = True
    elif hasattr(report, "wasxfail"):
        entry["passed"] = False
        entry["known_bug"] = True
        entry["reason"] = f"known bug (xfail): {report.wasxfail}"
    elif report.failed:
        entry["passed"] = False
        entry["reason"] = str(report.longrepr).strip().splitlines()[-1][:300] if report.longrepr else "failed"
    if report.when == "call":
        entry["latency_ms"] = int(report.duration * 1000)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write reports/tool_calling/<timestamp>.json + latest.json."""
    if os.getenv("CONVERSATION_REPORT", "1") == "0" or hasattr(session.config, "workerinput"):
        return
    cases = [c for c in _RESULTS.values() if not c.pop("skipped", False)]
    if not cases:
        return
    from qa.report import write_report
    live = live_enabled(session.config)
    summary = {"metrics": {
        "known_bugs": sum(1 for c in cases if c.get("known_bug")),
        "p95_latency_ms": sorted(c["latency_ms"] for c in cases)[int(0.95 * (len(cases) - 1))],
    }}
    write_report("tool_calling", summary, cases, llm_mode="live" if live else "fake",
                 model=os.getenv("GEMINI_MODEL", "") if live else "fake-scripted",
                 run_id=f"tool_calling-{time.strftime('%Y%m%d-%H%M%S')}")
