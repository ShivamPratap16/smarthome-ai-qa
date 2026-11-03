"""Fixtures for Playwright UI tests: live server + ChatPage + screenshot on failure.

Uses pytest-playwright's built-in `page` / `browser` fixtures (Chromium, headless by default;
add `--headed` to watch). Failed tests save a full-page screenshot to test-results/.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Page

from tests.api.live_server import LiveServer
from tests.api.report_plugin import SuiteReportCollector
from tests.web.pages.chat_page import ChatPage

ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_DIR = ROOT / "test-results"

_collector = SuiteReportCollector("web", "tests/web/")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _collector.add(report)


def pytest_sessionfinish(session: pytest.Session) -> None:
    _collector.write()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """After a failed test, save a screenshot of its page (if it used one)."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        page = item.funcargs.get("page")
        if page is not None and not page.is_closed():
            SCREENSHOT_DIR.mkdir(exist_ok=True)
            name = re.sub(r"[^\w.-]+", "_", item.nodeid.split("::", 1)[-1])
            path = SCREENSHOT_DIR / f"{name}.png"
            try:
                page.screenshot(path=str(path), full_page=True)
                report.sections.append(("screenshot", str(path)))
            except Exception as exc:  # noqa: BLE001 - never hide the real failure
                report.sections.append(("screenshot", f"could not save: {exc}"))


@pytest.fixture(scope="session")
def web_server() -> Iterator[LiveServer]:
    """Real uvicorn server serving the UI + API (fake LLM)."""
    with LiveServer(llm_mode="fake") as server:
        yield server


@pytest.fixture
def chat(page: Page, web_server: LiveServer) -> ChatPage:
    """A ChatPage already opened on the live server."""
    page.set_default_timeout(10_000)
    return ChatPage(page, web_server.base_url).open()
