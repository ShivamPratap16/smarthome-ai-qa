"""Fixtures for API tests: a live uvicorn server (fake LLM) + an ApiClient for it."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from tests.api.api_client import ApiClient
from tests.api.contract import OpenApiValidator
from tests.api.live_server import LiveServer
from tests.api.report_plugin import SuiteReportCollector

_collector = SuiteReportCollector("api", "tests/api/")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _collector.add(report)


def pytest_sessionfinish(session: pytest.Session) -> None:
    _collector.write()


@pytest.fixture(scope="session")
def live_server() -> Iterator[LiveServer]:
    """One real server for the whole API session (random port, temp db, LLM_MODE=fake)."""
    with LiveServer(llm_mode="fake") as server:
        yield server


@pytest.fixture(scope="session")
def api(live_server: LiveServer) -> Iterator[ApiClient]:
    """ApiClient talking to the live server over HTTP."""
    client = ApiClient.for_url(live_server.base_url)
    yield client
    client.close()


@pytest.fixture(scope="session")
def openapi(api: ApiClient) -> OpenApiValidator:
    """Validator built from the server's own /openapi.json."""
    return OpenApiValidator(api.openapi())


@pytest.fixture
def session_id() -> str:
    """A fresh chat session per test, so tests never share history."""
    return f"api-{uuid.uuid4().hex[:8]}"
