"""Turn pytest results into a Contract 5 report (reports/<suite>/latest.json).

Shared by tests/api and tests/web. Each conftest creates one collector:

    collector = SuiteReportCollector("api", "tests/api/")
    def pytest_runtest_logreport(report): collector.add(report)
    def pytest_sessionfinish(session): collector.write()

Counting rules (simple and honest):
* passed / failed tests count normally.
* xfail(strict=True) tests are KNOWN BUGS: they are listed in `cases` with
  category "known_bug" and passed=False, but left out of total/pass_rate (they are
  tracked by `metrics.known_bugs`). If a known bug gets fixed, strict xfail turns into a
  real failure ("XPASS(strict)") so somebody removes the marker.
* skipped tests are ignored.
"""

from __future__ import annotations

import os
import time

import pytest


class SuiteReportCollector:
    """Collects one case per test of a suite and writes the report at the end."""

    def __init__(self, suite: str, path_prefix: str):
        self.suite = suite
        self.path_prefix = path_prefix
        self.cases: dict[str, dict] = {}
        self.started = time.time()

    def add(self, report: pytest.TestReport) -> None:
        """Record the result of one test phase (setup / call / teardown)."""
        nodeid = report.nodeid.replace("\\", "/")
        if not nodeid.startswith(self.path_prefix):
            return
        case = self.cases.get(nodeid)
        if hasattr(report, "wasxfail"):
            if report.skipped:   # expected failure happened -> known bug
                self.cases[nodeid] = {"id": nodeid, "category": "known_bug", "passed": False,
                                      "known_bug": True, "reason": str(report.wasxfail),
                                      "latency_ms": int(report.duration * 1000)}
            return
        if report.skipped:
            return
        if report.when == "call" or report.failed:
            if case and not case["passed"] and report.when == "teardown":
                return   # keep the first failure
            self.cases[nodeid] = {
                "id": nodeid,
                "category": self._category(nodeid),
                "passed": report.passed,
                "reason": "" if report.passed else self._short_reason(report),
                "latency_ms": int(report.duration * 1000),
            }

    def _category(self, nodeid: str) -> str:
        """test file name without 'test_' and '.py', e.g. 'contract' or 'fuzz'."""
        filename = nodeid.split("::")[0].rsplit("/", 1)[-1]
        return filename.removeprefix("test_").removesuffix(".py")

    @staticmethod
    def _short_reason(report: pytest.TestReport) -> str:
        text = str(report.longrepr) if report.longrepr else "failed"
        lines = [ln for ln in text.splitlines() if ln.startswith("E ")]
        return (" | ".join(ln[2:].strip() for ln in lines[:3]) or text[-300:])[:500]

    def write(self, **extra_meta) -> None:
        """Write the report through qa.report.write_report (skipped if nothing ran)."""
        if not self.cases or os.getenv("QA_NO_REPORT") == "1":
            return
        from qa.report import write_report

        cases = list(self.cases.values())
        counted = [c for c in cases if not c.get("known_bug")]
        passed = sum(1 for c in counted if c["passed"])
        summary = {
            "total": len(counted), "passed": passed, "failed": len(counted) - passed,
            "pass_rate": round(passed / len(counted), 4) if counted else 0.0,
            "metrics": {"known_bugs": len(cases) - len(counted),
                        "duration_s": round(time.time() - self.started, 1)},
        }
        path = write_report(self.suite, summary, cases, llm_mode="fake", **extra_meta)
        print(f"\n[{self.suite}] report written: {path}")
