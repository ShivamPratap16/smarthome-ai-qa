"""Tests for the quality gate (qa/gate.py) using synthetic reports in tmp dirs."""

from __future__ import annotations

import json
from pathlib import Path

from gate_helpers import make_report

from qa import gate

GOOD_EVALS = {"hallucination_rate": 0.05, "faithfulness": 0.9, "p95_latency_ms": 1500}


def run_gate(reports_dir: Path, *extra: str) -> int:
    return gate.main(["--reports-dir", str(reports_dir), *extra])


def checks_for(reports_dir: Path) -> list[gate.Check]:
    return gate.evaluate(gate.load_latest_reports(reports_dir), gate.load_thresholds())


def write_baseline(reports_dir: Path, suites: dict) -> None:
    (reports_dir / "baseline.json").write_text(json.dumps({"suites": suites}), encoding="utf-8")


def test_all_good_passes(tmp_path, write_report, capsys):
    write_report(make_report("evals", 55, 60, GOOD_EVALS))
    write_report(make_report("api", 10, 10))
    assert run_gate(tmp_path) == 0
    out = capsys.readouterr().out
    assert "Quality gate PASSED" in out
    assert "| evals |" in out


def test_low_pass_rate_fails(tmp_path, write_report):
    write_report(make_report("evals", 40, 60, GOOD_EVALS))
    assert run_gate(tmp_path) == 1


def test_hallucination_over_limit_fails(tmp_path, write_report):
    write_report(make_report("evals", 58, 60, {**GOOD_EVALS, "hallucination_rate": 0.15}))
    failed = [c.check for c in checks_for(tmp_path) if c.status == gate.FAIL]
    assert failed == ["hallucination_rate <= 0.1"]


def test_missing_metric_fails(tmp_path, write_report):
    write_report(make_report("evals", 58, 60, {"hallucination_rate": 0.01}))  # no faithfulness
    assert any(c.status == gate.FAIL and "missing" in c.note for c in checks_for(tmp_path))


def test_missing_suite_is_skipped_not_failed(tmp_path, write_report):
    write_report(make_report("api", 10, 10))
    checks = checks_for(tmp_path)
    assert gate.suite_status(checks)["evals"] == gate.SKIP
    assert gate.gate_passed(checks)


def test_require_turns_missing_suite_into_failure(tmp_path, write_report):
    write_report(make_report("api", 10, 10))
    assert run_gate(tmp_path, "--require", "evals") == 1
    assert run_gate(tmp_path, "--require", "all") == 1
    assert run_gate(tmp_path, "--require", "api") == 0


def test_api_needs_every_case_to_pass(tmp_path, write_report):
    write_report(make_report("api", 9, 10))
    assert run_gate(tmp_path) == 1


def test_redteam_critical_case_must_pass(tmp_path, write_report):
    cases = [{"id": f"c{i}", "category": "inj", "passed": True, "tags": []} for i in range(39)]
    cases.append({"id": "crit", "category": "inj", "passed": False, "tags": ["critical"]})
    write_report(make_report("redteam", 39, 40, cases=cases))  # 97.5% overall, but a critical leak
    by_check = {c.check: c for c in checks_for(tmp_path) if c.suite == "redteam"}
    assert by_check["critical_pass_rate == 1"].status == gate.FAIL
    assert by_check["pass_rate >= 0.95"].status == gate.PASS


def test_severity_field_also_marks_critical():
    report = make_report("redteam", 1, 2, cases=[
        {"id": "a", "passed": True, "severity": "critical"},
        {"id": "b", "passed": False, "severity": "low"},
    ])
    assert gate.critical_pass_rate(report) == 1.0


def test_redteam_without_critical_tags_warns(tmp_path, write_report):
    write_report(make_report("redteam", 20, 20, cases=[{"id": "x", "passed": True}]))
    checks = checks_for(tmp_path)
    crit = next(c for c in checks if c.check.startswith("critical_pass_rate"))
    assert crit.status == gate.WARN
    assert gate.gate_passed(checks)


def test_load_latency_budget(tmp_path, write_report):
    write_report(make_report("load", 2, 2, {"p95_latency_ms": 4200}))
    assert run_gate(tmp_path) == 1


def test_regression_against_baseline_fails(tmp_path, write_report):
    # 0.88 is above the 0.85 threshold, but 7 points worse than the 0.95 baseline (tolerance 0.03).
    write_report(make_report("evals", 88, 100, GOOD_EVALS))
    write_baseline(tmp_path, {"evals": {"pass_rate": 0.95, "hallucination_rate": 0.05}})
    assert run_gate(tmp_path) == 1
    assert run_gate(tmp_path, "--no-baseline") == 0


def test_small_regression_within_tolerance_passes(tmp_path, write_report):
    write_report(make_report("evals", 93, 100, {**GOOD_EVALS, "hallucination_rate": 0.07}))
    write_baseline(tmp_path, {"evals": {"pass_rate": 0.95, "hallucination_rate": 0.05}})
    assert run_gate(tmp_path) == 0


def test_lower_is_better_metric_regression(tmp_path, write_report):
    write_report(make_report("evals", 95, 100, {**GOOD_EVALS, "hallucination_rate": 0.09}))
    write_baseline(tmp_path, {"evals": {"hallucination_rate": 0.02}})
    assert run_gate(tmp_path) == 1  # still under 0.10, but 7 points worse than the baseline


def test_update_baseline_then_pass(tmp_path, write_report):
    write_report(make_report("evals", 90, 100, GOOD_EVALS))
    assert run_gate(tmp_path, "--update-baseline") == 0
    saved = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert saved["suites"]["evals"]["pass_rate"] == 0.9
    assert run_gate(tmp_path) == 0


def test_step_summary_written(tmp_path, write_report, monkeypatch):
    write_report(make_report("api", 10, 10))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert run_gate(tmp_path) == 0
    assert "| Suite | Check |" in summary.read_text(encoding="utf-8")


def test_empty_reports_dir_passes_with_everything_skipped(tmp_path):
    assert run_gate(tmp_path / "does-not-exist") == 0
