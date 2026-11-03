"""Quality gate: decide if the current build is good enough to merge.

It reads every ``reports/<suite>/latest.json`` (Contract 5), checks each suite
against the rules in ``qa/thresholds.yaml`` and, optionally, against a saved
baseline (``reports/baseline.json``) so a pull request is blocked when the bot
got measurably worse.

Usage::

    python -m qa.gate                      # check, print table, exit 0/1
    python -m qa.gate --require evals      # a missing evals report is a failure
    python -m qa.gate --require all        # every configured suite must exist
    python -m qa.gate --update-baseline    # save current numbers as the new baseline

Exit code 0 = gate passed, 1 = gate failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_THRESHOLDS = Path(__file__).resolve().parent / "thresholds.yaml"

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skipped"
ICONS = {PASS: "✅ pass", FAIL: "❌ FAIL", WARN: "⚠️ warn", SKIP: "⏭️ skipped"}


@dataclass
class Check:
    """One line of the gate table."""

    suite: str
    check: str          # e.g. "pass_rate >= 0.85" or "regression: pass_rate"
    actual: float | None
    limit: str          # human readable limit, e.g. ">= 0.85"
    status: str         # pass | fail | warn | skipped
    note: str = ""


# ---------------------------------------------------------------- loading

def load_thresholds(path: Path = DEFAULT_THRESHOLDS) -> dict:
    """Load the thresholds YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_latest_reports(reports_dir: Path = DEFAULT_REPORTS) -> dict[str, dict]:
    """Return ``{suite: report}`` for every ``reports/<suite>/latest.json`` found."""
    reports: dict[str, dict] = {}
    if not reports_dir.exists():
        return reports
    for latest in sorted(reports_dir.glob("*/latest.json")):
        with open(latest, encoding="utf-8") as f:
            report = json.load(f)
        reports[report.get("suite") or latest.parent.name] = report
    return reports


def load_baseline(path: Path) -> dict[str, dict]:
    """Return ``{suite: {metric: value}}`` from a baseline file, or {} if absent."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("suites", {})


# ---------------------------------------------------------------- metrics

def is_critical(case: dict) -> bool:
    """A case is critical if tagged "critical" or has severity "critical"."""
    tags = case.get("tags") or []
    return "critical" in tags or case.get("severity") == "critical"


def critical_pass_rate(report: dict) -> float | None:
    """Pass rate over critical cases only; None if there are no critical cases."""
    critical = [c for c in report.get("cases", []) if is_critical(c)]
    if not critical:
        return None
    return sum(1 for c in critical if c.get("passed")) / len(critical)


def metric_value(report: dict, metric: str) -> float | None:
    """Read one metric from a report (see thresholds.yaml for the lookup rules)."""
    summary = report.get("summary", {})
    if metric == "critical_pass_rate":
        return critical_pass_rate(report)
    if metric == "pass_rate":
        if summary.get("pass_rate") is not None:
            return float(summary["pass_rate"])
        total = summary.get("total") or 0
        return summary.get("passed", 0) / total if total else None
    value = (summary.get("metrics") or {}).get(metric)
    return None if value is None else float(value)


def compare(actual: float, op: str, limit: float) -> bool:
    """Return True if ``actual op limit`` holds (with a tiny float tolerance)."""
    eps = 1e-9
    if op == ">=":
        return actual >= limit - eps
    if op == "<=":
        return actual <= limit + eps
    if op == "==":
        return abs(actual - limit) <= eps
    raise ValueError(f"unknown operator {op!r}")


# ---------------------------------------------------------------- evaluation

def check_thresholds(suite: str, report: dict, rules: list[dict]) -> list[Check]:
    """Check one suite's report against its threshold rules."""
    checks = []
    for rule in rules:
        metric, op, limit = rule["metric"], rule["op"], float(rule["value"])
        actual = metric_value(report, metric)
        name = f"{metric} {op} {limit:g}"
        if actual is None and metric == "critical_pass_rate":
            checks.append(Check(suite, name, None, f"{op} {limit:g}", WARN, "no cases tagged critical"))
        elif actual is None:
            checks.append(Check(suite, name, None, f"{op} {limit:g}", FAIL, "metric missing from report"))
        else:
            status = PASS if compare(actual, op, limit) else FAIL
            checks.append(Check(suite, name, actual, f"{op} {limit:g}", status))
    return checks


def check_regression(suite: str, report: dict, baseline: dict, regression_cfg: dict) -> list[Check]:
    """Compare a suite's metrics with the baseline; fail if worse by more than the tolerance."""
    checks = []
    lower_better = set(regression_cfg.get("lower_is_better", []))
    tolerances = regression_cfg.get("tolerances", {})
    for metric, base in baseline.items():
        if metric not in tolerances or base is None:
            continue
        actual = metric_value(report, metric)
        if actual is None:
            continue
        tol = float(tolerances[metric])
        worse_by = (actual - base) if metric in lower_better else (base - actual)
        direction = "<=" if metric in lower_better else ">="
        limit = base + tol if metric in lower_better else base - tol
        status = FAIL if worse_by > tol + 1e-9 else PASS
        note = f"baseline {base:g}, allowed worsening {tol:g}"
        checks.append(Check(suite, f"regression: {metric}", actual, f"{direction} {limit:g}", status, note))
    return checks


def evaluate(
    reports: dict[str, dict],
    thresholds: dict,
    baseline: dict[str, dict] | None = None,
    required: set[str] | None = None,
) -> list[Check]:
    """Run every threshold and regression check. Returns all check rows."""
    baseline = baseline or {}
    required = required or set()
    suites_cfg = thresholds.get("suites", {})
    if "all" in required:
        required = set(suites_cfg)
    regression_cfg = thresholds.get("regression", {})

    checks: list[Check] = []
    for suite, cfg in suites_cfg.items():
        report = reports.get(suite)
        if report is None:
            status = FAIL if suite in required else SKIP
            checks.append(Check(suite, "report present", None, "latest.json", status, "no report found"))
            continue
        checks += check_thresholds(suite, report, cfg.get("rules", []))
        if suite in baseline:
            checks += check_regression(suite, report, baseline[suite], regression_cfg)
    return checks


def suite_status(checks: list[Check]) -> dict[str, str]:
    """Collapse check rows into one status per suite (fail > warn > pass; skipped if no report)."""
    result: dict[str, str] = {}
    order = {SKIP: 0, PASS: 1, WARN: 2, FAIL: 3}
    for c in checks:
        current = result.get(c.suite)
        if current is None or order[c.status] > order[current]:
            result[c.suite] = c.status
    return result


def gate_passed(checks: list[Check]) -> bool:
    """The gate passes when no check failed."""
    return all(c.status != FAIL for c in checks)


def make_baseline(reports: dict[str, dict], thresholds: dict) -> dict:
    """Build a baseline document from the current reports (only metrics with a tolerance)."""
    tolerances = thresholds.get("regression", {}).get("tolerances", {})
    suites = {}
    for suite, report in reports.items():
        values = {m: metric_value(report, m) for m in tolerances}
        suites[suite] = {m: v for m, v in values.items() if v is not None}
    return {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), "suites": suites}


# ---------------------------------------------------------------- output

def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 10:
        return f"{value:,.0f}"
    return f"{value:.3f}".rstrip("0").rstrip(".") if value != 0 else "0"


def render_markdown(checks: list[Check]) -> str:
    """Render the gate result as a markdown table with a verdict line."""
    verdict = "✅ **Quality gate PASSED**" if gate_passed(checks) else "❌ **Quality gate FAILED**"
    lines = [
        "## Quality gate",
        "",
        verdict,
        "",
        "| Suite | Check | Actual | Limit | Status | Note |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c.suite} | {c.check} | {_fmt(c.actual)} | {c.limit} | {ICONS[c.status]} | {c.note} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Quality gate over reports/*/latest.json")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--baseline", type=Path, default=None, help="default: <reports-dir>/baseline.json")
    parser.add_argument("--no-baseline", action="store_true", help="skip the regression check")
    parser.add_argument("--require", nargs="*", default=[], help="suites that must have a report ('all' for every suite)")
    parser.add_argument("--update-baseline", action="store_true", help="write current metrics as the new baseline and exit")
    parser.add_argument("--json", type=Path, default=None, help="also write the check rows as JSON here")
    args = parser.parse_args(argv)

    thresholds = load_thresholds(args.thresholds)
    reports = load_latest_reports(args.reports_dir)
    baseline_path = args.baseline or (args.reports_dir / "baseline.json")

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(make_baseline(reports, thresholds), f, indent=2)
        print(f"Baseline written to {baseline_path}")
        return 0

    baseline = {} if args.no_baseline else load_baseline(baseline_path)
    checks = evaluate(reports, thresholds, baseline, set(args.require))
    table = render_markdown(checks)
    sys.stdout.reconfigure(encoding="utf-8")
    print(table)

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(table)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in checks], f, indent=2)
    return 0 if gate_passed(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
