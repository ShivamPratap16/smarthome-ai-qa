"""Build the static quality dashboard into ``site/``.

Reads (Contract 5):
  * ``reports/<suite>/latest.json``   – the current state of each suite
  * ``reports/<suite>/<timestamp>.json`` – older runs, used for trend charts
and the gate rules in ``qa/thresholds.yaml`` (via ``qa.gate``), so the dashboard
shows exactly the same pass/fail verdict as CI.

Writes ``site/index.html`` (data embedded, works from file://), ``site/data.json``
and ``site/assets/*``. No build tools needed: plain HTML + CSS + vanilla JS.

Usage::

    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --reports-dir C:/tmp/sample --out site
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa import gate

TEMPLATE_DIR = ROOT / "dashboard"

# Plain-words description of every suite: shown on the cards and in "How this is tested".
SUITE_INFO: dict[str, dict[str, str]] = {
    "evals": {
        "title": "Answer quality",
        "tool": "Golden set · LLM judge · promptfoo",
        "what": "A fixed set of real-looking customer questions with reference answers. Each bot answer is "
                "scored for faithfulness (does it stick to the manual?), relevancy and hallucination. "
                "Prompt v1 and v2 are compared on the same questions.",
    },
    "redteam": {
        "title": "Red-team & safety",
        "tool": "Adversarial prompt suite",
        "what": "Attacks a bad actor would try: prompt injection, jailbreaks, asking the bot to leak phone "
                "numbers, how to bypass a door lock or open fan wiring. A pass means the bot refused or was blocked. "
                "Cases tagged critical must all pass.",
    },
    "tool_calling": {
        "title": "Tool calling & conversations",
        "tool": "pytest · scripted fake LLM",
        "what": "Checks the bot calls the right device API with the right arguments (e.g. set fan speed to 3), "
                "retries once when the device cloud returns 503, and remembers context across turns.",
    },
    "api": {
        "title": "API contract",
        "tool": "pytest · Schemathesis",
        "what": "Every HTTP endpoint returns the documented status codes and JSON shape, including 422 on bad "
                "input and 503 when the LLM is down. Schemathesis fuzzes the OpenAPI schema.",
    },
    "web": {
        "title": "Web chat UI",
        "tool": "Playwright",
        "what": "A real browser opens the chat page, sends messages and checks that tool chips, refusal badges "
                "and the typing indicator appear, on desktop and mobile sizes.",
    },
    "data": {
        "title": "Data quality",
        "tool": "SQL checks · DuckDB",
        "what": "Looks inside the stored chat logs: no raw phone numbers or emails, no missing latency or cost "
                "values, every tool call linked to a message.",
    },
    "load": {
        "title": "Load & latency",
        "tool": "Locust (headless)",
        "what": "Simulated users chat at the same time; the gate fails if the 95th-percentile response time "
                "goes over budget.",
    },
}
SUITE_ORDER = list(SUITE_INFO)
COMPARE_METRICS = ["pass_rate", "faithfulness", "answer_relevancy", "hallucination_rate",
                   "context_precision", "context_recall", "p95_latency_ms"]


# ---------------------------------------------------------------- reading

def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: skipping unreadable report {path}: {exc}", file=sys.stderr)
        return None


def load_history(reports_dir: Path) -> dict[str, list[dict]]:
    """Return ``{suite: [report, ...]}`` oldest first, from timestamped files (plus latest if new)."""
    history: dict[str, list[dict]] = {}
    if not reports_dir.exists():
        return history
    for folder in sorted(p for p in reports_dir.iterdir() if p.is_dir()):
        runs = [r for p in sorted(folder.glob("*.json")) if p.name != "latest.json" and (r := _read_json(p))]
        latest = _read_json(folder / "latest.json") if (folder / "latest.json").exists() else None
        if latest and all(r.get("run_id") != latest.get("run_id") for r in runs):
            runs.append(latest)
        if runs:
            runs.sort(key=lambda r: r.get("started_at") or "")
            history[folder.name] = runs
    return history


# ---------------------------------------------------------------- shaping

def _summary(report: dict) -> dict:
    s = report.get("summary", {})
    return {"total": s.get("total", 0), "passed": s.get("passed", 0), "failed": s.get("failed", 0),
            "pass_rate": gate.metric_value(report, "pass_rate"), "metrics": s.get("metrics") or {}}


def _run_point(report: dict) -> dict:
    s = _summary(report)
    return {"run_id": report.get("run_id"), "started_at": report.get("started_at"),
            "prompt_version": report.get("prompt_version"), "git_sha": report.get("git_sha"),
            "pass_rate": s["pass_rate"], "metrics": s["metrics"]}


def compute_kpis(latest: dict[str, dict]) -> dict:
    """Headline numbers. Each value is None when the suite that provides it has no report."""
    total = sum(r.get("summary", {}).get("total", 0) for r in latest.values())
    passed = sum(r.get("summary", {}).get("passed", 0) for r in latest.values())
    evals, redteam, load = latest.get("evals"), latest.get("redteam"), latest.get("load")

    block_rate = None
    if redteam:
        block_rate = gate.metric_value(redteam, "attack_block_rate")
        if block_rate is None:
            block_rate = gate.metric_value(redteam, "pass_rate")

    p95, p95_source = None, None
    for suite, rep in (("load", load), ("evals", evals)):
        if rep and gate.metric_value(rep, "p95_latency_ms") is not None:
            p95, p95_source = gate.metric_value(rep, "p95_latency_ms"), suite
            break

    cost100 = None
    if evals:
        cost100 = gate.metric_value(evals, "cost_per_100_chats")
        cost_total = gate.metric_value(evals, "cost_usd")
        n = evals.get("summary", {}).get("total") or 0
        if cost100 is None and cost_total is not None and n:
            cost100 = cost_total / n * 100  # cost_usd is the total for the run

    return {
        "overall_pass_rate": passed / total if total else None, "total_cases": total, "passed_cases": passed,
        "hallucination_rate": gate.metric_value(evals, "hallucination_rate") if evals else None,
        "redteam_block_rate": block_rate,
        "p95_latency_ms": p95, "p95_source": p95_source,
        "cost_per_100_chats": cost100,
    }


def prompt_comparison(evals_runs: list[dict]) -> dict | None:
    """Latest evals run per prompt version; None unless at least two versions exist."""
    by_version: dict[str, dict] = {}
    for rep in evals_runs:  # oldest first, so later runs overwrite
        if rep.get("prompt_version"):
            by_version[rep["prompt_version"]] = rep
    if len(by_version) < 2:
        return None
    versions = sorted(by_version)
    rows = []
    for metric in COMPARE_METRICS:
        values = {v: gate.metric_value(by_version[v], metric) for v in versions}
        if any(val is not None for val in values.values()):
            rows.append({"metric": metric, "values": values})
    return {"versions": versions, "rows": rows,
            "runs": {v: {"run_id": r.get("run_id"), "started_at": r.get("started_at"),
                         "total": r.get("summary", {}).get("total")} for v, r in by_version.items()}}


def failing_cases(latest: dict[str, dict]) -> list[dict]:
    """Every failed case from the latest reports, with its reason."""
    rows = []
    for suite in sorted(latest, key=lambda s: SUITE_ORDER.index(s) if s in SUITE_ORDER else 99):
        for case in latest[suite].get("cases", []):
            if not case.get("passed"):
                rows.append({"suite": suite, "id": case.get("id"), "category": case.get("category") or "—",
                             "reason": case.get("reason") or "", "scores": case.get("scores") or {},
                             "critical": gate.is_critical(case)})
    return rows


def repo_url() -> str | None:
    """Repository link from env (REPO_URL or GitHub Actions) or the git remote."""
    if os.environ.get("REPO_URL"):
        return os.environ["REPO_URL"]
    if os.environ.get("GITHUB_REPOSITORY"):
        return f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}"
    try:
        out = subprocess.run(["git", "remote", "get-url", "origin"], check=False, cwd=ROOT, capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if out.startswith("git@github.com:"):
        out = "https://github.com/" + out.split(":", 1)[1]
    return out.removesuffix(".git") or None


def build_data(reports_dir: Path, thresholds_path: Path) -> dict:
    """Assemble everything the page needs into one JSON-able dict."""
    thresholds = gate.load_thresholds(thresholds_path)
    latest = gate.load_latest_reports(reports_dir)
    history = load_history(reports_dir)
    baseline = gate.load_baseline(reports_dir / "baseline.json")
    checks = gate.evaluate(latest, thresholds, baseline)
    statuses = gate.suite_status(checks)

    known = SUITE_ORDER + sorted(set(latest) - set(SUITE_ORDER))
    suites = []
    for key in known:
        info = SUITE_INFO.get(key, {"title": key, "tool": "", "what": ""})
        rep = latest.get(key)
        suites.append({
            "key": key, **info, "present": rep is not None,
            "status": statuses.get(key, "skipped" if rep is None else "pass"),
            "summary": _summary(rep) if rep else None,
            "started_at": rep.get("started_at") if rep else None,
            "git_sha": rep.get("git_sha") if rep else None,
            "prompt_version": rep.get("prompt_version") if rep else None,
            "model": rep.get("model") if rep else None,
            "llm_mode": rep.get("llm_mode") if rep else None,
            "checks": [asdict(c) for c in checks if c.suite == key],
        })

    times = [r.get("started_at") for r in latest.values() if r.get("started_at")]
    newest = max(latest.values(), key=lambda r: r.get("started_at") or "", default={})
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "last_run_at": max(times) if times else None,
        # model/mode come from the eval run (the one that exercises the real LLM), not from whichever suite ran last
        "git_sha": newest.get("git_sha"),
        "model": (latest.get("evals") or newest).get("model"),
        "llm_mode": (latest.get("evals") or newest).get("llm_mode"),
        "repo_url": repo_url(),
        "sample": any(r.get("sample") for r in latest.values()),
        "gate": {"passed": gate.gate_passed(checks), "checks": [asdict(c) for c in checks],
                 "has_baseline": bool(baseline)},
        "thresholds": thresholds.get("suites", {}),
        "kpis": compute_kpis(latest),
        "suites": suites,
        "history": {s: [_run_point(r) for r in runs] for s, runs in history.items()},
        "prompt_compare": prompt_comparison(history.get("evals", [])),
        "failures": failing_cases(latest),
    }


# ---------------------------------------------------------------- writing

def write_site(data: dict, out: Path) -> Path:
    """Write index.html (with data embedded), data.json and assets into ``out``."""
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    for name in ("styles.css", "app.js"):
        shutil.copy(TEMPLATE_DIR / name, out / "assets" / name)
    payload = json.dumps(data, indent=1, ensure_ascii=False)
    (out / "data.json").write_text(payload, encoding="utf-8")
    template = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out / "index.html"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the static QA dashboard")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--thresholds", type=Path, default=gate.DEFAULT_THRESHOLDS)
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    args = parser.parse_args(argv)

    data = build_data(args.reports_dir, args.thresholds)
    index = write_site(data, args.out)
    present = sum(s["present"] for s in data["suites"])
    print(f"Dashboard written to {index} ({present}/{len(data['suites'])} suites with reports)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
