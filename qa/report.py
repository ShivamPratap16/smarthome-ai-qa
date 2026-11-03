"""Write test-suite reports in the shared format (SPEC Contract 5).

Every suite (evals, redteam, api, web, load, data, tool_calling) calls::

    from qa.report import write_report
    write_report("evals", summary={"total": 60, "passed": 55, "metrics": {...}},
                 cases=[{"id": "aero-beep-01", "category": "manual_qa", "passed": True}],
                 prompt_version="v2", model="gemini-...", llm_mode="replay")

and gets three files under ``reports/<suite>/``:

* ``<YYYYmmdd-HHMMSS>.json`` - this run (kept forever, the dashboard draws trends from these)
* ``latest.json``            - a copy of the newest run (the quality gate reads this)
* ``history.jsonl``          - one short line per run (summary only), easy to grep or plot

The reports folder is ``reports/`` at the repo root, or ``$REPORTS_DIR``, or the
``reports_dir=`` keyword argument (used by tests).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_sha() -> str:
    """Short git commit id of the repo, or "nogit" when git/commits are not available."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else "nogit"


def reports_root(reports_dir: str | Path | None = None) -> Path:
    """Folder that holds all suite folders."""
    return Path(reports_dir or os.getenv("REPORTS_DIR") or ROOT / "reports")


def complete_summary(summary: dict, cases: list[dict]) -> dict:
    """Fill in total / passed / failed / pass_rate from the cases when the caller left them out."""
    out = dict(summary)
    out.setdefault("total", len(cases))
    out.setdefault("passed", sum(1 for c in cases if c.get("passed")))
    out.setdefault("failed", out["total"] - out["passed"])
    if "pass_rate" not in out:
        out["pass_rate"] = round(out["passed"] / out["total"], 4) if out["total"] else 0.0
    out.setdefault("metrics", {})
    return out


def write_report(suite: str, summary: dict, cases: list[dict], **meta) -> Path:
    """Write ``reports/<suite>/<timestamp>.json`` + ``latest.json`` + a history line.

    `meta` may contain prompt_version, model, llm_mode, run_id, started_at, reports_dir and any
    extra fields you want stored at the top level of the report. Returns the timestamped path.
    """
    reports_dir = meta.pop("reports_dir", None)
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    report = {
        "suite": suite,
        "run_id": meta.pop("run_id", f"{suite}-{stamp}"),
        "started_at": meta.pop("started_at", now.isoformat(timespec="seconds")),
        "git_sha": meta.pop("git_sha", None) or git_sha(),
        "prompt_version": meta.pop("prompt_version", os.getenv("PROMPT_VERSION", "v2")),
        "model": meta.pop("model", os.getenv("GEMINI_MODEL", "")),
        "llm_mode": meta.pop("llm_mode", os.getenv("LLM_MODE", "fake")),
        **meta,
        "summary": complete_summary(summary, cases),
        "cases": cases,
    }

    folder = reports_root(reports_dir) / suite
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}.json"
    n = 1
    while path.exists():                      # two runs in the same second (e.g. --compare)
        path = folder / f"{stamp}-{n}.json"
        n += 1
    text = json.dumps(report, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    (folder / "latest.json").write_text(text, encoding="utf-8")

    line = {k: report[k] for k in ("run_id", "started_at", "git_sha", "prompt_version", "model",
                                   "llm_mode")}
    line["summary"] = report["summary"]
    with open(folder / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path
