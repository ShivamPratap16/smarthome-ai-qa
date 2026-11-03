"""qa/report.py writes Contract-5 reports that the gate and dashboard can read."""

from __future__ import annotations

import json
from pathlib import Path

from qa.report import complete_summary, git_sha, write_report


def test_write_report_creates_timestamped_latest_and_history(tmp_path: Path):
    cases = [{"id": "a", "category": "manual_qa", "passed": True},
             {"id": "b", "category": "refusal", "passed": False, "reason": "answered"}]
    path = write_report("evals", {"metrics": {"faithfulness": 0.9}}, cases, prompt_version="v2",
                        model="fake", llm_mode="fake", reports_dir=tmp_path)
    folder = tmp_path / "evals"
    assert path.parent == folder and path.name != "latest.json"
    latest = json.loads((folder / "latest.json").read_text(encoding="utf-8"))
    assert latest == json.loads(path.read_text(encoding="utf-8"))
    for key in ("suite", "run_id", "started_at", "git_sha", "prompt_version", "model", "llm_mode",
                "summary", "cases"):
        assert key in latest
    assert latest["summary"] == {"metrics": {"faithfulness": 0.9}, "total": 2, "passed": 1,
                                 "failed": 1, "pass_rate": 0.5}
    history = (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1 and json.loads(history[0])["summary"]["pass_rate"] == 0.5


def test_two_runs_in_same_second_do_not_overwrite(tmp_path: Path):
    p1 = write_report("redteam", {}, [], reports_dir=tmp_path)
    p2 = write_report("redteam", {}, [], reports_dir=tmp_path)
    assert p1 != p2 and p1.exists() and p2.exists()
    assert len((tmp_path / "redteam" / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_extra_meta_and_explicit_summary_values_are_kept(tmp_path: Path):
    path = write_report("load", {"total": 10, "passed": 9, "pass_rate": 0.9}, [], reports_dir=tmp_path,
                        run_id="custom", sample=True)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["run_id"] == "custom" and report["sample"] is True
    assert report["summary"]["failed"] == 1 and report["summary"]["pass_rate"] == 0.9


def test_reports_dir_env_is_used():   # conftest sets REPORTS_DIR to a temp folder
    path = write_report("api", {}, [{"id": "x", "passed": True}])
    assert "reports" in str(path) and path.exists()


def test_complete_summary_handles_empty():
    assert complete_summary({}, [])["pass_rate"] == 0.0


def test_git_sha_is_a_string():
    sha = git_sha()
    assert isinstance(sha, str) and sha
