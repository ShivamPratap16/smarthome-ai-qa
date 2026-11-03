"""Tests for the dashboard builder and the sample report generator."""

from __future__ import annotations

import importlib.util
import json

from gate_helpers import ROOT, make_report


def load_script(name: str):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_script("build_dashboard")
sample = load_script("make_sample_reports")
THRESHOLDS = build.gate.DEFAULT_THRESHOLDS


def test_build_with_no_reports(tmp_path):
    data = build.build_data(tmp_path / "none", THRESHOLDS)
    assert all(not s["present"] for s in data["suites"])
    assert data["kpis"]["overall_pass_rate"] is None
    assert data["prompt_compare"] is None
    index = build.write_site(data, tmp_path / "site")
    assert "DASHBOARD_DATA" in index.read_text(encoding="utf-8")


def test_build_from_sample_reports(tmp_path):
    reports = tmp_path / "reports"
    sample.build_all(reports, failing=True)
    data = build.build_data(reports, THRESHOLDS)
    assert data["sample"] is True
    assert data["gate"]["passed"] is False
    assert len(data["history"]["evals"]) == 8
    assert data["prompt_compare"]["versions"] == ["v1", "v2"]
    assert any(f["critical"] for f in data["failures"])
    site = tmp_path / "site"
    build.write_site(data, site)
    assert json.loads((site / "data.json").read_text(encoding="utf-8"))["kpis"]["total_cases"] > 0
    assert (site / "assets" / "app.js").exists()


def test_kpis_cost_and_p95():
    latest = {
        "evals": make_report("evals", 9, 10, {"cost_usd": 0.002, "hallucination_rate": 0.1, "p95_latency_ms": 2000}),
        "load": make_report("load", 1, 1, {"p95_latency_ms": 1200}),
    }
    k = build.compute_kpis(latest)
    assert abs(k["cost_per_100_chats"] - 0.02) < 1e-9
    assert k["p95_latency_ms"] == 1200 and k["p95_source"] == "load"
    assert k["overall_pass_rate"] == 10 / 11


def test_embedded_json_cannot_close_script_tag(tmp_path, write_report):
    write_report(make_report("api", 0, 1, cases=[{"id": "x", "passed": False, "reason": "</script><b>"}]))
    data = build.build_data(tmp_path, THRESHOLDS)
    html = build.write_site(data, tmp_path / "site").read_text(encoding="utf-8")
    assert "</script><b>" not in html


def test_sample_generator_refuses_real_reports_dir():
    assert sample.main(["--out", str(ROOT / "reports")]) == 2


def test_history_includes_latest_once(tmp_path, write_report):
    write_report(make_report("api", 10, 10, run_id="r1", started_at="2026-11-01T00:00:00"), stamp="a", latest=False)
    write_report(make_report("api", 9, 10, run_id="r2", started_at="2026-11-02T00:00:00"), stamp="b")
    hist = build.load_history(tmp_path)
    assert [r["run_id"] for r in hist["api"]] == ["r1", "r2"]
