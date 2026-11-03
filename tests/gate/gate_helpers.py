"""Helpers shared by the gate/dashboard tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def make_report(suite: str, passed: int, total: int, metrics: dict | None = None,
                cases: list[dict] | None = None, **meta) -> dict:
    """Build a minimal Contract-5 report dict."""
    return {
        "suite": suite, "run_id": meta.get("run_id", f"{suite}-1"),
        "started_at": meta.get("started_at", "2026-11-01T10:00:00+00:00"),
        "git_sha": "abc1234", "prompt_version": meta.get("prompt_version", "v2"),
        "model": "test-model", "llm_mode": "fake",
        "summary": {"total": total, "passed": passed, "failed": total - passed,
                    "pass_rate": passed / total if total else 0.0, "metrics": metrics or {}},
        "cases": cases or [],
    }
