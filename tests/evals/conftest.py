"""Offline settings for the eval tests: never touch Gemini, never write to the real reports/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Fake LLM mode and a temporary reports folder for every test."""
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
