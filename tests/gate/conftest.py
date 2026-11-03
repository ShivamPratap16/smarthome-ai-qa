"""Make the repo root importable (so `from qa import gate` works) and provide report fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def write_report(tmp_path: Path):
    """Write a report as <tmp>/<suite>/latest.json (plus a timestamped copy)."""

    def _write(report: dict, stamp: str = "20261101-100000", latest: bool = True) -> Path:
        folder = tmp_path / report["suite"]
        folder.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report)
        (folder / f"{stamp}.json").write_text(text, encoding="utf-8")
        if latest:
            (folder / "latest.json").write_text(text, encoding="utf-8")
        return folder

    return _write
