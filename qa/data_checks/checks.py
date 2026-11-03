"""Load the ``.sql`` data checks, fill in their parameters and run them on an engine.

A check is a SELECT that returns the rows that BREAK a rule. Zero rows = pass.
Each file starts with a header::

    -- name: pii_phone_in_messages
    -- severity: critical            (critical | high | medium | low)
    -- protects: what goes wrong if this check is missing

Parameters such as ``{{PHONE_RE}}`` are replaced from ``default_params()`` so prices and regexes
live in one place. The same loader is used for the Athena versions in ``qa/aws/athena_queries``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from assistant.llm import PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT

CHECKS_DIR = Path(__file__).resolve().parent / "checks"
MONITORING_DIR = Path(__file__).resolve().parent / "monitoring"

# Detectors are STRICTER than the app's mask_pii on purpose: an independent second opinion.
# Written without look-arounds so they work in Python `re`, DuckDB (RE2) and Athena (Java regex).
PHONE_RE = r"(^|[^0-9])(0|\+?91[ -]?)?[6-9]([ .-]?[0-9]){9}([^0-9]|$)"   # 10-digit Indian mobile
EMAIL_RE = r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"                      # used on lower(text)
AADHAAR_RE = r"(^|[^0-9])[0-9]{4}[ -]?[0-9]{4}[ -]?[0-9]{4}([^0-9]|$)"   # 12 digits, 4-4-4

MAX_LATENCY_MS = 30_000
SAMPLE_IDS = 5   # how many offending row ids we keep in the report (ids only, never content)


def default_params() -> dict[str, str]:
    """Values for the ``{{...}}`` placeholders in the SQL files."""
    return {
        "PHONE_RE": PHONE_RE,
        "EMAIL_RE": EMAIL_RE,
        "AADHAAR_RE": AADHAAR_RE,
        "PRICE_IN_PER_TOKEN": repr(PRICE_PER_M_INPUT / 1e6),
        "PRICE_OUT_PER_TOKEN": repr(PRICE_PER_M_OUTPUT / 1e6),
        "MAX_LATENCY_MS": str(MAX_LATENCY_MS),
    }


@dataclass
class Check:
    """One SQL check loaded from a file."""

    name: str
    severity: str
    protects: str
    sql: str
    path: Path


@dataclass
class CheckResult:
    """Outcome of running one check."""

    name: str
    severity: str
    protects: str
    passed: bool
    violations: int
    sample_ids: list = field(default_factory=list)
    error: str | None = None

    def as_case(self) -> dict:
        """Shape used in the report's ``cases`` list (Contract 5)."""
        if self.error:
            reason = f"query failed: {self.error}"
        elif self.passed:
            reason = "0 violating rows"
        else:
            reason = f"{self.violations} violating rows, e.g. row ids {self.sample_ids}"
        return {"id": self.name, "category": self.severity, "passed": self.passed,
                "reason": reason, "scores": {"violations": self.violations},
                "protects": self.protects}


def render(sql: str, params: dict[str, str] | None = None) -> str:
    """Replace ``{{KEY}}`` placeholders. Fails loudly if one is left over."""
    for key, value in (params or default_params()).items():
        sql = sql.replace("{{" + key + "}}", value)
    leftover = re.search(r"\{\{(\w+)\}\}", sql)
    if leftover:
        raise KeyError(f"no value for SQL placeholder {leftover.group(1)}")
    return sql


def parse_check(path: Path) -> Check:
    """Read one ``.sql`` file and its ``-- key: value`` header."""
    text = path.read_text(encoding="utf-8")
    header: dict[str, str] = {}
    last_key = None
    for line in text.splitlines():
        if not line.startswith("--"):
            break
        body = line[2:].strip()
        match = re.match(r"(\w+):\s*(.*)", body)
        if match and match.group(1) in ("name", "severity", "protects", "returns", "engine", "what"):
            last_key = match.group(1)
            header[last_key] = match.group(2)
        elif last_key:   # continuation line of the previous key
            header[last_key] += " " + body
    if "name" not in header:
        raise ValueError(f"{path.name}: missing '-- name:' header")
    return Check(name=header["name"], severity=header.get("severity", "medium"),
                 protects=header.get("protects", ""), sql=text, path=path)


def load_checks(folder: str | Path = CHECKS_DIR) -> list[Check]:
    """All checks in a folder, in file-name order."""
    return [parse_check(p) for p in sorted(Path(folder).glob("*.sql"))]


def run_check(engine, check: Check, params: dict[str, str] | None = None) -> CheckResult:
    """Run one check. A query error counts as a failure (a broken check must not pass silently)."""
    try:
        rows = engine.query(render(check.sql, params))
    except Exception as exc:  # noqa: BLE001 - we report any engine error as a failed check
        return CheckResult(check.name, check.severity, check.protects, False, 0, [], str(exc))
    ids = [r.get("row_id") for r in rows[:SAMPLE_IDS]]
    return CheckResult(check.name, check.severity, check.protects, not rows, len(rows), ids)


def run_checks(engine, checks: list[Check] | None = None,
               params: dict[str, str] | None = None) -> list[CheckResult]:
    """Run all checks (default: every file in ``checks/``)."""
    return [run_check(engine, c, params) for c in (checks if checks is not None else load_checks())]
