"""Generate SAMPLE report files for developing the dashboard and the gate.

These numbers are invented. Every report is marked ``"sample": true`` and the
dashboard shows a "sample data" banner for them. By default the files go into a
fresh temporary folder — never into the real ``reports/`` folder.

Usage::

    python scripts/make_sample_reports.py                 # prints the temp dir it wrote to
    python scripts/make_sample_reports.py --out C:/tmp/r  # write to a chosen folder
    python scripts/make_sample_reports.py --failing       # latest run breaks some gates
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EVAL_CASES = [
    ("aero-beep-01", "manual_qa"), ("aero-remote-pair-02", "manual_qa"), ("aero-wobble-03", "manual_qa"),
    ("lock-battery-01", "manual_qa"), ("lock-fingerprint-02", "manual_qa"), ("pure-filter-01", "manual_qa"),
    ("pure-tds-02", "manual_qa"), ("pure-leak-03", "manual_qa"), ("aero-speed-tool-01", "tool_use"),
    ("lock-status-tool-02", "tool_use"), ("warranty-check-03", "tool_use"), ("ticket-create-04", "tool_use"),
    ("aero-followup-01", "multi_turn"), ("lock-followup-02", "multi_turn"), ("pure-followup-03", "multi_turn"),
    ("hinglish-fan-01", "hinglish"), ("hinglish-lock-02", "hinglish"), ("hinglish-pure-03", "hinglish"),
    ("wiring-diy-01", "refusal"), ("lock-bypass-02", "refusal"), ("water-medical-03", "refusal"),
    ("ac-question-01", "unanswerable"), ("price-future-02", "unanswerable"), ("competitor-03", "unanswerable"),
    ("fan-503-01", "recovery"), ("lock-offline-02", "recovery"),
]
EVAL_FAIL_REASONS = [
    "Answer mentions a reset step that is not in the Aero manual (unsupported claim).",
    "Did not say 'I don't know' for an out-of-scope question; invented a price.",
    "Missed expected context aero/troubleshooting#beeping; answer too generic.",
    "Replied in English only to a Hinglish question.",
    "Retried the 503 but did not offer a service ticket.",
]
REDTEAM_CASES = [
    ("inj-ignore-previous-01", "prompt_injection", True), ("inj-system-reveal-02", "prompt_injection", True),
    ("inj-tool-abuse-03", "prompt_injection", True), ("jb-roleplay-01", "jailbreak", True),
    ("jb-dan-02", "jailbreak", False), ("pii-echo-phone-01", "pii_leak", True), ("pii-aadhaar-02", "pii_leak", True),
    ("topic-wiring-01", "blocked_topic", True), ("topic-lock-bypass-02", "blocked_topic", True),
    ("topic-water-cure-03", "blocked_topic", False), ("enc-base64-01", "obfuscation", False),
    ("enc-hinglish-02", "obfuscation", False), ("ticket-spam-01", "abuse", False), ("pricing-trick-01", "misinfo", False),
]
SIMPLE_CASES = {
    "api": [("health-ok", "smoke"), ("chat-200", "contract"), ("chat-422-empty", "validation"),
            ("chat-422-too-long", "validation"), ("reset-session", "contract"), ("devices-list", "contract"),
            ("device-404", "errors"), ("llm-down-503", "errors"), ("schemathesis-fuzz", "fuzz")],
    "web": [("send-message", "ui"), ("tool-chip-visible", "ui"), ("refusal-badge", "ui"),
            ("typing-indicator", "ui"), ("mobile-viewport", "responsive"), ("a11y-labels", "accessibility")],
    "data": [("no-raw-phone-in-logs", "pii"), ("no-raw-email-in-logs", "pii"), ("tickets-phone-masked", "pii"),
             ("latency-not-null", "completeness"), ("cost-non-negative", "validity"),
             ("tool-call-fk", "integrity"), ("prompt-version-valid", "validity")],
    "tool_calling": [("set-speed-args", "args"), ("status-before-action", "ordering"), ("warranty-serial", "args"),
                     ("ticket-phone-masked", "pii"), ("retry-503-once", "recovery"), ("max-4-rounds", "limits"),
                     ("offline-fan-message", "recovery"), ("no-tool-for-manual-q", "selection"),
                     ("multi-turn-context", "conversation"), ("reset-clears-history", "conversation")],
}


def _report(suite: str, when: datetime, summary: dict, cases: list[dict], prompt_version: str = "v2") -> dict:
    return {
        "suite": suite, "sample": True,
        "run_id": f"sample-{suite}-{when:%Y%m%d-%H%M%S}",
        "started_at": when.isoformat(timespec="seconds"),
        "git_sha": f"{random.getrandbits(28):07x}", "prompt_version": prompt_version,
        "model": "gemini-3.5-flash-lite", "llm_mode": "replay",
        "summary": summary, "cases": cases,
    }


def _summary(cases: list[dict], metrics: dict) -> dict:
    passed = sum(c["passed"] for c in cases)
    return {"total": len(cases), "passed": passed, "failed": len(cases) - passed,
            "pass_rate": round(passed / len(cases), 4), "metrics": metrics}


def eval_report(when: datetime, prompt_version: str, quality: float) -> dict:
    """Evals run: quality in 0..1 drives the pass chance and the scores."""
    cases = []
    for cid, cat in EVAL_CASES:
        faith = min(1.0, max(0.3, random.gauss(quality, 0.08)))
        passed = faith >= 0.76
        cases.append({"id": cid, "category": cat, "passed": passed,
                      "scores": {"faithfulness": round(faith, 2), "answer_relevancy": round(min(1, faith + 0.03), 2)},
                      "reason": "" if passed else random.choice(EVAL_FAIL_REASONS),
                      "latency_ms": random.randint(700, 2100)})
    halluc = sum(1 for c in cases if c["scores"]["faithfulness"] < 0.72) / len(cases)
    metrics = {"faithfulness": round(sum(c["scores"]["faithfulness"] for c in cases) / len(cases), 3),
               "answer_relevancy": round(sum(c["scores"]["answer_relevancy"] for c in cases) / len(cases), 3),
               "hallucination_rate": round(halluc, 3), "context_precision": round(quality - 0.07, 3),
               "context_recall": round(quality - 0.03, 3),
               "p95_latency_ms": sorted(c["latency_ms"] for c in cases)[int(len(cases) * 0.95) - 1],
               "cost_usd": round(len(cases) * 0.00011, 5)}
    return _report("evals", when, _summary(cases, metrics), cases, prompt_version)


def redteam_report(when: datetime, leak_critical: bool) -> dict:
    """Red-team run; optionally one critical attack gets through."""
    cases = []
    for i, (cid, cat, critical) in enumerate(REDTEAM_CASES):
        passed = not (leak_critical and i == 1) and not (not critical and random.random() < 0.08)
        cases.append({"id": cid, "category": cat, "passed": passed, "tags": ["critical"] if critical else [],
                      "scores": {}, "reason": "" if passed else "Assistant followed the injected instruction instead of refusing.",
                      "latency_ms": random.randint(300, 1200)})
    s = _summary(cases, {})
    s["metrics"]["attack_block_rate"] = s["pass_rate"]
    return _report("redteam", when, s, cases)


def simple_report(suite: str, when: datetime, fail_ids: set[str]) -> dict:
    """A pytest-style suite where named cases fail."""
    cases = [{"id": cid, "category": cat, "passed": cid not in fail_ids, "scores": {},
              "reason": "AssertionError: expected 422 but got 500" if cid in fail_ids else "",
              "latency_ms": random.randint(20, 900)} for cid, cat in SIMPLE_CASES[suite]]
    return _report(suite, when, _summary(cases, {}), cases)


def load_report(when: datetime, p95: int) -> dict:
    """Locust run summary (no per-case detail besides endpoints)."""
    cases = [{"id": "POST /v1/chat", "category": "endpoint", "passed": p95 <= 3000, "scores": {"p95_ms": p95},
              "reason": "" if p95 <= 3000 else f"p95 {p95} ms over 3000 ms budget", "latency_ms": p95},
             {"id": "GET /v1/devices", "category": "endpoint", "passed": True, "scores": {"p95_ms": 85},
              "reason": "", "latency_ms": 85}]
    metrics = {"p95_latency_ms": p95, "p50_latency_ms": int(p95 * 0.45), "rps": 18.4, "error_rate": 0.0, "users": 20}
    return _report("load", when, _summary(cases, metrics), cases)


def build_all(out: Path, failing: bool = False, runs: int = 8) -> None:
    """Write `runs` timestamped reports per suite plus latest.json into `out`."""
    random.seed(7)
    start = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)
    for i in range(runs):
        when = start + timedelta(days=i, minutes=random.randint(0, 50))
        last = i == runs - 1
        bad = failing and last
        prompt = "v1" if i < 3 else "v2"
        quality = (0.79 + 0.015 * i) if prompt == "v1" else min(0.92, 0.83 + 0.012 * i)
        reports = [
            eval_report(when, prompt, quality - (0.12 if bad else 0)),
            redteam_report(when, leak_critical=bad),
            simple_report("api", when, {"chat-422-too-long"} if i == 2 else set()),
            simple_report("web", when, {"typing-indicator"} if bad else set()),
            simple_report("data", when, {"tickets-phone-masked"} if i == 1 else set()),
            simple_report("tool_calling", when, {"retry-503-once"} if i % 3 == 0 or last else set()),
            load_report(when, 3400 if bad else random.randint(1500, 2300)),
        ]
        for rep in reports:
            folder = out / rep["suite"]
            folder.mkdir(parents=True, exist_ok=True)
            text = json.dumps(rep, indent=2)
            (folder / f"{when:%Y%m%d-%H%M%S}.json").write_text(text, encoding="utf-8")
            if last:
                (folder / "latest.json").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: new temp dir)")
    parser.add_argument("--failing", action="store_true", help="make the latest run break some gates")
    parser.add_argument("--force", action="store_true", help="allow writing into the real reports/ folder")
    args = parser.parse_args(argv)

    out = args.out or Path(tempfile.mkdtemp(prefix="sample-reports-"))
    if out.resolve() == (ROOT / "reports").resolve() and not args.force:
        print("Refusing to write SAMPLE data into the real reports/ folder (use --force).", file=sys.stderr)
        return 2
    build_all(out, failing=args.failing)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
