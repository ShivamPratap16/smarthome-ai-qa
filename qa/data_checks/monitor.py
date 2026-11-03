"""Quality monitoring from the chat logs: refusal rate, latency, cost, tool errors and drift.

Runs the SQL files in ``monitoring/`` on any engine and turns the rows into report metrics.
"""

from __future__ import annotations

import math

from qa.data_checks.checks import MONITORING_DIR, render
from qa.data_checks.drift import weekly_drift


def run_monitoring_query(engine, name: str) -> list[dict]:
    """Run ``monitoring/<name>.sql`` and return its rows."""
    sql = (MONITORING_DIR / f"{name}.sql").read_text(encoding="utf-8")
    return engine.query(render(sql))


def p95(values: list[float]) -> float:
    """95th percentile, nearest-rank method (0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def collect_metrics(engine) -> tuple[dict, dict]:
    """Return (metrics for the report summary, details such as daily series and drift).

    metrics: refusal_rate, avg_latency_ms, p95_latency_ms, cost_usd, cost_per_conversation,
             tool_error_rate, topic_psi, refusal_rate_z, drift_detected (0/1)
    """
    quality = run_monitoring_query(engine, "daily_quality")
    tools = run_monitoring_query(engine, "daily_tool_errors")
    topics = run_monitoring_query(engine, "daily_topics")
    latencies = [float(r["latency_ms"]) for r in run_monitoring_query(engine, "latencies")]

    turns = sum(int(r["turns"]) for r in quality)
    refusals = sum(int(r["refusals"] or 0) for r in quality)
    cost = sum(float(r["cost_usd"] or 0) for r in quality)
    conversations = int(engine.query("SELECT count(*) AS n FROM conversations")[0]["n"])
    tool_calls = sum(int(r["tool_calls"]) for r in tools)
    tool_errors = sum(int(r["tool_errors"] or 0) for r in tools)
    drift = weekly_drift(quality, topics)

    metrics = {
        "refusal_rate": round(refusals / turns, 4) if turns else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": p95(latencies),
        "cost_usd": round(cost, 6),
        "cost_per_conversation": round(cost / conversations, 8) if conversations else 0.0,
        "tool_error_rate": round(tool_errors / tool_calls, 4) if tool_calls else 0.0,
        "topic_psi": drift["topic_psi"],
        "refusal_rate_z": drift["refusal_rate_z"],
        "drift_detected": int(drift["drift_detected"]),
    }
    daily = []
    tool_by_day = {r["day"]: r for r in tools}
    for r in quality:
        t = tool_by_day.get(r["day"], {})
        daily.append({
            "day": r["day"], "turns": int(r["turns"]),
            "refusal_rate": round(int(r["refusals"] or 0) / int(r["turns"]), 4),
            "avg_latency_ms": round(float(r["avg_latency_ms"] or 0), 1),
            "cost_usd": round(float(r["cost_usd"] or 0), 8),
            "cost_per_conversation": round(float(r["cost_usd"] or 0) / int(r["conversations"]), 8),
            "tool_error_rate": (round(int(t["tool_errors"] or 0) / int(t["tool_calls"]), 4)
                                if t else 0.0),
        })
    return metrics, {"daily": daily, "drift": drift, "totals": {
        "conversations": int(conversations), "bot_turns": turns, "tool_calls": tool_calls}}
