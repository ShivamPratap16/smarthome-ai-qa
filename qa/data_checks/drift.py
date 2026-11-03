"""Simple drift detectors: is this week's traffic different from last week's?

* PSI (Population Stability Index) for a *mix* of categories, e.g. which product users ask about.
      PSI = sum over categories of (actual% - expected%) * ln(actual% / expected%)
  Rule of thumb: < 0.10 stable, 0.10-0.25 moderate change, > 0.25 big change (alert).
* Two-proportion z-score for a *rate*, e.g. refusal rate this week vs last week.
      z = (p2 - p1) / sqrt(p * (1 - p) * (1/n1 + 1/n2)),  p = pooled rate
  |z| > 3 means the change is very unlikely to be random noise (alert).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

PSI_ALERT = 0.25
Z_ALERT = 3.0
_EPS = 1e-4   # avoids log(0) when a category is missing in one window


def psi(expected: dict[str, float], actual: dict[str, float]) -> float:
    """PSI between two count (or share) dicts over the union of their categories."""
    keys = set(expected) | set(actual)
    e_total = sum(expected.values()) or 1.0
    a_total = sum(actual.values()) or 1.0
    total = 0.0
    for k in keys:
        e = max(expected.get(k, 0) / e_total, _EPS)
        a = max(actual.get(k, 0) / a_total, _EPS)
        total += (a - e) * math.log(a / e)
    return round(total, 4)


def two_proportion_z(hits_1: int, n_1: int, hits_2: int, n_2: int) -> float:
    """z-score of the change from rate 1 (baseline) to rate 2 (current). 0 if not computable."""
    if n_1 == 0 or n_2 == 0:
        return 0.0
    p1, p2 = hits_1 / n_1, hits_2 / n_2
    pooled = (hits_1 + hits_2) / (n_1 + n_2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_1 + 1 / n_2))
    if se == 0:
        return 0.0
    return round((p2 - p1) / se, 3)


def split_weeks(days: list[str]) -> tuple[set[str], set[str]]:
    """(last_week, this_week) day sets: the 7 days ending at the newest day, and the 7 before."""
    if not days:
        return set(), set()
    end = date.fromisoformat(max(days))
    this_week = {(end - timedelta(d)).isoformat() for d in range(7)}
    last_week = {(end - timedelta(d)).isoformat() for d in range(7, 14)}
    return last_week, this_week


def weekly_drift(daily_quality: list[dict], daily_topics: list[dict]) -> dict:
    """Compare this week with last week: topic-mix PSI and refusal-rate z-score.

    `daily_quality` rows need day/turns/refusals; `daily_topics` rows need day/topic/n
    (the outputs of the monitoring SQL).
    """
    last, this = split_weeks([r["day"] for r in daily_quality] + [r["day"] for r in daily_topics])
    topics_last: dict[str, float] = {}
    topics_this: dict[str, float] = {}
    for r in daily_topics:
        bucket = topics_this if r["day"] in this else topics_last if r["day"] in last else None
        if bucket is not None:
            bucket[r["topic"]] = bucket.get(r["topic"], 0) + int(r["n"])

    def sums(days: set[str]) -> tuple[int, int]:
        rows = [r for r in daily_quality if r["day"] in days]
        return sum(int(r["refusals"] or 0) for r in rows), sum(int(r["turns"]) for r in rows)

    ref_last, n_last = sums(last)
    ref_this, n_this = sums(this)
    topic_psi = psi(topics_last, topics_this) if topics_last and topics_this else 0.0
    z = two_proportion_z(ref_last, n_last, ref_this, n_this)
    return {
        "topic_psi": topic_psi,
        "refusal_rate_last_week": round(ref_last / n_last, 4) if n_last else 0.0,
        "refusal_rate_this_week": round(ref_this / n_this, 4) if n_this else 0.0,
        "refusal_rate_z": z,
        "drift_detected": topic_psi > PSI_ALERT or abs(z) > Z_ALERT,
        "topics_last_week": topics_last,
        "topics_this_week": topics_this,
    }
