"""AWS Lambda: turn the latest QA reports in S3 into CloudWatch metrics + an alarm flag.

Flow (runs on a schedule, e.g. every hour, or after each CI run):
  1. read  s3://<bucket>/reports/evals/latest.json  and  reports/redteam/latest.json
  2. pick  PassRate, HallucinationRate, P95Latency, CostUSD (+ RedteamPassRate)
  3. send them to CloudWatch with put_metric_data (namespace "HawaAssist/Quality")
  4. compare with thresholds; publish QualityAlarm = 1 if something is bad
     -> a CloudWatch alarm on HallucinationRate / QualityAlarm can e-mail the team (SNS)

Environment variables: REPORTS_BUCKET (required), REPORTS_PREFIX (default "reports"),
METRIC_NAMESPACE, MAX_HALLUCINATION_RATE (0.10), MIN_PASS_RATE (0.85),
MIN_REDTEAM_PASS_RATE (1.0), MAX_P95_LATENCY_MS (3000).

Only uses boto3, which the Lambda Python runtime already includes - no packaging needed.
"""

from __future__ import annotations

import json
import os

import boto3

NAMESPACE = "HawaAssist/Quality"

# metric name -> (suite, where in the report, CloudWatch unit)
METRIC_SOURCES = {
    "PassRate": ("evals", ("summary", "pass_rate"), "None"),
    "HallucinationRate": ("evals", ("summary", "metrics", "hallucination_rate"), "None"),
    "P95Latency": ("evals", ("summary", "metrics", "p95_latency_ms"), "Milliseconds"),
    "CostUSD": ("evals", ("summary", "metrics", "cost_usd"), "None"),
    "RedteamPassRate": ("redteam", ("summary", "pass_rate"), "None"),
}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def load_report(s3, bucket: str, key: str) -> dict | None:
    """Read one JSON report from S3; None if it does not exist yet."""
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except s3.exceptions.NoSuchKey:
        return None
    return json.loads(body)


def _dig(data: dict | None, path: tuple[str, ...]):
    for part in path:
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def extract_metrics(reports: dict[str, dict | None]) -> dict[str, float]:
    """Pick the CloudWatch metrics out of the reports ({suite: report}). Missing ones are skipped."""
    metrics = {}
    for name, (suite, path, _unit) in METRIC_SOURCES.items():
        value = _dig(reports.get(suite), path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[name] = float(value)
    return metrics


def evaluate(metrics: dict[str, float]) -> list[str]:
    """Reasons why quality is bad (empty list = all good). Thresholds come from env vars."""
    reasons = []
    if metrics.get("HallucinationRate", 0.0) > _env_float("MAX_HALLUCINATION_RATE", 0.10):
        reasons.append(f"HallucinationRate {metrics['HallucinationRate']:.3f} above limit")
    if "PassRate" in metrics and metrics["PassRate"] < _env_float("MIN_PASS_RATE", 0.85):
        reasons.append(f"PassRate {metrics['PassRate']:.3f} below limit")
    if "RedteamPassRate" in metrics and metrics["RedteamPassRate"] < _env_float("MIN_REDTEAM_PASS_RATE", 1.0):
        reasons.append(f"RedteamPassRate {metrics['RedteamPassRate']:.3f} below limit")
    if metrics.get("P95Latency", 0.0) > _env_float("MAX_P95_LATENCY_MS", 3000):
        reasons.append(f"P95Latency {metrics['P95Latency']:.0f} ms above limit")
    return reasons


def publish(cloudwatch, metrics: dict[str, float], alarm: bool, prompt_version: str,
            namespace: str = NAMESPACE) -> int:
    """Send all metrics (+ QualityAlarm 0/1) in one put_metric_data call. Returns the count."""
    units = {name: unit for name, (_s, _p, unit) in METRIC_SOURCES.items()}
    dims = [{"Name": "PromptVersion", "Value": prompt_version}]
    data = [{"MetricName": name, "Dimensions": dims, "Value": value, "Unit": units[name]}
            for name, value in metrics.items()]
    data.append({"MetricName": "QualityAlarm", "Dimensions": dims, "Value": 1.0 if alarm else 0.0,
                 "Unit": "Count"})
    cloudwatch.put_metric_data(Namespace=namespace, MetricData=data)
    return len(data)


def handler(event: dict | None, context=None) -> dict:
    """Lambda entry point. `event` may override {"bucket": ..., "prefix": ...}."""
    event = event or {}
    bucket = event.get("bucket") or os.environ["REPORTS_BUCKET"]
    prefix = (event.get("prefix") or os.environ.get("REPORTS_PREFIX", "reports")).strip("/")
    s3 = boto3.client("s3")
    cloudwatch = boto3.client("cloudwatch")

    reports = {suite: load_report(s3, bucket, f"{prefix}/{suite}/latest.json")
               for suite in ("evals", "redteam")}
    metrics = extract_metrics(reports)
    reasons = evaluate(metrics)
    if not metrics:
        reasons.append("no reports found")
    prompt_version = (reports.get("evals") or reports.get("redteam") or {}).get("prompt_version", "unknown")
    published = publish(cloudwatch, metrics, bool(reasons), prompt_version,
                        os.environ.get("METRIC_NAMESPACE", NAMESPACE))
    result = {"alarm": bool(reasons), "reasons": reasons, "metrics": metrics,
              "prompt_version": prompt_version, "published": published}
    print(json.dumps(result))   # goes to CloudWatch Logs
    return result
