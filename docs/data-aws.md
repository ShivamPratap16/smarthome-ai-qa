# Data checks + AWS (S3, Athena, Lambda, CloudWatch, IAM)

The assistant writes every chat into a SQLite log (`conversations`, `messages`, `tool_calls`,
`tickets`). This part of the project checks that those logs are **safe** (no personal data),
**correct** (no broken links between tables, costs add up) and **healthy** (refusal rate,
latency, cost, tool errors, drift). It also shows how the same thing runs on AWS.

Everything runs **offline and free**: no AWS account needed.

| Real AWS service | What we use locally instead |
|---|---|
| S3 (file storage) | a local folder + **moto** (a fake AWS inside Python, used in tests) |
| Athena (SQL over files in S3) | **DuckDB** reading the same Parquet files (almost the same SQL) |
| Lambda + CloudWatch | the handler is a normal Python function; moto fakes S3 + CloudWatch |
| IAM / CloudFormation | JSON/YAML files, checked by tests and `cfn-lint` (not deployed) |

## Run it

```bash
.venv/Scripts/python -m qa.data_checks.run              # seed demo logs if missing, run checks, write reports/data/latest.json
.venv/Scripts/python -m qa.data_checks.run --engine duckdb   # same checks through DuckDB
.venv/Scripts/python -m qa.data_checks.run --reseed     # rebuild the demo DB
.venv/Scripts/python -m qa.data_checks.run --db data/chatlogs.db   # check the app's real log file
.venv/Scripts/python -m qa.aws.export                   # logs -> Parquet lake in data/lake/ (dt=YYYY-MM-DD folders)
.venv/Scripts/python -m qa.aws.athena_runner            # Athena SQL, offline on DuckDB over the Parquet lake
.venv/Scripts/python -m pytest tests/data -q            # 66 tests + 3 xfail (known app bugs)
```

If numpy prints `OpenBLAS error: Memory allocation still failed`, set `OPENBLAS_NUM_THREADS=1`
(the test conftest already does this).

**Demo data.** `qa/data_checks/seed.py` runs the *real* `SupportAssistant` with a rule-based fake
LLM over 14 days of questions: manual questions, fan speed, device status, warranty, service tickets
with phone numbers, an email, an Aadhaar number, prompt injections, unsafe DIY questions, and
injected device-cloud 503 faults. So the masking and refusals in the DB are produced by the app's
own code. Only two things are simulated afterwards: timestamps are spread over 14 days, and a
realistic LLM/network delay is added to latency (the fake LLM answers in ~1 ms).

## Files

| Path | What it is |
|---|---|
| `qa/data_checks/checks/*.sql` | 18 checks. Each returns the rows that **break** a rule (0 rows = pass). Header: name, severity, what it protects. |
| `qa/data_checks/monitoring/*.sql` | Daily quality, tool errors, topic mix, latencies. |
| `qa/data_checks/engines.py` | `SqliteEngine` and `DuckDBEngine` - same `query(sql)` method. |
| `qa/data_checks/checks.py` | Loads `.sql` files, fills `{{PLACEHOLDERS}}` (regexes, prices, latency budget), runs them. |
| `qa/data_checks/drift.py` | PSI and z-score drift detectors (this week vs last week). |
| `qa/data_checks/monitor.py` | Turns monitoring rows into report metrics. |
| `qa/data_checks/seed.py`, `run.py` | Demo data generator; the runner (writes `reports/data/latest.json`). |
| `qa/aws/export.py` | SQLite -> date-partitioned Parquet -> S3 (encrypted). Blocks the export if PII is found. |
| `qa/aws/athena_queries/` | The same checks for Athena + `ddl/create_tables.sql` (external tables). |
| `qa/aws/athena_runner.py` | Runs those queries on real Athena (boto3) or offline on DuckDB. |
| `qa/aws/lambda_quality_monitor/handler.py` | Lambda: reads eval/red-team reports from S3 -> CloudWatch metrics + alarm flag. |
| `infra/iam/*.json`, `infra/template.yaml` | Least-privilege IAM policies; CloudFormation stack (bucket, Athena workgroup, Lambda, schedule, alarms). |
| `tests/data/` | pytest: planted leaks, parity SQLite = DuckDB = Athena-offline, moto S3/CloudWatch, drift, IAM. |

## The checks

| Check | Severity | Why it matters |
|---|---|---|
| `pii_phone_in_messages`, `pii_email_in_messages`, `pii_aadhaar_in_messages` | critical | Logs are read by many people/tools; personal data must be masked (DPDP Act thinking). |
| `pii_in_tool_calls`, `pii_in_tickets` | critical | The phone number passed to `create_service_ticket` is logged too. |
| `tool_call_has_parent_message`, `message_has_conversation`, `no_orphan_tickets`, `ticket_id_unique` | high | Broken links = we cannot trace which answer did what; a reused ticket id = a lost complaint. |
| `tokens_cost_non_negative`, `cost_matches_tokens`, `llm_usage_recorded` | high/medium | Cost dashboards are only as good as these numbers. Prices come from `assistant/llm.py`. |
| `refusal_consistency` | high | `refused=1` must have a reason; `blocked_by` must be a known guardrail. |
| `latency_within_bounds`, `prompt_version_set`, `valid_roles`, `duplicate_messages`, `message_after_conversation_start` | medium/low | Broken timers, missing experiment tags, double logging, clock problems. |

The PII detectors are **stricter than the app's own `mask_pii`** (e.g. they also catch a leading
`0` or `9876 543 210`). That is the point of an independent check: it found two app bugs (below).
Regexes avoid look-arounds so the same pattern works in Python, DuckDB (RE2) and Athena.
SQLite has no regex, so we register a Python function named `regexp_like` (Athena's name); in
DuckDB it is a one-line macro. One `.sql` file -> three engines.

## Monitoring + drift (in the report's `summary.metrics`)

`refusal_rate`, `avg_latency_ms`, `p95_latency_ms`, `cost_usd`, `cost_per_conversation`,
`tool_error_rate`, and two drift numbers comparing the last 7 days with the 7 before:

* **PSI** (Population Stability Index) on the topic mix (fan / lock / purifier / warranty / ticket):
  `sum((now% - before%) * ln(now% / before%))`. < 0.1 stable, 0.1-0.25 moderate, > 0.25 alert.
* **z-score** on the refusal rate (two-proportion test). |z| > 3 = very unlikely to be noise.

Daily series and drift details are stored under `monitoring` in the report for the dashboard.
Drift is a *warning* metric, not a failing check (traffic is allowed to change).

## AWS in simple words

* **S3** - storage for files ("objects") in "buckets". Our Parquet chat logs and QA reports live there.
  Parquet = a column-based, compressed file format; Athena reads it fast and cheap.
* **Partitions** - folders like `messages/dt=2026-10-28/`. A query for one week reads only 7 folders,
  and Athena charges per byte scanned, so this saves money.
* **Athena** - run SQL directly on files in S3. No server, pay per query (~$5 per TB scanned).
  The tables (`create_tables.sql`) only describe where files are. Athena SQL = Presto/Trino;
  DuckDB is close enough that our check files run on both.
* **Lambda** - run a small function without a server. Ours runs every hour (EventBridge schedule),
  reads `reports/evals/latest.json` + `reports/redteam/latest.json` from S3 and sends numbers to CloudWatch.
* **CloudWatch** - AWS monitoring. **Metrics** = numbers over time (`PassRate`, `HallucinationRate`,
  `P95Latency`, `CostUSD`, `RedteamPassRate`, `QualityAlarm`). **Alarms** watch a metric and notify
  (SNS email) when it crosses a threshold, e.g. `HallucinationRate > 0.10`. **Logs** = the Lambda's print output.
* **IAM** - who may do what. The Lambda gets a **role** (an identity AWS services can use) with a
  **least-privilege** policy: read only `reports/*` in one bucket, write metrics only to the
  `HawaAssist/Quality` namespace, write only its own logs. No `*` actions. (`PutMetricData` has
  no resource-level permission, so its `Resource` is `*` but locked by a namespace condition.)

Data-protection controls in this design: PII masked before logging (app) -> independent SQL PII
checks (QA) -> export refuses unmasked data -> S3 encryption (SSE-S3), public access blocked,
HTTPS-only bucket policy -> chat logs expire after 90 days -> least-privilege IAM -> report files
contain row ids only, never message text.

## How to run on real AWS (later)

1. Create an AWS account, an IAM user for yourself with MFA, and run `aws configure`
   (region e.g. `ap-south-1`, Mumbai). Set a billing alarm first.
2. Package the Lambda and upload it to any bucket you own:
   ```bash
   cd qa/aws/lambda_quality_monitor && zip ../../../quality-monitor.zip handler.py && cd -
   aws s3 mb s3://<your-code-bucket>
   aws s3 cp quality-monitor.zip s3://<your-code-bucket>/lambda/quality-monitor.zip
   ```
3. Deploy the stack (creates the lake bucket, Glue DB, Athena workgroup, Lambda, schedule, alarms):
   ```bash
   aws cloudformation deploy --template-file infra/template.yaml --stack-name hawa-quality \
     --capabilities CAPABILITY_IAM \
     --parameter-overrides CodeBucket=<your-code-bucket> AlertEmail=<you@example.com>
   aws cloudformation describe-stacks --stack-name hawa-quality --query "Stacks[0].Outputs"
   ```
4. Upload logs + reports: `python -m qa.aws.export --bucket <LakeBucketName> --reports`
5. In the Athena console pick workgroup `hawa-qa`, paste `qa/aws/athena_queries/ddl/create_tables.sql`
   (replace `YOUR-BUCKET`), run it.
6. Run the checks on Athena:
   ```bash
   python -m qa.aws.athena_runner --mode athena --database hawa_chatlogs \
     --workgroup hawa-qa --output s3://<LakeBucketName>/athena-results/
   ```
7. Test the Lambda: `aws lambda invoke --function-name hawa-quality-monitor out.json && cat out.json`,
   then see the metrics in CloudWatch -> Metrics -> `HawaAssist/Quality`.
8. Clean up when done: empty the bucket, then `aws cloudformation delete-stack --stack-name hawa-quality`.

Cost for this demo is close to zero (tiny files, a few Athena queries, one Lambda run per hour,
6 custom metrics ≈ $1.80/month - delete the stack after the demo).

## App bugs found by these checks (not fixed here - owned by the assistant code)

Tests are `xfail(strict=True)` in `tests/data/test_data_app_bugs.py`; they turn red once fixed.

1. **`mask_pii` misses some phone formats** (`assistant/guardrails.py`): `09876543210` (trunk
   prefix 0) and `9876 543 210` are stored raw in `messages.content`.
2. **Ticket ids collide after a restart** (`assistant/devices.py` + `store.add_ticket`): ids
   restart at `HW-100001` with every new `DeviceCloud`, while `chatlogs.db` persists, and
   `INSERT OR REPLACE` silently overwrites the older ticket. Check `ticket_id_unique` catches it.

## Interview talking points
- **Why SQL checks on logs if the app already masks PII?** Defence in depth: the app's regex can
  have gaps (it did). An independent, stricter detector on the stored data is the safety net.
- **Why DuckDB?** Same SQL style as Athena, runs in-process on Parquet, free - so CI runs the same
  queries we would run on AWS. A parity test proves SQLite, DuckDB and Athena-offline agree.
- **Why moto?** Tests talk to "AWS" through boto3 exactly like production, but nothing leaves the laptop.
- **Why a Lambda + CloudWatch alarm?** Offline evals catch regressions before release; the alarm
  catches quality dropping after release (new prompt, model update, drift) and pages someone.
