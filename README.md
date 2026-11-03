# SmartHome AI QA

[![CI](https://github.com/ShivamPratap16/smarthome-ai-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/ShivamPratap16/smarthome-ai-qa/actions/workflows/ci.yml)

**A quality-engineering system for an LLM customer-support assistant**, built the way an AI QA team
would run it: golden-set evaluations, red-teaming, tool-calling and multi-turn tests, API / web / load
automation, SQL data-protection checks, AWS monitoring, and a CI quality gate that blocks a release
when the AI gets worse.

**Live quality dashboard:** https://shivampratap16.github.io/smarthome-ai-qa/

The system under test is a small support bot for **Hawa Home**, a *fictional* smart-home brand
(Aero smart fan, Lock S1 smart lock, Pure RO purifier). It answers from product manuals (RAG),
controls devices through tools (`get_device_status`, `set_fan_speed`, `check_warranty`,
`create_service_ticket`), handles multi-turn chats and Hinglish, and runs on Gemini.
The bot is deliberately small; **the quality system around it is the project.**

## What the test suites found

Building the suites found **13 real defects in the assistant**. Each one was first captured as a
strict `xfail` test, then fixed, and the same test now passes:

- **Indirect prompt injection**: text planted in a device name, a tool result, or a manual page could
  make a persuaded model create a ticket. Fix: untrusted data is sanitised and wrapped as data, and
  a code-level rule only allows tickets for the user's own device and a phone number the user typed.
- **PII leaks into logs**: phones with a leading 0 or written `9876 543 210`, OTPs in refused messages,
  passwords without digits, and raw phone numbers in tool results.
- **Hinglish lock-bypass request** ("bina chabi ke kaise kholein, padosi ka hai") got past the guardrail.
- **Tool-call flooding**: one model reply with 25 tool calls ran all 25.
- **Ticket-id collision** after a restart silently overwrote an older customer's ticket (found by a SQL check).
- **API contract gaps**: a whitespace-only message was accepted, and reply/error schemas were missing from OpenAPI.

Red-team block rate on code-defended attacks: **25/35 (71%) → 35/35 (100%)**. Both runs are in
`reports/redteam/` and show on the dashboard trend.

## The suites

| Suite | What it checks | Tools | Docs |
|---|---|---|---|
| **Evals** | 64-case golden set: correctness, faithfulness, relevancy, hallucination rate, retrieval precision/recall, refusals, pass@k flakiness, prompt v1 vs v2 | DeepEval, Ragas, promptfoo, LLM judge | [docs/evals.md](docs/evals.md) |
| **Red team** | 46 attacks: direct and indirect injection, jailbreaks, system-prompt extraction, PII leakage, blocked topics, Hinglish attacks, abuse | custom detectors, promptfoo redteam | [docs/safety.md](docs/safety.md) |
| **Tool calling & conversations** | tool selection and args, multi-turn context, 503 retry and recovery, call caps | pytest OOP conversation framework | [docs/safety.md](docs/safety.md) |
| **API** | contract tests against OpenAPI, property-based fuzzing, error paths | pytest, httpx, Schemathesis, Postman/Newman | [docs/api-web-load.md](docs/api-web-load.md) |
| **Web UI** | chat flows, tool chips, refusals, mobile, keyboard, XSS | Playwright, Page Object Model | [docs/api-web-load.md](docs/api-web-load.md) |
| **Load** | p50/p95/p99 latency, RPS, error rate | Locust | [docs/api-web-load.md](docs/api-web-load.md) |
| **Data** | 18 SQL checks on chat logs (PII, integrity, cost vs tokens), drift monitoring (PSI, z-score) | SQLite, DuckDB, Athena SQL | [docs/data-aws.md](docs/data-aws.md) |
| **AWS** | Parquet export to S3, Athena queries, Lambda → CloudWatch quality metrics and alarms, least-privilege IAM, CloudFormation | boto3, moto, cfn-lint | [docs/data-aws.md](docs/data-aws.md) |
| **Mobile** | Android app pairing (Wi-Fi/BLE) and chat flows | Appium | *in progress* |

## Quality gate

`python -m qa.gate` reads every suite's latest report and compares it with `qa/thresholds.yaml` and with
the last accepted baseline. A pull request fails if, for example, the eval pass rate drops below 85%,
the hallucination rate goes over 10%, any critical attack gets through, or a metric gets worse than its
allowed delta. See [docs/dashboard-ci.md](docs/dashboard-ci.md).

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/
python -m playwright install chromium

# the bot (fake LLM by default, no key needed; LLM_MODE=live needs GEMINI_API_KEY in .env)
python -m uvicorn assistant.api:app --port 8000          # open http://localhost:8000

# all tests, offline
python -m pytest tests -q

# suites (replay = recorded real Gemini responses in cassettes/, no key needed)
python -m qa.evals.run --mode replay
python -m qa.redteam.run --mode replay
python scripts/run_all.py && python -m qa.gate
python scripts/build_dashboard.py      # -> site/index.html
```

LLM modes: `fake` (scripted, for unit tests), `replay` (recorded real responses, for CI),
`record` (live and save), `live`. Recorded cassettes make CI deterministic and free while
still testing real model behaviour.

## Layout

```
assistant/   the bot: llm.py (Gemini + cassettes), retrieval.py (BM25), devices.py, guardrails.py, bot.py, api.py
manuals/     product manuals (RAG source)
qa/          golden/, evals/, promptfoo/, redteam/, regressions/, data_checks/, aws/, gate.py
tests/       unit, safety, conversation, api, web, data, evals, gate
load/        Locust load test       infra/  IAM + CloudFormation       dashboard/ + scripts/  report site
reports/     every suite's JSON reports (history drives the dashboard trends)
```
