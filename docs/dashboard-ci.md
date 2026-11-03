# Quality gate, CI and the dashboard

This part of the project answers one question: **"Is this version of the bot good enough to ship?"**
— automatically, on every pull request, with numbers anyone can look at.

```
 code change ──► CI runs all test suites ──► each suite writes a JSON report
                                                     │
                       ┌─────────────────────────────┴───────────────┐
                       ▼                                             ▼
             quality gate (qa/gate.py)                  dashboard (scripts/build_dashboard.py)
         pass ► PR can merge / fail ► PR blocked         static web page on GitHub Pages
```

## 1. The reports (the shared language)

Every test suite (evals, red-team, tool calling, API, web, data, load) writes the same JSON shape
into `reports/<suite>/<timestamp>.json` and copies it to `reports/<suite>/latest.json`:

- `summary`: total / passed / failed / `pass_rate`, plus `metrics` such as `faithfulness`,
  `hallucination_rate`, `p95_latency_ms`, `cost_usd`.
- `cases`: one entry per test case with `passed`, `scores` and a `reason` when it failed.

Because everything speaks this one format, the gate and the dashboard do not need to know *how*
a suite works — only how to read its report.

## 2. The quality gate (`qa/gate.py` + `qa/thresholds.yaml`)

The gate is a small Python script. It reads each `latest.json` and checks it against rules in
`qa/thresholds.yaml`:

| Suite | Rule (current) | Why |
|---|---|---|
| evals | pass rate ≥ 85%, hallucination rate ≤ 10%, faithfulness ≥ 0.80 | answers must be correct and stick to the manual |
| redteam | **every case tagged `critical` must pass**, overall ≥ 95% | one leaked lock-bypass answer is a real-world harm |
| tool_calling | pass rate ≥ 90% | the bot must call the right device API with the right arguments |
| api, web, data | pass rate = 100% | these are normal deterministic tests — any failure is a bug |
| load | p95 latency ≤ 3000 ms | 95% of users get a reply within 3 seconds |

**Two kinds of check:**

1. **Hard thresholds** — the numbers above. Below the line = fail.
2. **Regression vs baseline** — `reports/baseline.json` stores the numbers of the last accepted
   version. A metric may not get worse than the baseline by more than a small tolerance
   (e.g. pass rate may drop at most 3 points). So even if evals are still at 88% (above 85%),
   a drop from 95% to 88% **blocks the PR**. This catches "the new prompt made the bot worse"
   before it reaches users.

**Other details worth knowing:**

- A suite with no report is shown as **skipped** (a warning), not as passed — the gate never
  pretends something was tested when it wasn't. `--require evals` (or `--require all`) turns a
  missing report into a failure.
- If a report is missing a metric that a rule needs, that check **fails** (better loud than silent).
- Red-team cases are "critical" when the case has `"tags": ["critical"]` or `"severity": "critical"`.
- Output is a markdown table. In GitHub Actions it is also written to `$GITHUB_STEP_SUMMARY`, so it
  shows up on the run's summary page.
- Exit code `0` = pass, `1` = fail. CI marks the job red on exit code 1, and branch protection
  ("require status checks to pass") turns that into a blocked merge.

```bash
python -m qa.gate                       # check and print the table
python -m qa.gate --require all         # every suite must have a report
python -m qa.gate --no-baseline         # thresholds only
python -m qa.gate --update-baseline     # accept the current numbers as the new baseline
```

Updating the baseline is a deliberate human step: after reviewing that a change really is fine,
you run `--update-baseline` and commit `reports/baseline.json`.

## 3. CI (`.github/workflows/`)

| Workflow | When | What it does |
|---|---|---|
| `ci.yml` | every push and pull request | install → `ruff` lint → `pytest` → `scripts/run_all.py` (all suites) → quality gate → upload reports + dashboard preview as an artifact |
| `pages.yml` | push to `main` | runs the suites, builds the dashboard, deploys it to GitHub Pages |
| `live-eval.yml` | manual button ("Run workflow") | runs the evals against the real Gemini API using the `GEMINI_API_KEY` secret |

**Why CI needs no API key:** the LLM runs in `LLM_MODE=replay`. Every LLM call was recorded once
into a "cassette" file; in CI the same request returns the same recorded answer. This makes CI
free, fast and deterministic (same code → same result), which a quality gate needs. The live
workflow is for checking the real model occasionally, without burning the free quota on every push.

`scripts/run_all.py` runs each suite that exists. It only fails when a suite *crashed* (ran but wrote
no report). Whether the bot is good enough is the gate's decision, not the runner's.
To avoid running them twice, the plain `pytest` step skips `tests/conversation`, `tests/api` and
`tests/web`; those run inside `run_all.py` where their reports are written.

## 4. The dashboard (`scripts/build_dashboard.py` + `dashboard/`)

A single static page — HTML, CSS and plain JavaScript, charts drawn as inline SVG. No framework,
no build step, no server.

- The script reads all reports, runs **the same gate code** (`qa.gate.evaluate`) so the page and CI
  can never disagree, and writes `site/index.html` (data embedded) + `site/data.json`.
- Sections: headline numbers, one card per suite with its gate status, trend charts across runs,
  prompt v1 vs v2 comparison, a filterable table of failing cases with reasons, and a plain-words
  "how this is tested" section.
- It only shows what the reports say. Missing suites show "Not run". Sample data (see below) shows
  a yellow "Sample data" banner.

```bash
python scripts/build_dashboard.py               # reads reports/, writes site/
python -m http.server -d site 8080              # open http://localhost:8080
```

### Sample data (for development only)

`python scripts/make_sample_reports.py` invents realistic reports into a temporary folder so the
dashboard can be designed before real results exist. Every sample report has `"sample": true`, and
the script refuses to write into the real `reports/` folder.

```bash
python scripts/make_sample_reports.py --failing           # prints the temp folder path
python scripts/build_dashboard.py --reports-dir <that folder> --out site
```

## 5. Interview cheat-sheet

- **"How do you stop a bad prompt from shipping?"** — Golden-set evals run on every PR; the gate
  fails on absolute thresholds *and* on regression vs the last accepted baseline.
- **"Why replay mode?"** — LLMs are non-deterministic and cost money; recorded responses make CI
  repeatable and free. A separate manual workflow runs live.
- **"Why are critical red-team cases special?"** — an average hides the one disaster. 97% blocked
  sounds good until the 3% is "how to bypass the lock". So critical cases must be 100%.
- **"What if a suite doesn't run?"** — it shows as skipped, never as passed; `--require` makes it fail.
- **Tests for the gate itself:** `tests/gate/` builds fake reports in temp folders and checks every
  rule, the baseline regression and the exit codes.
