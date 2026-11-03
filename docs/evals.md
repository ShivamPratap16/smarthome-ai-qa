# Answer-quality evals (LLM evaluation suite)

**Question this answers:** "Is the bot giving correct, grounded, safe answers - and did my last
change make it better or worse?" We run a fixed set of test questions (the *golden dataset*)
through the bot and score every answer. The result is `reports/evals/latest.json`, which the
quality gate (`qa/gate.py`) checks before a release.

## Files

| Path | What it is |
|---|---|
| `qa/golden/golden.jsonl` | 64 hand-written test cases from the manuals (+ `README.md`, `validate.py`) |
| `qa/evals/run.py` | the runner: plays cases, scores them, writes the report |
| `qa/evals/checks.py` | deterministic (no-LLM) checks |
| `qa/evals/judge.py` | LLM judge (one call per case) + DeepEval adapter for Gemini |
| `qa/evals/retrieval_metrics.py` | context precision / recall (Ragas ID-based metrics) |
| `qa/report.py` | `write_report()` - the shared report writer every suite uses |
| `qa/promptfoo/` | promptfoo config: prompt v1 vs v2 side by side |
| `qa/regressions/from_failure.py` | production chat failure -> new golden case |
| `tests/evals/` | offline tests for all of the above |

## Run it

```bash
python -m qa.golden.validate                                  # dataset is valid?
python -m qa.evals.run --mode fake                            # offline plumbing check (numbers are meaningless)
python -m qa.evals.run --mode replay --prompt v2              # CI: free, uses recorded cassettes
python -m qa.evals.run --mode record --compare v1 v2          # ONE live run that records everything
python -m qa.evals.run --mode replay --compare v1 v2          # A/B: reports/evals_compare/latest.json
python -m qa.evals.run --mode live --subset smoke --limit 3 --repeats 3 --deepeval off   # flakiness
python -m pytest tests/evals -q                               # unit tests, no API key
```

Options: `--subset all|smoke|<category>`, `--ids a,b`, `--limit N`, `--no-judge`,
`--deepeval on|off` (default on, smoke cases only), `--log-db data/eval_chatlogs.db` (keep the
eval chats in SQLite), `--reports-dir`.

Replay must use the same options as the record run (same cases, same prompt versions, same
DeepEval setting), because a cassette is found by the exact request it answers.

## How one case is scored

1. **Play** the user turns into a fresh `SupportAssistant` with a fresh `DeviceCloud`
   (plus an injected fault for recovery cases; a fixed "today" so warranty results never change).
2. **Deterministic checks** (free, exact):
   - right tool called with the right arguments; forbidden tools not called
   - `must_refuse` honoured, and no *over-refusal* of normal questions
   - `key_facts` present, `must_not_include` absent (e.g. a raw phone number)
   - unanswerable: the bot must say it doesn't know, and must not contain numbers that appear in
     no source (manual chunks, tool results, user messages) - invented numbers are the most common
     hallucination
   - Hinglish question -> Hinglish reply
3. **Retrieval metrics** (free): compare retrieved chunk ids with `expected_context_ids`.
4. **LLM judge** (1 call) only for `manual_qa`, `tool_use`, `multi_turn`, `hinglish`.
   Refusal, recovery and unanswerable are decided by rules - no quota spent.
5. **DeepEval** `FaithfulnessMetric` + `GEval` correctness on the smoke cases only.

A case passes only if every check that applies to it passes (judge thresholds:
faithfulness >= 0.7, correctness >= 0.6, relevancy >= 0.5).

## The metrics in simple words

- **Faithfulness / groundedness** - of the facts in the answer, how many are backed by the text
  the bot was given (manual chunks + tool results)? 1.0 = everything is backed. Low = the model
  made things up, even if they sound right.
- **Answer relevancy** - does the answer actually address the question asked?
- **Correctness** - does the answer agree with our human reference answer on the key facts?
  (Faithful but incomplete answers score high on faithfulness and lower on correctness.)
- **Hallucination rate** - out of the answered (not refused) cases we could check, the share where
  the bot stated something unsupported: judge said hallucination / faithfulness < 0.7, or on an
  unanswerable question it invented an answer instead of saying "I don't know". Gate: <= 10%.
- **Context recall** - of the manual sections that were needed, how many did retrieval find?
  Low recall = the right answer was never in front of the model (a search problem, not an LLM problem).
- **Context precision** - are the needed sections ranked at the top? We use average precision:
  needed chunk at rank 1 -> 1.0, at rank 2 -> 0.5. `context_precision_raw` is the plain share of
  retrieved chunks that were needed (max 0.25 when 4 are retrieved and 1 is needed - that is normal).
- **Refusal accuracy** - unsafe requests (DIY wiring, bypassing a lock, medical water claims)
  that were refused. **Over-refusal rate** - normal questions wrongly refused.
- **Tool accuracy** - cases where the expected tool calls happened and no forbidden ones.
- **pass_rate_<category>** - pass rate per category, so a drop in e.g. Hinglish is visible even
  if the overall number looks fine.
- **p95 latency** - 95% of cases were answered faster than this. **cost_usd** - token cost of the run.

## Non-determinism: pass@k and flakiness

An LLM can answer the same question differently each time, so "it passed once" is weak evidence.
`--repeats k` runs every case k times (use `live`; replay always returns the same recording):
- **pass@k** - share of cases that passed at least once in k runs (can it do it?)
- **pass^k / pass_all_k** - share that passed every time (can we rely on it?). A case counts as
  passed in the report only if it passed all k runs.
- **flaky_rate** - share that passed sometimes and failed sometimes; each case gets a
  `consistency` score (passes / k). Flaky cases are where a prompt needs tightening.
We also use temperature 0 and record/replay so CI is deterministic; repeats are how we measure
the randomness that remains in production.

## Why these tools (and the quota trade-off)

- **Our own rubric judge** (G-Eval style) returns faithfulness, relevancy, correctness and a
  hallucination flag in ONE call per case. DeepEval's metrics need 3-4 calls per case each, which
  the free Gemini tier cannot afford for 60+ cases.
- **DeepEval** (`FaithfulnessMetric`, `GEval`) runs on the 16 smoke cases with a small adapter
  (`make_deepeval_model`) that sends its prompts through our Gemini client - so it is cached by the
  same cassettes. It cross-checks our judge with an industry-standard implementation.
- **Ragas** `IDBasedContextRecall` / `IDBasedContextPrecision` need no LLM at all - they compare
  chunk ids. (Ragas 0.4 needs `langchain-community<0.4`, pinned in requirements.txt.)
- **promptfoo** gives the side-by-side v1 vs v2 web view that non-engineers can read.
- The judge is the same `get_llm()` client as the bot, so judge answers are recorded once and
  replayed for free; in `fake` mode a word-overlap judge keeps CI fully offline.
- LangSmith would add hosted tracing/datasets; we keep everything local and file-based instead
  (reports JSON + SQLite chat logs), which is free and reviewable in git.

## promptfoo (prompt v1 vs v2)

```bash
python -m qa.promptfoo.build_tests            # golden.jsonl -> qa/promptfoo/tests.yaml (--smoke for 16)
cd qa/promptfoo
PROMPTFOO_PYTHON=../../.venv/Scripts/python.exe LLM_MODE=replay npx promptfoo@latest eval --no-cache
npx promptfoo@latest view                     # browser table: v1 and v2 answers side by side
```
`provider.py` plays each case into the bot (v1 or v2 from the provider config) and appends two
footer lines (`[tools] [...]`, `[refused] true|false`). Assertions: `icontains` key facts,
`not-icontains` banned phrases, a `python` assertion for tool calls (`assert_tools.py`) and
`llm-rubric` for refusal / unanswerable cases. The rubric grader is `provider.py:call_grader`,
which uses our cached Gemini client (keyword rule in fake mode). Bot calls hit the same
cassettes as `qa.evals.run`, so after the evals record run promptfoo replays for free.

## Production failure -> regression test

```bash
python -m qa.regressions.from_failure --list                  # failed chats in data/chatlogs.db
python -m qa.regressions.from_failure --latest-failed         # newest one -> golden.jsonl
python -m qa.regressions.from_failure --conversation <id> --dry-run
python -m qa.regressions.from_failure --report reports/evals/latest.json --case <id> --db data/eval_chatlogs.db
```
A chat counts as failed if the model refused without a guardrail block (over-refusal), a tool
returned an error, or the bot said it doesn't know. The new case has the real (PII-masked) user
turns, a guessed category/product, `needs_review: true` and a `source` block pointing to the
conversation. A human writes the correct answer and removes the flag; from then on the bug can
never come back silently. Same conversation twice -> not added twice.

## Interview talking points
- Deterministic checks first, LLM judge only where rules can't decide -> cheaper and more stable.
- Separate retrieval metrics from answer metrics: tells you *where* a wrong answer came from.
- Hallucination rate is tracked per run and gated, so a prompt change that makes the bot
  "more helpful" but less grounded is blocked before release.
- Unanswerable questions with tempting distractors (the lock's 90 dB alarm vs. "how loud is the
  fan?") are the best hallucination traps.
- "Test the tests": `tests/evals` feeds scripted perfect / hallucinating / over-refusing bots
  through the runner and checks that the scores react correctly.
