"""Run the golden dataset through the assistant and score every answer.

    python -m qa.evals.run --mode fake                       # offline plumbing check (CI)
    python -m qa.evals.run --mode replay --prompt v2         # free, uses recorded cassettes
    python -m qa.evals.run --mode record --prompt v2         # live Gemini + save cassettes
    python -m qa.evals.run --mode live --subset smoke --limit 3 --repeats 3   # flakiness check
    python -m qa.evals.run --mode replay --compare v1 v2     # prompt A/B comparison

Per case we:
  1. play the user turns into a fresh SupportAssistant (fresh DeviceCloud, optional fault)
  2. deterministic checks: tools/args, forbidden tools, refusal, key facts, banned phrases,
     abstain-on-unanswerable, invented numbers, Hinglish reply language
  3. retrieval metrics (no LLM): context precision / recall from expected_context_ids (Ragas)
  4. ONE rubric-judge call (faithfulness, relevancy, correctness, hallucination) for the
     categories a rule cannot decide (manual_qa, tool_use, multi_turn, hinglish)
  5. optional DeepEval FaithfulnessMetric + GEval correctness on the `smoke` subset
Then writes reports/evals/<timestamp>.json + latest.json (qa.report.write_report).
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from assistant.bot import BotReply, SupportAssistant
from assistant.devices import DeviceCloud
from assistant.llm import DEFAULT_MODEL, CassetteMiss, LLMClient, LLMError, LLMResponse, get_llm
from assistant.store import ChatStore
from qa.evals import checks
from qa.evals.judge import RubricJudge, fake_judge_llm, make_deepeval_model
from qa.evals.retrieval_metrics import ragas_available, retrieval_scores
from qa.golden.validate import GOLDEN_PATH, load_golden
from qa.report import write_report

# Warranty answers depend on "today"; a fixed date keeps tool results (and so cassettes) stable.
EVAL_TODAY = date(2026, 11, 1)
JUDGED_CATEGORIES = {"manual_qa", "tool_use", "multi_turn", "hinglish"}
PASS_FAITHFULNESS = 0.7
PASS_CORRECTNESS = 0.6
PASS_RELEVANCY = 0.5
STATE_CHANGING_TOOLS = ["set_fan_speed", "create_service_ticket"]


class CountingLLM:
    """Wraps an LLMClient and counts calls (to report how much quota a run needs)."""

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.calls = 0

    def generate(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        """Forward to the wrapped client."""
        self.calls += 1
        return self.inner.generate(system, messages, tools, temperature)


@dataclass
class Conversation:
    """Everything the bot produced while playing one golden case."""

    replies: list[BotReply]
    session_id: str
    error: str | None = None

    @property
    def final(self) -> BotReply | None:
        return self.replies[-1] if self.replies else None

    @property
    def tool_calls(self) -> list[tuple[str, dict]]:
        return [(t.name, t.args) for r in self.replies for t in r.tool_calls]

    @property
    def tool_results_text(self) -> list[str]:
        out = []
        for r in self.replies:
            for t in r.tool_calls:
                out.append(f"tool {t.name}({t.args}) -> " + (str(t.result) if t.error is None
                                                              else f"error {t.error}"))
        return out


@dataclass
class Settings:
    """Options for one eval run."""

    mode: str = "fake"
    prompt_version: str = "v2"
    use_judge: bool = True
    use_deepeval: bool = False
    deepeval_scope: str = "smoke"          # "smoke" or "all"
    repeats: int = 1
    max_llm_errors: int = 3
    log_db: str = ":memory:"
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- running a case
def play_case(case: dict, llm: LLMClient, prompt_version: str, store: ChatStore,
              repeat: int = 0) -> Conversation:
    """Send every user turn of the case to a fresh assistant."""
    devices = DeviceCloud(fail_next=case.get("fault"), today=EVAL_TODAY)
    assistant = SupportAssistant(llm=llm, devices=devices, prompt_version=prompt_version, store=store)
    session_id = f"eval-{case['id']}-{prompt_version}-r{repeat}"
    convo = Conversation(replies=[], session_id=session_id)
    for turn in case["turns"]:
        try:
            convo.replies.append(assistant.chat(session_id, turn))
        except CassetteMiss as exc:
            convo.error = f"CassetteMiss (re-record needed): {exc.key[:12]}"
            break
        except LLMError as exc:
            convo.error = f"LLMError {exc.status}: {exc.msg[:120]}"
            break
    return convo


def judge_question(case: dict, convo: Conversation) -> str:
    """The conversation as text for the judge (earlier bot replies help resolve 'it', 'that')."""
    lines = []
    for i, turn in enumerate(case["turns"]):
        lines.append(f"customer: {turn}")
        if i < len(convo.replies) - 1:
            lines.append(f"bot: {convo.replies[i].text}")
    return "\n".join(lines)


def score_case(case: dict, convo: Conversation, judge: RubricJudge | None,
               deepeval_model=None) -> dict:
    """Run all checks for one played case. Returns a Contract-5 case dict."""
    result = {"id": case["id"], "category": case["category"], "product": case["product"],
              "tags": case.get("tags", []), "session_id": convo.session_id, "passed": False,
              "scores": {}, "reason": "", "latency_ms": 0}
    final = convo.final
    if convo.error or final is None:
        result["reason"] = convo.error or "no reply"
        result["error"] = True
        return result

    answer = final.text
    result.update(answer=answer[:600], refused=final.refused, blocked_by=final.blocked_by,
                  tool_calls=[n for n, _ in convo.tool_calls], context_ids=final.context_ids,
                  latency_ms=sum(r.latency_ms for r in convo.replies),
                  cost_usd=round(sum(r.cost_usd for r in convo.replies), 8))
    scores: dict = {}
    problems: list[str] = []
    category = case["category"]

    # 1. refusal behaviour
    if case["must_refuse"]:
        scores["refusal_ok"] = 1.0 if final.refused else 0.0
        if not final.refused:
            problems.append("should have refused but answered")
    elif final.refused:
        scores["refusal_ok"] = 0.0
        problems.append(f"refused a legitimate request (blocked_by={final.blocked_by})")

    # 2. tools
    forbidden = list(case.get("forbidden_tools", []))
    if category in ("refusal", "unanswerable"):
        forbidden += STATE_CHANGING_TOOLS
    tools_ok, tool_problem = checks.check_tools(case["expected_tools"], forbidden, convo.tool_calls)
    if case["expected_tools"] or forbidden:
        scores["tool_match"] = 1.0 if tools_ok else 0.0
    if not tools_ok:
        problems.append(tool_problem)

    # 3. key facts and banned phrases
    if case.get("key_facts"):
        fact_score, missing = checks.check_key_facts(answer, case["key_facts"])
        scores["key_facts"] = round(fact_score, 4)
        if missing:
            problems.append(f"missing key facts {missing}")
    banned = checks.check_must_not_include(answer, case.get("must_not_include", []))
    if banned:
        problems.append(f"contains banned phrase(s) {banned}")

    # 4. invented numbers (cheap hallucination signal) + unanswerable handling
    sources = final.contexts + case["turns"] + convo.tool_results_text + [r.text for r in convo.replies[:-1]]
    invented = checks.unsupported_numbers(answer, sources) if not final.refused else []
    scores["invented_numbers"] = len(invented)
    hallucinated: bool | None = None
    if category == "unanswerable":
        said_unknown = checks.abstained(answer)
        scores["abstained"] = 1.0 if said_unknown else 0.0
        hallucinated = (not said_unknown) or bool(invented)
        if not said_unknown:
            problems.append("did not say it doesn't know (possible invented answer)")
        if invented:
            problems.append(f"numbers not in any source: {invented}")

    # 5. language
    if category == "hinglish" and not final.refused:
        scores["language_match"] = 1.0 if checks.is_hinglish(answer) else 0.0
        if not scores["language_match"]:
            problems.append("replied in English to a Hinglish question")

    # 6. retrieval (only when retrieval ran, i.e. not blocked by a guardrail)
    if final.blocked_by is None:
        scores.update(retrieval_scores(final.context_ids, case["expected_context_ids"]))

    # 7. LLM judge (one call) for open-ended answers
    if judge is not None and category in JUDGED_CATEGORIES and not final.refused:
        context = final.contexts + convo.tool_results_text
        try:
            j = judge.score(judge_question(case, convo), answer, context, case["expected_answer"])
            scores.update(faithfulness=j.faithfulness, answer_relevancy=j.answer_relevancy,
                          correctness=j.correctness)
            hallucinated = j.hallucination or j.faithfulness < PASS_FAITHFULNESS
            result["judge_reason"] = j.reason
            if j.unsupported_claims:
                result["unsupported_claims"] = j.unsupported_claims
            if j.faithfulness < PASS_FAITHFULNESS:
                problems.append(f"low faithfulness {j.faithfulness:.2f}: {j.reason}")
            if j.correctness < PASS_CORRECTNESS:
                problems.append(f"low correctness {j.correctness:.2f}: {j.reason}")
            if j.answer_relevancy < PASS_RELEVANCY:
                problems.append(f"low relevancy {j.answer_relevancy:.2f}")
        except (LLMError, ValueError) as exc:
            result["judge_error"] = str(exc)[:200]
            problems.append(f"judge failed: {str(exc)[:100]}")

    # 8. DeepEval metrics (opt-in, smoke subset by default)
    if deepeval_model is not None and category in JUDGED_CATEGORIES and not final.refused:
        scores.update(run_deepeval(case, convo, deepeval_model))

    if hallucinated is not None:
        scores["hallucinated"] = 1.0 if hallucinated else 0.0
    result["scores"] = scores
    result["passed"] = not problems
    result["reason"] = "; ".join(problems)
    return result


def run_deepeval(case: dict, convo: Conversation, model) -> dict:
    """DeepEval FaithfulnessMetric + GEval correctness for one case (several judge calls)."""
    from deepeval.metrics import FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase
    try:
        from deepeval.test_case import SingleTurnParams as Params   # deepeval >= 4
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams as Params

    final = convo.final
    test_case = LLMTestCase(input=judge_question(case, convo), actual_output=final.text,
                            expected_output=case["expected_answer"],
                            retrieval_context=final.contexts + convo.tool_results_text or ["(none)"])
    out: dict = {}
    metrics = {
        "deepeval_faithfulness": FaithfulnessMetric(model=model, include_reason=False, async_mode=False),
        "deepeval_correctness": GEval(
            name="Correctness", model=model, async_mode=False,
            evaluation_params=[Params.INPUT, Params.ACTUAL_OUTPUT, Params.EXPECTED_OUTPUT],
            evaluation_steps=["Check whether the facts (numbers, steps, yes/no) in 'actual output' "
                              "agree with 'expected output'.",
                              "Penalise contradictions and invented facts heavily; extra correct "
                              "details and a Hinglish reply are fine."]),
    }
    for name, metric in metrics.items():
        try:
            metric.measure(test_case)
            out[name] = round(float(metric.score), 4)
        except Exception as exc:  # noqa: BLE001 - one broken metric must not kill the run
            out[f"{name}_error"] = str(exc)[:120]
    return out


# --------------------------------------------------------------------------- aggregation
def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def p95(values: list[int]) -> int:
    """95th percentile (nearest-rank)."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def merge_repeats(runs: list[dict]) -> dict:
    """Combine k runs of the same case: passed only if ALL passed; numeric scores averaged."""
    first = dict(runs[0])
    k = len(runs)
    passes = sum(1 for r in runs if r["passed"])
    scores: dict = {}
    for key in {key for r in runs for key in r["scores"]}:
        vals = [r["scores"][key] for r in runs if isinstance(r["scores"].get(key), (int, float))]
        if vals:
            scores[key] = round(statistics.fmean(vals), 4)
    scores.update(pass_count=passes, repeats=k, consistency=round(passes / k, 4))
    first["scores"] = scores
    first["passed"] = passes == k
    first["flaky"] = 0 < passes < k
    failing = [r["reason"] for r in runs if not r["passed"]]
    first["reason"] = (f"passed {passes}/{k} runs; " + failing[0]) if failing else ""
    first["latency_ms"] = int(statistics.fmean(r.get("latency_ms", 0) for r in runs))
    return first


def summarize(cases: list[dict], repeats: int = 1) -> dict:
    """Suite summary + metrics (per-category pass rates, judge means, hallucination rate ...)."""
    metrics: dict = {}

    def collect(key: str) -> list[float]:
        return [c["scores"][key] for c in cases if isinstance(c["scores"].get(key), (int, float))]

    for key in ("faithfulness", "answer_relevancy", "correctness", "context_precision",
                "context_recall", "context_precision_raw", "key_facts", "deepeval_faithfulness",
                "deepeval_correctness"):
        value = _mean(collect(key))
        if value is not None:
            metrics[key] = value
    halluc = collect("hallucinated")
    metrics["hallucination_rate"] = _mean(halluc) if halluc else 0.0
    metrics["hallucination_checked"] = len(halluc)

    refusal_cases = [c for c in cases if c["category"] == "refusal" and not c.get("error")]
    if refusal_cases:
        metrics["refusal_accuracy"] = round(sum(1 for c in refusal_cases if c.get("refused"))
                                            / len(refusal_cases), 4)
    should_answer = [c for c in cases if c["category"] != "refusal" and not c.get("error")]
    if should_answer:
        metrics["over_refusal_rate"] = round(sum(1 for c in should_answer if c.get("refused"))
                                             / len(should_answer), 4)
    tool_scores = collect("tool_match")
    if tool_scores:
        metrics["tool_accuracy"] = _mean(tool_scores)

    for cat in sorted({c["category"] for c in cases}):
        in_cat = [c for c in cases if c["category"] == cat]
        metrics[f"pass_rate_{cat}"] = round(sum(c["passed"] for c in in_cat) / len(in_cat), 4)

    latencies = [c.get("latency_ms", 0) for c in cases if not c.get("error")]
    metrics["p95_latency_ms"] = p95(latencies)
    total_cost = sum(c.get("cost_usd", 0.0) for c in cases)
    metrics["cost_usd"] = round(total_cost, 6)
    metrics["cost_per_100_chats"] = round(total_cost / len(cases) * 100, 6) if cases else 0.0
    metrics["errors"] = sum(1 for c in cases if c.get("error"))

    if repeats > 1:
        metrics["repeats"] = repeats
        metrics["pass_at_k"] = round(sum(1 for c in cases if c["scores"].get("pass_count", 0) > 0)
                                     / len(cases), 4) if cases else 0.0
        metrics["pass_all_k"] = round(sum(1 for c in cases if c["passed"]) / len(cases), 4) if cases else 0.0
        metrics["flaky_rate"] = round(sum(1 for c in cases if c.get("flaky")) / len(cases), 4) if cases else 0.0

    passed = sum(1 for c in cases if c["passed"])
    return {"total": len(cases), "passed": passed, "failed": len(cases) - passed,
            "pass_rate": round(passed / len(cases), 4) if cases else 0.0, "metrics": metrics}


# --------------------------------------------------------------------------- whole suite
def select_cases(cases: list[dict], subset: str = "all", ids: list[str] | None = None,
                 limit: int | None = None, include_review: bool = False) -> list[dict]:
    """Filter by subset ("all", "smoke", or a category name), ids and limit.

    Cases with `needs_review: true` (new regressions without a reference answer yet) are skipped
    unless include_review=True.
    """
    if not include_review:
        cases = [c for c in cases if not c.get("needs_review")]
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    elif subset == "smoke":
        cases = [c for c in cases if "smoke" in c.get("tags", [])]
    elif subset != "all":
        cases = [c for c in cases if c["category"] == subset]
    return cases[:limit] if limit else cases


def build_clients(mode: str) -> tuple[LLMClient, LLMClient]:
    """(bot llm, judge llm) for a mode. Fake mode uses a rule-based bot and a heuristic judge."""
    bot_llm = get_llm(mode)
    judge_llm = fake_judge_llm() if mode == "fake" else bot_llm   # same cached client
    return bot_llm, judge_llm


def run_suite(cases: list[dict], settings: Settings, bot_llm: LLMClient | None = None,
              judge_llm: LLMClient | None = None) -> tuple[dict, list[dict], dict]:
    """Play + score every case. Returns (summary, case results, run info)."""
    if bot_llm is None or judge_llm is None:
        default_bot, default_judge = build_clients(settings.mode)
        bot_llm, judge_llm = bot_llm or default_bot, judge_llm or default_judge
    bot_llm, judge_llm = CountingLLM(bot_llm), CountingLLM(judge_llm)   # count quota use
    judge = RubricJudge(judge_llm) if settings.use_judge else None
    deepeval_model = None
    if settings.use_deepeval:
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
        deepeval_model = make_deepeval_model(judge_llm, os.getenv("GEMINI_MODEL") or DEFAULT_MODEL)
    store = ChatStore(settings.log_db)

    results: list[dict] = []
    consecutive_errors = 0
    aborted = False
    for case in cases:
        runs = []
        for r in range(settings.repeats):
            convo = play_case(case, bot_llm, settings.prompt_version, store, repeat=r)
            in_scope = settings.deepeval_scope == "all" or "smoke" in case.get("tags", [])
            runs.append(score_case(case, convo, judge, deepeval_model if in_scope else None))
        merged = merge_repeats(runs) if settings.repeats > 1 else runs[0]
        results.append(merged)
        is_llm_error = merged.get("error") and "LLMError" in merged.get("reason", "")
        consecutive_errors = consecutive_errors + 1 if is_llm_error else 0
        status = "PASS" if merged["passed"] else "FAIL"
        print(f"[{status}] {case['id']:<24} {merged['reason'][:110]}", flush=True)
        if consecutive_errors >= settings.max_llm_errors:
            print(f"Stopping: {consecutive_errors} LLM errors in a row (quota?).", file=sys.stderr)
            aborted = True
            break

    summary = summarize(results, settings.repeats)
    info = {"bot_llm_calls": bot_llm.calls,
            "judge_llm_calls": judge_llm.calls,
            "deepeval_calls": getattr(deepeval_model, "calls", 0),
            "retrieval_backend": "ragas" if ragas_available() else "builtin",
            "judge": "fake-heuristic" if settings.mode == "fake" else "gemini-rubric",
            "incomplete": aborted, "cases_planned": len(cases)}
    summary["metrics"]["llm_calls"] = info["bot_llm_calls"] + info["judge_llm_calls"]
    return summary, results, info


def model_name(mode: str) -> str:
    """Model name stored in the report."""
    return "fake" if mode == "fake" else (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL)


def run_and_report(cases: list[dict], settings: Settings, reports_dir: str | None = None) -> tuple[Path, dict]:
    """Run the suite for one prompt version and write reports/evals/..."""
    summary, results, info = run_suite(cases, settings)
    path = write_report("evals", summary, results, prompt_version=settings.prompt_version,
                        model=model_name(settings.mode), llm_mode=settings.mode,
                        reports_dir=reports_dir, golden_file=str(GOLDEN_PATH.name), **info,
                        **settings.extra)
    return path, {"summary": summary, "cases": results, "info": info}


def compare_runs(runs: dict[str, dict]) -> tuple[dict, list[dict]]:
    """Build the comparison summary + per-case rows (which cases got fixed / broke)."""
    versions = list(runs)
    base, new = versions[0], versions[-1]
    metric_names = sorted({k for r in runs.values() for k in r["summary"]["metrics"]} | {"pass_rate"})
    table = {}
    for m in metric_names:
        values = {v: (runs[v]["summary"]["pass_rate"] if m == "pass_rate"
                      else runs[v]["summary"]["metrics"].get(m)) for v in versions}
        row = dict(values)
        if isinstance(values[base], (int, float)) and isinstance(values[new], (int, float)):
            row["delta"] = round(values[new] - values[base], 4)
        table[m] = row
    by_id = {v: {c["id"]: c for c in runs[v]["cases"]} for v in versions}
    rows = []
    for cid, case in by_id[new].items():
        old = by_id[base].get(cid)
        old_pass = bool(old and old["passed"])
        change = ("fixed" if case["passed"] and not old_pass else
                  "regressed" if old_pass and not case["passed"] else "same")
        rows.append({"id": cid, "category": case["category"], "passed": case["passed"],
                     "scores": {f"{base}_passed": int(old_pass), f"{new}_passed": int(case["passed"])},
                     "change": change, "reason": case["reason"] if change == "regressed" else ""})
    passed = sum(1 for r in rows if r["passed"])
    summary = {"total": len(rows), "passed": passed, "failed": len(rows) - passed,
               "metrics": {"fixed": sum(r["change"] == "fixed" for r in rows),
                           "regressed": sum(r["change"] == "regressed" for r in rows)},
               "versions": versions, "table": table}
    return summary, rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Hawa Assist answer-quality evals")
    p.add_argument("--mode", default=os.getenv("LLM_MODE", "fake"),
                   choices=["fake", "replay", "record", "live"])
    p.add_argument("--prompt", default="v2", help="prompt version (v1 or v2)")
    p.add_argument("--subset", default="all", help="all | smoke | <category>")
    p.add_argument("--ids", default="", help="comma-separated case ids")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--repeats", type=int, default=1, help="run each case N times (pass@k, flakiness)")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two prompt versions")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM rubric judge")
    p.add_argument("--deepeval", choices=["on", "off"], default="on",
                   help="DeepEval FaithfulnessMetric + GEval (4 judge calls per case, so smoke only by "
                        "default; in --compare only for the last prompt version)")
    p.add_argument("--deepeval-scope", choices=["smoke", "all"], default="smoke")
    p.add_argument("--log-db", default=":memory:", help="SQLite chat log for the eval conversations")
    p.add_argument("--reports-dir", default=None)
    p.add_argument("--golden", default=str(GOLDEN_PATH))
    p.add_argument("--max-llm-errors", type=int, default=3)
    p.add_argument("--include-review", action="store_true", help="also run needs_review cases")
    args = p.parse_args(argv)

    os.environ["LLM_MODE"] = args.mode
    cases = select_cases(load_golden(Path(args.golden)), args.subset,
                         [i.strip() for i in args.ids.split(",") if i.strip()], args.limit,
                         args.include_review)
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2
    use_deepeval = args.deepeval == "on"
    if use_deepeval:
        try:
            import deepeval  # noqa: F401
        except ImportError:
            print("deepeval not installed - skipping DeepEval metrics", file=sys.stderr)
            use_deepeval = False

    def settings_for(version: str, deepeval_on: bool = True) -> Settings:
        return Settings(mode=args.mode, prompt_version=version, use_judge=not args.no_judge,
                        use_deepeval=use_deepeval and deepeval_on, deepeval_scope=args.deepeval_scope,
                        repeats=args.repeats, max_llm_errors=args.max_llm_errors, log_db=args.log_db,
                        extra={"subset": args.subset if not args.ids else "ids"})

    if args.compare:
        runs = {}
        for version in args.compare:
            print(f"=== prompt {version} ===")
            last = version == args.compare[-1]
            path, runs[version] = run_and_report(cases, settings_for(version, deepeval_on=last),
                                                 args.reports_dir)
            print(f"report: {path}")
        summary, rows = compare_runs(runs)
        path = write_report("evals_compare", summary, rows, prompt_version="-vs-".join(args.compare),
                            model=model_name(args.mode), llm_mode=args.mode, reports_dir=args.reports_dir)
        print_compare(summary)
        print(f"comparison: {path}")
        return 0

    path, run = run_and_report(cases, settings_for(args.prompt), args.reports_dir)
    print_summary(run["summary"], run["info"])
    print(f"report: {path}")
    return 0


def print_summary(summary: dict, info: dict) -> None:
    """Short console summary."""
    print(f"\n{summary['passed']}/{summary['total']} passed (pass_rate {summary['pass_rate']:.2%})")
    for key, value in sorted(summary["metrics"].items()):
        print(f"  {key:<26} {value}")
    print(f"  LLM calls: bot={info['bot_llm_calls']} judge(incl. deepeval)={info['judge_llm_calls']} "
          f"deepeval={info['deepeval_calls']}  retrieval={info['retrieval_backend']}")


def print_compare(summary: dict) -> None:
    """Console table for --compare."""
    a, b = summary["versions"][0], summary["versions"][-1]
    print(f"\n{'metric':<26}{a:>10}{b:>10}{'delta':>10}")
    for metric, row in summary["table"].items():
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"{metric:<26}{fmt(row.get(a)):>10}{fmt(row.get(b)):>10}{fmt(row.get('delta', '')):>10}")
    print(f"fixed by {b}: {summary['metrics']['fixed']}, regressed: {summary['metrics']['regressed']}")


if __name__ == "__main__":
    sys.exit(main())
