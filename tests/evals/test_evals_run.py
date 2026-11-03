"""The eval runner end-to-end in fake mode, plus "test the tests" cases with scripted LLMs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant import bot as bot_module
from assistant.llm import FakeLLM, LLMResponse, ToolCall
from qa.evals import run
from qa.golden.validate import load_golden

CASES = {c["id"]: c for c in load_golden()}
PERFECT = json.dumps({"faithfulness": 1, "answer_relevancy": 1, "correctness": 1,
                      "hallucination": False, "reason": "ok"})
SETTINGS = run.Settings(mode="fake", use_deepeval=False)


def run_one(case_id: str, bot_script: list, judge_script: list | None = None) -> dict:
    _, results, _ = run.run_suite([CASES[case_id]], SETTINGS, bot_llm=FakeLLM(bot_script),
                                  judge_llm=FakeLLM(judge_script or [PERFECT]))
    return results[0]


def test_fake_mode_end_to_end_writes_report(tmp_path: Path):
    code = run.main(["--mode", "fake", "--subset", "smoke", "--deepeval", "off",
                     "--reports-dir", str(tmp_path)])
    assert code == 0
    report = json.loads((tmp_path / "evals" / "latest.json").read_text(encoding="utf-8"))
    smoke = [c for c in CASES.values() if "smoke" in c.get("tags", [])]
    assert report["suite"] == "evals" and report["llm_mode"] == "fake"
    assert report["summary"]["total"] == len(smoke)
    metrics = report["summary"]["metrics"]
    for key in ("faithfulness", "answer_relevancy", "hallucination_rate", "context_precision",
                "context_recall", "p95_latency_ms", "cost_usd", "pass_rate_refusal", "pass_rate_hinglish"):
        assert key in metrics, key
    assert all({"id", "category", "passed", "scores", "reason"} <= set(c) for c in report["cases"])
    blocked = next(c for c in report["cases"] if c["id"] == "ref-wiring-01")
    assert blocked["passed"] and blocked["blocked_by"] == "blocked_topic"


def test_deepeval_metrics_run_offline_with_fake_judge(tmp_path: Path):
    pytest.importorskip("deepeval")
    run.main(["--mode", "fake", "--ids", "aero-boost-05", "--reports-dir", str(tmp_path)])
    case = json.loads((tmp_path / "evals" / "latest.json").read_text(encoding="utf-8"))["cases"][0]
    assert "deepeval_faithfulness" in case["scores"] and "deepeval_correctness" in case["scores"]


def test_correct_tool_answer_passes():
    result = run_one("tool-speed-01", [
        LLMResponse("", [ToolCall("set_fan_speed", {"device_id": "fan-living", "speed": 3})]),
        "Done - the living room fan is now on speed 3."])
    assert result["passed"], result["reason"]
    assert result["scores"]["tool_match"] == 1.0 and result["scores"]["hallucinated"] == 0.0


def test_wrong_tool_args_fail():
    result = run_one("tool-speed-01", [
        LLMResponse("", [ToolCall("set_fan_speed", {"device_id": "fan-bedroom", "speed": 3})]),
        "Done."])
    assert not result["passed"] and "missing tool call" in result["reason"]


def test_invented_answer_to_unanswerable_question_is_a_hallucination():
    result = run_one("un-noise-04", ["The Aero is very quiet: only 42 dB at speed 5."])
    assert not result["passed"]
    assert result["scores"]["hallucinated"] == 1.0 and result["scores"]["abstained"] == 0.0


def test_honest_i_dont_know_passes_unanswerable():
    result = run_one("un-noise-04", ["I don't know - the manual doesn't list a noise level. "
                                     "Shall I raise a service ticket?"])
    assert result["passed"], result["reason"]


def test_double_503_recovery_passes_without_judge():
    result = run_one("rec-503-twice-02", [
        LLMResponse("", [ToolCall("set_fan_speed", {"device_id": "fan-living", "speed": 5})])],
        judge_script=[])
    assert result["answer"] == bot_module.DEVICE_DOWN_MESSAGE
    assert result["passed"], result["reason"]
    assert "faithfulness" not in result["scores"]          # deterministic category: no judge call


def test_low_judge_score_fails_case():
    bad = json.dumps({"faithfulness": 0.2, "answer_relevancy": 0.9, "correctness": 0.1,
                      "hallucination": True, "reason": "says 400 RPM"})
    result = run_one("aero-boost-05", ["Boost runs at 380 RPM for 2 hours."], [bad])
    assert not result["passed"] and "low faithfulness" in result["reason"]


def test_model_refusal_of_unsafe_request_passes():
    result = run_one("ref-sealed-05", ["Sorry, I can't help with that. Please raise a service ticket."])
    assert result["passed"] and result["refused"]


def test_over_refusal_fails():
    result = run_one("aero-boost-05", ["Sorry, I can't help with that."])
    assert not result["passed"] and "refused a legitimate request" in result["reason"]


def test_english_reply_to_hinglish_question_fails_language_check():
    result = run_one("hi-filter-04", ["Replace the sediment filter every 6 months or 3,000 litres."])
    assert result["scores"]["language_match"] == 0.0 and not result["passed"]


def test_cassette_miss_is_reported_not_crashing():
    from assistant.llm import CassetteMiss
    result = run_one("aero-boost-05", [CassetteMiss("abc123")])
    assert not result["passed"] and result.get("error") and "CassetteMiss" in result["reason"]


def test_merge_repeats_detects_flaky_cases():
    runs = [{"id": "a", "category": "manual_qa", "passed": p, "scores": {"faithfulness": f},
             "reason": "" if p else "low", "latency_ms": 10} for p, f in ((True, 1.0), (False, 0.5), (True, 1.0))]
    merged = run.merge_repeats(runs)
    assert not merged["passed"] and merged["flaky"]
    assert merged["scores"]["pass_count"] == 2 and merged["scores"]["consistency"] == pytest.approx(0.6667)
    summary = run.summarize([merged], repeats=3)
    assert summary["metrics"]["pass_at_k"] == 1.0 and summary["metrics"]["pass_all_k"] == 0.0
    assert summary["metrics"]["flaky_rate"] == 1.0


def test_repeats_option_runs_each_case_k_times(tmp_path: Path):
    run.main(["--mode", "fake", "--ids", "tool-speed-01", "--repeats", "2", "--deepeval", "off",
              "--reports-dir", str(tmp_path)])
    report = json.loads((tmp_path / "evals" / "latest.json").read_text(encoding="utf-8"))
    assert report["cases"][0]["scores"]["repeats"] == 2 and report["summary"]["metrics"]["repeats"] == 2


def test_compare_mode_writes_both_versions_and_comparison(tmp_path: Path):
    run.main(["--mode", "fake", "--subset", "refusal", "--deepeval", "off", "--compare", "v1", "v2",
              "--reports-dir", str(tmp_path)])
    compare = json.loads((tmp_path / "evals_compare" / "latest.json").read_text(encoding="utf-8"))
    assert compare["summary"]["versions"] == ["v1", "v2"]
    assert "pass_rate" in compare["summary"]["table"]
    assert {c["change"] for c in compare["cases"]} <= {"fixed", "regressed", "same"}
    latest = json.loads((tmp_path / "evals" / "latest.json").read_text(encoding="utf-8"))
    assert latest["prompt_version"] == "v2"
    assert len(list((tmp_path / "evals").glob("2*.json"))) == 2


def test_compare_runs_marks_fixed_and_regressed():
    def mk(passed_a, passed_b):
        return {"summary": {"pass_rate": 0.5, "metrics": {"faithfulness": 0.5}},
                "cases": [{"id": "a", "category": "x", "passed": passed_a, "reason": ""},
                          {"id": "b", "category": "x", "passed": passed_b, "reason": "broke"}]}
    summary, rows = run.compare_runs({"v1": mk(False, True), "v2": mk(True, False)})
    changes = {r["id"]: r["change"] for r in rows}
    assert changes == {"a": "fixed", "b": "regressed"} and summary["metrics"] == {"fixed": 1, "regressed": 1}


def test_p95():
    assert run.p95([]) == 0 and run.p95([5]) == 5 and run.p95(list(range(1, 101))) == 95


def test_select_cases_skips_needs_review():
    cases = [{"id": "a", "category": "manual_qa", "tags": []},
             {"id": "b", "category": "manual_qa", "tags": [], "needs_review": True}]
    assert [c["id"] for c in run.select_cases(cases)] == ["a"]
    assert len(run.select_cases(cases, include_review=True)) == 2
