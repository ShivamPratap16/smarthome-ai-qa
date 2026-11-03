"""Deterministic checks, retrieval metrics and judge parsing."""

from __future__ import annotations

import json

import pytest

from qa.evals import checks
from qa.evals.judge import RubricJudge, extract_json, fake_judge_llm, heuristic_scores
from qa.evals.retrieval_metrics import average_precision, raw_precision, recall, retrieval_scores
from assistant.llm import FakeLLM


def test_tool_match_is_loose_on_types_and_ignores_phone():
    exp = {"name": "set_fan_speed", "args": {"device_id": "fan-living", "speed": 3}}
    assert checks.tool_call_matches(exp, "set_fan_speed", {"device_id": "fan-living", "speed": 3.0})
    assert not checks.tool_call_matches(exp, "set_fan_speed", {"device_id": "fan-living", "speed": 4})
    ticket = {"name": "create_service_ticket", "args": {"device_id": "fan-living", "phone": "9876543210"}}
    assert checks.tool_call_matches(ticket, "create_service_ticket",
                                    {"device_id": "fan-living", "phone": "[PHONE]", "issue": "noise"})


def test_check_tools_reports_missing_and_forbidden():
    ok, why = checks.check_tools([{"name": "get_device_status", "args": {"device_id": "lock-main"}}],
                                 ["set_fan_speed"], [("set_fan_speed", {"speed": 7})])
    assert not ok and "missing" in why and "forbidden" in why
    assert checks.check_tools([], [], [("get_device_status", {})]) == (True, "")


def test_key_facts_normalises_numbers_and_case():
    score, missing = checks.check_key_facts("Replace every 6 Months or 3,000 litres", ["6 months", "3000", "UV"])
    assert missing == ["UV"] and score == pytest.approx(2 / 3)


@pytest.mark.parametrize("text,expected", [
    ("I don't know the price. I can raise a ticket.", True),
    ("I’m not sure - the manual doesn't mention that.", True),
    ("Mujhe iska pata nahi hai.", True),
    ("The Aero costs Rs 3,999.", False),
])
def test_abstained(text, expected):
    assert checks.abstained(text) is expected


def test_unsupported_numbers_flags_invented_specs_only():
    sources = ["Boost mode runs the fan at 380 RPM for a fixed 2 hours."]
    assert checks.unsupported_numbers("Boost: 380 RPM for 2 hours.", sources) == []
    assert checks.unsupported_numbers("It makes 42 dB at 380 RPM.", sources) == ["42"]
    assert checks.unsupported_numbers("1. Press B\n2. Wait", []) == []   # list numbering ignored


def test_is_hinglish():
    assert checks.is_hinglish("Ho gaya, living room ka fan ab speed 4 pe hai.")
    assert not checks.is_hinglish("Done, the living room fan is now on speed 4.")


def test_retrieval_formulas():
    retrieved = ["a", "x", "b", "y"]
    assert recall(retrieved, ["a", "b", "c"]) == pytest.approx(2 / 3)
    assert raw_precision(retrieved, ["a", "b"]) == 0.5
    assert average_precision(retrieved, ["a", "b"]) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert average_precision(["x", "a"], ["a"]) == 0.5
    assert retrieval_scores(retrieved, []) == {}


def test_ragas_and_builtin_agree():
    pytest.importorskip("ragas")
    retrieved, expected = ["a", "x", "b", "y"], ["a", "b", "c"]
    with_ragas = retrieval_scores(retrieved, expected, use_ragas=True)
    builtin = retrieval_scores(retrieved, expected, use_ragas=False)
    assert with_ragas == builtin


def test_extract_json_handles_fences_and_noise():
    assert extract_json('Sure!\n```json\n{"faithfulness": 0.8}\n```') == {"faithfulness": 0.8}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_rubric_judge_parses_and_clamps_scores():
    reply = json.dumps({"faithfulness": 1.4, "answer_relevancy": "0.5", "correctness": None,
                        "hallucination": True, "unsupported_claims": ["42 dB"], "reason": "x"})
    judge = RubricJudge(FakeLLM([reply]))
    result = judge.score("q", "a", ["ctx"], "ref")
    assert (result.faithfulness, result.answer_relevancy, result.correctness) == (1.0, 0.5, 0.0)
    assert result.hallucination and result.unsupported_claims == ["42 dB"] and judge.calls == 1


def test_fake_judge_rewards_grounded_answers():
    judge = RubricJudge(fake_judge_llm())
    good = judge.score("boost speed?", "Boost runs at 380 RPM for 2 hours.",
                       ["Boost mode runs the fan at 380 RPM for 2 hours."], "380 RPM for 2 hours")
    bad = judge.score("boost speed?", "Boost uses turbine magnets and 900 watts.",
                      ["Boost mode runs the fan at 380 RPM for 2 hours."], "380 RPM for 2 hours")
    assert good.faithfulness > bad.faithfulness and good.correctness > bad.correctness
    assert heuristic_scores("q", "", [], "r")["faithfulness"] == 1.0
