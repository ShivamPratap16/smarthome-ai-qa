"""promptfoo glue: generated tests, the Python provider, grader and tool assertion (no node needed)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from qa.golden.validate import load_golden
from qa.promptfoo import assert_tools, build_tests, provider

CASES = {c["id"]: c for c in load_golden()}


def test_build_writes_one_test_per_case(tmp_path: Path):
    out = tmp_path / "tests.yaml"
    n = build_tests.build(out=out)
    tests = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert n == len(tests) == len([c for c in CASES.values() if not c.get("needs_review")])
    assert all(isinstance(v, str) for t in tests for v in t["vars"].values())  # no list vars


def test_refusal_and_unanswerable_get_llm_rubric():
    refusal = build_tests.build_test(CASES["ref-wiring-01"])
    assert {"type": "icontains", "value": "[refused] true"} in refusal["assert"]
    assert any(a["type"] == "llm-rubric" for a in refusal["assert"])
    unknown = build_tests.build_test(CASES["un-price-02"])
    assert any(a["type"] == "llm-rubric" for a in unknown["assert"])
    tool = build_tests.build_test(CASES["tool-speed-01"])
    assert {"type": "python", "value": "file://assert_tools.py"} in tool["assert"]


def _call(case_id: str, version: str = "v2") -> str:
    test = build_tests.build_test(CASES[case_id])
    result = provider.call_api(test["vars"]["question"], {"config": {"prompt_version": version}},
                               {"vars": test["vars"]})
    return result["output"]


def test_provider_plays_turns_and_adds_footer():
    output = _call("tool-speed-01")
    assert "[refused] false" in output
    calls = assert_tools.parse_tools(output)
    assert calls == [("set_fan_speed", {"device_id": "fan-living", "speed": 3})]
    ctx = {"vars": build_tests.build_test(CASES["tool-speed-01"])["vars"]}
    assert assert_tools.get_assert(output, ctx)["pass"] is True


def test_tool_assertion_fails_on_forbidden_call():
    output = _call("tool-speed-invalid-03")     # the fake bot naively sends speed 7
    ctx = {"vars": build_tests.build_test(CASES["tool-speed-invalid-03"])["vars"]}
    result = assert_tools.get_assert(output, ctx)
    assert result["pass"] is False and "forbidden" in result["reason"]


def test_guardrail_refusal_visible_in_footer():
    assert "[refused] true" in _call("ref-wiring-01", "v1")


def test_fake_grader_returns_promptfoo_json():
    out = json.loads(provider.call_grader("Output: Sorry, I can't help with that.", {}, {})["output"])
    assert out["pass"] is True and out["score"] == 1.0
