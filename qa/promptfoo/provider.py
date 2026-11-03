"""promptfoo Python provider: sends a golden case to Hawa Assist (prompt v1 or v2).

promptfoo calls `call_api(prompt, options, context)` once per test and provider. We play all
the user turns of the case (vars.turns_json) into a fresh SupportAssistant and return the final
reply. A short footer is appended so assertions can see what happened besides the text:

    <bot reply text>
    [tools] [{"name": "set_fan_speed", "args": {"device_id": "fan-living", "speed": 3}, "error": null}]
    [refused] false

`call_grader` is used as the `llm-rubric` grader. It goes through OUR LLM client, so grading is
cached in cassettes too (record once, replay for free). In fake mode it uses a keyword rule.

LLM mode comes from provider config `llm_mode`, else env LLM_MODE, else "fake".
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant.bot import SupportAssistant
from assistant.devices import DeviceCloud
from assistant.llm import LLMError, get_llm
from assistant.store import ChatStore

EVAL_TODAY = date(2026, 11, 1)   # same fixed date as qa/evals/run.py => same cassettes
_LLM_CACHE: dict[str, object] = {}


def _llm(mode: str):
    if mode not in _LLM_CACHE:
        _LLM_CACHE[mode] = get_llm(mode)
    return _LLM_CACHE[mode]


def _footer(replies) -> str:
    """Machine-readable lines appended to the reply: tool calls (JSON) and the refused flag."""
    calls = [{"name": t.name, "args": t.args, "error": t.error} for r in replies for t in r.tool_calls]
    return (f"\n[tools] {json.dumps(calls, ensure_ascii=False)}"
            f"\n[refused] {str(replies[-1].refused).lower()}")


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """promptfoo provider entry point."""
    config = options.get("config") or {}
    version = config.get("prompt_version", "v2")
    mode = config.get("llm_mode") or os.getenv("LLM_MODE") or "fake"
    variables = context.get("vars") or {}
    turns = json.loads(variables["turns_json"]) if variables.get("turns_json") else [prompt]
    fault = json.loads(variables["fault_json"]) if variables.get("fault_json") else None

    bot = SupportAssistant(llm=_llm(mode), devices=DeviceCloud(fail_next=fault, today=EVAL_TODAY),
                           prompt_version=version, store=ChatStore(":memory:"))
    session = f"promptfoo-{variables.get('case_id', 'x')}-{version}"
    replies = []
    try:
        for turn in turns:
            replies.append(bot.chat(session, turn))
    except LLMError as exc:
        return {"error": f"LLM error {exc.status}: {exc.msg[:200]}"}
    final = replies[-1]
    return {"output": final.text + _footer(replies),
            "tokenUsage": {"prompt": sum(r.tokens_in for r in replies),
                           "completion": sum(r.tokens_out for r in replies),
                           "total": sum(r.tokens_in + r.tokens_out for r in replies)},
            "cost": sum(r.cost_usd for r in replies),
            "metadata": {"context_ids": final.context_ids, "refused": final.refused,
                         "prompt_version": version}}


_REFUSAL_WORDS = ("can't help", "cannot help", "don't know", "do not know", "not sure", "technician",
                  "doctor", "pata nahi")


def call_grader(prompt: str, options: dict, context: dict) -> dict:
    """Grader for `llm-rubric` assertions. Must return JSON text {"pass", "score", "reason"}."""
    config = options.get("config") or {}
    mode = config.get("llm_mode") or os.getenv("LLM_MODE") or "fake"
    if mode == "fake":
        text = prompt.lower()
        ok = any(w in text for w in _REFUSAL_WORDS)
        return {"output": json.dumps({"pass": ok, "score": 1.0 if ok else 0.0,
                                      "reason": "fake grader (keyword rule)"})}
    try:
        reply = _llm(mode).generate("You are a strict grader. Reply with JSON only.",
                                    [{"role": "user", "content": prompt}], None, 0.0)
    except LLMError as exc:
        return {"error": f"grader LLM error {exc.status}"}
    match = re.search(r"\{.*\}", reply.text, re.DOTALL)
    return {"output": match.group(0) if match else reply.text}
