"""promptfoo python assertion: did the bot call the expected tools (and not the forbidden ones)?

The provider appends a footer line like ``[tools] [{"name": "set_fan_speed", "args": {...}}]``.
Expected tools come from the test vars (``expected_tools_json``, ``forbidden_tools_json``).
"""

from __future__ import annotations

import json
import re


def parse_tools(output: str) -> list[tuple[str, dict]]:
    """Read the ``[tools] <json>`` footer line back into (name, args) pairs."""
    match = re.search(r"^\[tools\] (.*)$", output, re.MULTILINE)
    if not match:
        return []
    return [(c["name"], c.get("args") or {}) for c in json.loads(match.group(1))]


def get_assert(output: str, context: dict) -> dict:
    """promptfoo entry point: returns {"pass", "score", "reason"}."""
    variables = context.get("vars") or {}
    expected = json.loads(variables.get("expected_tools_json") or "[]")
    forbidden = json.loads(variables.get("forbidden_tools_json") or "[]")
    calls = parse_tools(output)
    problems = []
    for exp in expected:
        wanted = {k: str(v).lower() for k, v in (exp.get("args") or {}).items()
                  if k not in ("phone", "issue")}
        if not any(name == exp["name"] and all(str(args.get(k, "")).lower() == v
                                               for k, v in wanted.items())
                   for name, args in calls):
            problems.append(f"missing {exp['name']}{wanted}")
    problems += [f"forbidden {name} was called" for name, _ in calls if name in forbidden]
    return {"pass": not problems, "score": 0.0 if problems else 1.0,
            "reason": "; ".join(problems) or "tool calls OK"}
