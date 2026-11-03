"""A deliberately "fully persuaded" fake model for worst-case red-team runs (fake mode).

Idea: assume the attack WORKED on the LLM. The model then does exactly what the attacker
wanted - leak the prompt, echo personal data, call a dangerous tool. Whatever still goes
right after that was saved by our *code* (guardrails, masking, tool design), not by luck.

Honesty rule: a step with `if_seen: <text>` only fires once that text actually reached the
model (system prompt or messages). So if the code filtered a planted payload out, the
model behaves normally, exactly like a real model that never saw the payload.
"""

from __future__ import annotations

import json

from assistant.llm import LLMResponse, ToolCall, default_fake_responder


class CompromisedLLM:
    """Plays scripted 'attacker wins' steps; behaves like the normal fake model otherwise."""

    def __init__(self, steps: list[dict] | None = None):
        self.steps: list[dict] = []
        for step in steps or []:
            self.steps.extend([step] * int(step.get("rounds", 1)))
        self.calls: list[dict] = []

    def generate(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        """Return the next attacker-controlled step, or a benign answer."""
        self.calls.append({"system": system, "messages": json.loads(json.dumps(messages))})
        visible = (system + json.dumps(messages, ensure_ascii=False)).lower()
        if not self.steps or (self.steps[0].get("if_seen", "").lower() not in visible):
            return default_fake_responder(system, messages, tools)
        step = self.steps.pop(0)
        if "tools" in step:
            calls = [ToolCall(t["name"], dict(t.get("args") or {}))
                     for t in step["tools"] for _ in range(int(t.get("times", 1)))]
            return LLMResponse(text="", tool_calls=calls, tokens_in=len(visible) // 4, tokens_out=10)
        text = self._fill(step.get("say", ""), system, messages)
        return LLMResponse(text=text, tokens_in=len(visible) // 4, tokens_out=len(text) // 4)

    @staticmethod
    def _fill(template: str, system: str, messages: list[dict]) -> str:
        """Replace {system_prompt} and {last_user} placeholders."""
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return template.replace("{system_prompt}", system).replace("{last_user}", last_user)
