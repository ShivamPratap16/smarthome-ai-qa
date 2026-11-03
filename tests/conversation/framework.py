"""A small, reusable framework for testing multi-turn conversations with Hawa Assist.

Tests read like a script of the chat:

    (Conversation()
        .user("Is my living room fan on?", model=[call("get_device_status", device_id="fan-living"),
                                                  say("Yes, it is at speed 2.")])
        .expect_tool("get_device_status", device_id="fan-living")
        .user("set it to 4", model=[call("set_fan_speed", device_id="fan-living", speed=4), say("Done")])
        .expect_tool("set_fan_speed", device_id="fan-living", speed=4)
        .expect_model_saw("Is my living room fan on?"))     # history was sent to the model

* `.user()` sends the message immediately; every `expect_*` checks the LAST reply.
* `model=[...]` queues what the scripted FakeLLM answers during that turn (fake mode).
  With a real model (live mode) leave `model` out and use the loose checks
  (`expect_text_contains_any`, `expect_tool_called`) because the wording changes between runs.
* Each method returns `self`, so a whole conversation is one readable chain.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from assistant.bot import BotReply, SupportAssistant
from assistant.devices import DeviceCloud
from assistant.llm import FakeLLM, LLMClient, LLMResponse, ToolCall
from assistant.store import ChatStore
from qa.redteam.detectors import find_pii, looks_like_refusal, store_texts


def say(text: str) -> LLMResponse:
    """Scripted model answer with plain text."""
    return LLMResponse(text=text, tokens_in=50, tokens_out=max(1, len(text) // 4))


def call(name: str, **args) -> LLMResponse:
    """Scripted model answer that requests one tool call."""
    return LLMResponse(text="", tool_calls=[ToolCall(name, args)], tokens_in=50, tokens_out=10)


def calls(*tool_calls: tuple[str, dict]) -> LLMResponse:
    """Scripted model answer that requests several tool calls in one round."""
    return LLMResponse(text="", tool_calls=[ToolCall(n, a) for n, a in tool_calls], tokens_in=50, tokens_out=10)


@dataclass
class Turn:
    """One user message and what came back."""

    user: str
    reply: BotReply
    llm_calls_before: int     # index into FakeLLM.calls where this turn's model calls start


class ConversationError(AssertionError):
    """An expectation failed. The message includes the transcript so far."""


class Conversation:
    """Fluent builder that drives a SupportAssistant through a multi-turn chat."""

    def __init__(self, llm: LLMClient | None = None, devices: DeviceCloud | None = None,
                 prompt_version: str = "v2", session_id: str = "conv-test", user_id: str = "demo-user"):
        self.llm = llm if llm is not None else FakeLLM()
        self.devices = devices if devices is not None else DeviceCloud()
        self.store = ChatStore(":memory:")
        self.bot = SupportAssistant(llm=self.llm, devices=self.devices,
                                    prompt_version=prompt_version, store=self.store)
        self.session_id = session_id
        self.user_id = user_id
        self.turns: list[Turn] = []

    # ------------------------------------------------------------------ driving
    @property
    def is_fake(self) -> bool:
        """True when the model is the scripted FakeLLM (exact assertions are fair)."""
        return isinstance(self.llm, FakeLLM)

    def user(self, text: str, model: list[LLMResponse | str] | None = None) -> Conversation:
        """Send a user message. `model` = scripted FakeLLM answers for this turn."""
        if model:
            if not self.is_fake:
                raise TypeError("model=[...] scripts only work with FakeLLM")
            self.llm.queue.extend(model)
        before = len(getattr(self.llm, "calls", []))
        reply = self.bot.chat(self.session_id, text, user_id=self.user_id)
        self.turns.append(Turn(text, reply, before))
        if model and self.llm.queue:
            leftover = len(self.llm.queue)
            self.llm.queue.clear()
            self._fail(f"{leftover} scripted model answer(s) were never used")
        if model and len(self.model_calls_this_turn()) > len(model):
            self._fail(f"model was called {len(self.model_calls_this_turn())} times, "
                       f"only {len(model)} answers were scripted")
        return self

    def reset(self) -> Conversation:
        """Start over in the same session (like the UI's 'new chat' button)."""
        self.bot.reset(self.session_id)
        return self

    @property
    def last(self) -> BotReply:
        """The most recent reply."""
        if not self.turns:
            raise ConversationError("no user message sent yet")
        return self.turns[-1].reply

    def model_calls_this_turn(self) -> list[dict]:
        """What the FakeLLM received during the last turn (system, messages, tools)."""
        return self.llm.calls[self.turns[-1].llm_calls_before:]

    # ------------------------------------------------------------------ expectations
    def _fail(self, message: str) -> None:
        raise ConversationError(f"{message}\n--- transcript ---\n{self.transcript()}")

    def expect_tool(self, name: str, **args) -> Conversation:
        """A tool with this name was called with (at least) these exact args."""
        for tc in self.last.tool_calls:
            if tc.name == name and all(tc.args.get(k) == v for k, v in args.items()):
                return self
        self._fail(f"expected tool {name}({args}), got {self._tools()}")

    def expect_tool_called(self, name: str) -> Conversation:
        """A tool with this name was called (any args) - good for live mode."""
        if name not in [tc.name for tc in self.last.tool_calls]:
            self._fail(f"expected a call to {name}, got {self._tools()}")
        return self

    def expect_tool_sequence(self, *names: str) -> Conversation:
        """Tools were called exactly in this order during the last turn."""
        got = [tc.name for tc in self.last.tool_calls]
        if got != list(names):
            self._fail(f"expected tool sequence {list(names)}, got {got}")
        return self

    def expect_no_tool(self, name: str | None = None) -> Conversation:
        """No tool (or no tool with this name) was called."""
        bad = [tc.name for tc in self.last.tool_calls if name in (None, tc.name)]
        if bad:
            self._fail(f"expected no {name or 'tool'} call, got {bad}")
        return self

    def expect_tool_error(self, name: str, status: int) -> Conversation:
        """The tool was called and failed with this HTTP-like status."""
        for tc in self.last.tool_calls:
            if tc.name == name and tc.error and tc.error.startswith(str(status)):
                return self
        self._fail(f"expected {name} to fail with {status}, got {self._tools()}")

    def expect_tool_count(self, n: int) -> Conversation:
        """Exactly n tool executions in the last turn."""
        if len(self.last.tool_calls) != n:
            self._fail(f"expected {n} tool calls, got {len(self.last.tool_calls)}")
        return self

    def expect_text(self, text: str) -> Conversation:
        """The reply text is exactly this (fake mode / fixed system messages only)."""
        if self.last.text != text:
            self._fail(f"expected text {text!r}, got {self.last.text!r}")
        return self

    def expect_text_contains(self, *parts: str) -> Conversation:
        """Every part appears in the reply (case-insensitive)."""
        missing = [p for p in parts if p.lower() not in self.last.text.lower()]
        if missing:
            self._fail(f"reply misses {missing}: {self.last.text!r}")
        return self

    def expect_text_contains_any(self, *options: str) -> Conversation:
        """At least one option appears - a loose check for non-deterministic wording."""
        if not any(o.lower() in self.last.text.lower() for o in options):
            self._fail(f"reply has none of {list(options)}: {self.last.text!r}")
        return self

    def expect_text_matches(self, pattern: str) -> Conversation:
        """The reply matches a regex (case-insensitive)."""
        if not re.search(pattern, self.last.text, re.IGNORECASE):
            self._fail(f"reply does not match /{pattern}/: {self.last.text!r}")
        return self

    def expect_text_not_contains(self, *parts: str) -> Conversation:
        """None of the parts appear in the reply."""
        found = [p for p in parts if p.lower() in self.last.text.lower()]
        if found:
            self._fail(f"reply must not contain {found}: {self.last.text!r}")
        return self

    def expect_refusal(self, blocked_by: str | None = None) -> Conversation:
        """The bot declined (guardrail or model). Optionally check which guardrail fired."""
        if not (self.last.refused or looks_like_refusal(self.last.text)):
            self._fail(f"expected a refusal, got {self.last.text!r}")
        if blocked_by is not None and self.last.blocked_by != blocked_by:
            self._fail(f"expected blocked_by={blocked_by!r}, got {self.last.blocked_by!r}")
        return self

    def expect_not_refused(self) -> Conversation:
        """The bot actually helped."""
        if self.last.refused:
            self._fail(f"unexpected refusal ({self.last.blocked_by}): {self.last.text!r}")
        return self

    def expect_no_pii(self) -> Conversation:
        """No personal data in the reply, its tool args, or anything logged so far."""
        texts = [self.last.text] + [json.dumps(tc.args) for tc in self.last.tool_calls]
        texts += store_texts(self.store, self.session_id)
        found = sorted({p for t in texts for p in find_pii(t)})
        if found:
            self._fail(f"PII found: {found}")
        return self

    def expect_model_saw(self, text: str) -> Conversation:
        """The model's input during the last turn contained `text` (history / tool results)."""
        seen = json.dumps(self.model_calls_this_turn(), ensure_ascii=False)
        if text not in seen:
            self._fail(f"model never saw {text!r} this turn")
        return self

    def expect_model_not_saw(self, text: str) -> Conversation:
        """The model's input during the last turn did NOT contain `text`."""
        if text in json.dumps(self.model_calls_this_turn(), ensure_ascii=False):
            self._fail(f"model should not have seen {text!r}")
        return self

    def expect_model_calls(self, n: int) -> Conversation:
        """The LLM was called exactly n times during the last turn."""
        got = len(self.model_calls_this_turn())
        if got != n:
            self._fail(f"expected {n} LLM calls this turn, got {got}")
        return self

    def expect_device(self, device_id: str, **state) -> Conversation:
        """The device cloud now holds this state, e.g. expect_device("fan-living", speed=4)."""
        device = self.devices.devices[device_id]
        wrong = {k: device.get(k) for k, v in state.items() if device.get(k) != v}
        if wrong:
            self._fail(f"{device_id} expected {state}, actual {wrong}")
        return self

    def expect(self, check, description: str) -> Conversation:
        """Escape hatch: any predicate on the last BotReply."""
        if not check(self.last):
            self._fail(f"custom check failed: {description}")
        return self

    # ------------------------------------------------------------------ reporting
    def _tools(self) -> list[str]:
        return [f"{tc.name}({tc.args}) -> {tc.error or 'ok'}" for tc in self.last.tool_calls]

    def transcript(self) -> str:
        """The chat so far, for readable failure messages."""
        lines = []
        for t in self.turns:
            lines.append(f"USER: {t.user[:200]}")
            for tc in t.reply.tool_calls:
                lines.append(f"  TOOL {tc.name}({tc.args}) -> {tc.error or tc.result}")
            flag = f" [refused:{t.reply.blocked_by or 'model'}]" if t.reply.refused else ""
            lines.append(f"BOT{flag}: {t.reply.text[:300]}")
        return "\n".join(lines)
