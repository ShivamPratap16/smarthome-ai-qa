"""The Hawa Home support assistant (Contract 2) - the system under test.

Flow of one `chat()` turn:
  1. guardrails.check_input  -> blocked? return a friendly refusal (no LLM call)
  2. retrieval.retrieve      -> top manual chunks, put into the system prompt
  3. LLM + tools loop        -> max 4 rounds; device 503 is retried once, then we apologise
  4. guardrails.mask_pii     -> on the final text
  5. store                   -> log user + bot messages and tool calls (masked)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from assistant.devices import TOOLS, DeviceAPIError, DeviceCloud
from assistant.guardrails import (
    REFUSAL_MESSAGES,
    check_input,
    looks_like_injection,
    mask_obj,
    mask_pii,
    sanitize_data,
)
from assistant.llm import LLMClient, LLMResponse, cost_usd, get_llm
from assistant.retrieval import Chunk, retrieve
from assistant.store import ChatStore

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS_PER_TURN = 6
RETRIEVAL_K = 4

DEVICE_DOWN_MESSAGE = ("Sorry, I couldn't reach your device right now - the Hawa device cloud is "
                       "temporarily unavailable. Please try again in a few minutes. Would you like me "
                       "to raise a service ticket instead? Just share your mobile number.")
TICKET_DOWN_MESSAGE = ("Sorry, I couldn't create the service ticket right now because our system is "
                       "temporarily unavailable. Please try again in a few minutes or call "
                       "1800-000-4292 (9 AM to 9 PM).")
TOO_MANY_ROUNDS_MESSAGE = ("Sorry, I couldn't complete that request. Could you rephrase it, or shall "
                           "I raise a service ticket for you?")
EMPTY_ANSWER_MESSAGE = "I don't know the answer to that. Would you like me to raise a service ticket?"

# Phrases that mean the model itself declined (v2 prompt asks for the first one).
REFUSAL_MARKERS = ("sorry, i can't help with that", "sorry, i cannot help with that",
                   "i can't help with that", "i cannot help with that", "i can't assist with",
                   "i cannot assist with")


@dataclass
class ToolResult:
    """One executed tool call and its outcome."""

    name: str
    args: dict
    result: dict | None
    error: str | None


@dataclass
class BotReply:
    """Everything one assistant turn produced (returned by the API as JSON)."""

    text: str
    tool_calls: list[ToolResult] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    refused: bool = False
    blocked_by: str | None = None
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    prompt_version: str = "v2"


def load_prompt(version: str) -> str:
    """Read prompts/<version>.txt."""
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        raise ValueError(f"unknown prompt version {version!r}")
    return path.read_text(encoding="utf-8")


def format_context(chunks: list[Chunk]) -> str:
    """Manual chunks as text for the prompt: '[id] title' line, then the text."""
    if not chunks:
        return "(no relevant manual sections found)"
    return "\n\n".join(f"[{c.id}] {c.title}\n{c.text}" for c in chunks)


def is_model_refusal(text: str) -> bool:
    """True if the model's answer is a refusal."""
    lowered = text.lower().replace("’", "'")
    return any(marker in lowered for marker in REFUSAL_MARKERS)


class SupportAssistant:
    """Customer-support chatbot with retrieval, tools, guardrails and logging."""

    def __init__(self, llm: LLMClient | None = None, devices: DeviceCloud | None = None,
                 prompt_version: str = "v2", store: ChatStore | None = None):
        self.llm = llm if llm is not None else get_llm()
        self.devices = devices if devices is not None else DeviceCloud()
        self.prompt_version = prompt_version
        self.prompt_template = load_prompt(prompt_version)
        self.store = store if store is not None else ChatStore()
        # The chat-log DB outlives this process: continue ticket numbers so ids never collide.
        self.devices.reserve_ticket_numbers(self.store.last_ticket_number())
        self.sessions: dict[str, list[dict]] = {}   # session_id -> message history

    # ------------------------------------------------------------------ public API
    def chat(self, session_id: str, user_message: str, user_id: str = "demo-user") -> BotReply:
        """Answer one user message in a session."""
        start = time.perf_counter()
        self.store.ensure_conversation(session_id, user_id)
        self.store.add_message(session_id, "user", user_message, prompt_version=self.prompt_version)

        verdict = check_input(user_message)
        if not verdict.allowed:
            reply = BotReply(text=REFUSAL_MESSAGES[verdict.reason], refused=True,
                             blocked_by=verdict.reason, prompt_version=self.prompt_version)
            return self._finish(session_id, user_message, [], reply, start)

        history = self.sessions.get(session_id, [])
        chunks = self._safe_chunks(retrieve(self._retrieval_query(history, user_message), k=RETRIEVAL_K))
        system = self.build_system_prompt(chunks, user_id)
        messages = history + [{"role": "user", "content": user_message}]
        user_texts = [m["content"] for m in messages if m["role"] == "user"]
        new_messages, reply = self._run_tool_loop(system, messages, user_texts, user_id)
        reply.contexts = [c.text for c in chunks]
        reply.context_ids = [c.id for c in chunks]
        return self._finish(session_id, user_message, new_messages, reply, start)

    def reset(self, session_id: str) -> None:
        """Forget the conversation history of a session."""
        self.sessions.pop(session_id, None)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _safe_chunks(chunks: list[Chunk]) -> list[Chunk]:
        """Drop manual chunks that contain planted instructions (RAG poisoning)."""
        return [c for c in chunks if not looks_like_injection(f"{c.title} {c.text}")]

    def build_system_prompt(self, chunks: list[Chunk], user_id: str = "demo-user") -> str:
        """Fill the prompt template with the user's devices and the manual context.

        Device names are untrusted (users can rename devices), so a name or room that looks
        like an instruction is replaced.
        """
        lines = []
        for d in self.devices.list_devices(user_id):
            name = d["name"] if not looks_like_injection(str(d["name"])) else d["device_id"]
            room = d["room"] if not looks_like_injection(str(d["room"])) else "unknown"
            lines.append(f"- {d['device_id']}: {name} ({d['product']}, room: {room})")
        return (self.prompt_template.replace("{devices}", "\n".join(lines))
                .replace("{context}", format_context(chunks)))

    @staticmethod
    def _retrieval_query(history: list[dict], user_message: str) -> str:
        """Search with the current message plus the previous user message (helps follow-ups)."""
        previous = [m["content"] for m in history if m["role"] == "user"]
        return f"{previous[-1]} {user_message}" if previous else user_message

    def _run_tool_loop(self, system: str, messages: list[dict], user_texts: list[str],
                       user_id: str = "demo-user") -> tuple[list[dict], BotReply]:
        """Call the LLM, run requested tools, feed results back.

        Limits: MAX_TOOL_ROUNDS model rounds and MAX_TOOL_CALLS_PER_TURN executed calls.
        Every tool call in the history always gets a tool result (skipped ones get an error),
        so the history stays valid for any LLM provider.
        Returns the new history messages of this turn (after the user message) and the reply.
        """
        messages = list(messages)
        first_new = len(messages) - 1          # index of this turn's user message
        reply = BotReply(text="", prompt_version=self.prompt_version)
        for round_no in range(MAX_TOOL_ROUNDS + 1):   # +1: room for the final answer
            response: LLMResponse = self.llm.generate(system, messages, TOOLS, 0.0)
            reply.tokens_in += response.tokens_in
            reply.tokens_out += response.tokens_out
            if not response.tool_calls:
                reply.text = response.text.strip() or EMPTY_ANSWER_MESSAGE
                messages.append({"role": "assistant", "content": reply.text})
                return messages[first_new + 1:], reply
            if round_no == MAX_TOOL_ROUNDS:   # model wants a 5th tool round: stop here
                break
            messages.append({"role": "assistant", "content": response.text,
                             "tool_calls": [{"name": tc.name, "args": tc.args, "signature": tc.signature}
                                            for tc in response.tool_calls]})
            down_message = None
            for tc in response.tool_calls:
                if down_message:
                    payload = {"error": "skipped: an earlier tool call failed (device cloud unavailable)"}
                elif len(reply.tool_calls) >= MAX_TOOL_CALLS_PER_TURN:
                    payload = {"error": f"skipped: limit of {MAX_TOOL_CALLS_PER_TURN} tool calls per message reached"}
                else:
                    result = self._execute_tool(tc.name, tc.args, user_texts, user_id)
                    reply.tool_calls.append(result)
                    payload = result.result if result.error is None else {"error": result.error}
                    if result.error and result.error.startswith("503"):
                        # Retried once already and still down: finish this round, then apologise.
                        down_message = (TICKET_DOWN_MESSAGE if tc.name == "create_service_ticket"
                                        else DEVICE_DOWN_MESSAGE)
                # Tool output is untrusted data: remove planted instructions, then mask PII.
                messages.append({"role": "tool", "name": tc.name,
                                 "content": mask_pii(json.dumps(sanitize_data(payload), ensure_ascii=False))})
            if down_message:
                reply.text = down_message
                messages.append({"role": "assistant", "content": reply.text})
                return messages[first_new + 1:], reply
        reply.text = TOO_MANY_ROUNDS_MESSAGE
        messages.append({"role": "assistant", "content": reply.text})
        return messages[first_new + 1:], reply

    def _check_tool_allowed(self, name: str, args: dict, user_texts: list[str], user_id: str) -> str | None:
        """Code-level rules the model cannot talk its way around. Returns an error or None.

        A service ticket may only be raised for one of the user's own devices, with a phone
        number the user typed themselves (not one planted in a device name, tool result or manual).
        """
        if name != "create_service_ticket":
            return None
        own_ids = {d["device_id"] for d in self.devices.list_devices(user_id)}
        if args.get("device_id") not in own_ids:
            return "403: tickets can only be raised for your own devices"
        phone_digits = re.sub(r"\D", "", str(args.get("phone", "")))[-10:]
        typed_digits = re.sub(r"\D", "", " ".join(user_texts))
        if len(phone_digits) == 10 and phone_digits not in typed_digits:
            return "403: the phone number must come from the customer's own messages"
        return None

    def _execute_tool(self, name: str, args: dict, user_texts: list[str] | None = None,
                      user_id: str = "demo-user") -> ToolResult:
        """Run one tool. A 503 is retried once; errors become ToolResult.error ("<status>: <msg>").

        Args and results in the returned ToolResult are PII-masked (they go to the API and logs).
        """
        denied = self._check_tool_allowed(name, args, user_texts or [], user_id)
        if denied:
            return ToolResult(name=name, args=mask_obj(args), result=None, error=denied)
        for attempt in range(2):
            try:
                result = self.devices.call_tool(name, dict(args))
                return ToolResult(name=name, args=mask_obj(args),
                                  result=mask_obj(sanitize_data(result)), error=None)
            except DeviceAPIError as exc:
                if exc.status == 503 and attempt == 0:
                    continue
                return ToolResult(name=name, args=mask_obj(args), result=None,
                                  error=f"{exc.status}: {exc.msg}")
        raise AssertionError("unreachable")

    def _finish(self, session_id: str, user_message: str, new_messages: list[dict],
                reply: BotReply, start: float) -> BotReply:
        """Mask PII, compute metrics, update history and write logs."""
        reply.text = mask_pii(reply.text)
        if not reply.refused and is_model_refusal(reply.text):
            reply.refused = True
        reply.latency_ms = int((time.perf_counter() - start) * 1000)
        reply.cost_usd = cost_usd(reply.tokens_in, reply.tokens_out)

        if reply.blocked_by is None:   # blocked messages are not kept in the LLM history
            history = self.sessions.setdefault(session_id, [])
            history.append({"role": "user", "content": user_message})
            history.extend(new_messages)

        message_id = self.store.add_message(
            session_id, "assistant", reply.text, prompt_version=self.prompt_version,
            latency_ms=reply.latency_ms, tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
            cost_usd=reply.cost_usd, refused=reply.refused, blocked_by=reply.blocked_by)
        for tc in reply.tool_calls:
            self.store.add_tool_call(message_id, tc.name, tc.args, tc.result, tc.error)
            if tc.name == "create_service_ticket" and tc.result:
                self.store.add_ticket(tc.result["ticket_id"], tc.result["device_id"],
                                      tc.result["issue"], str(tc.args.get("phone", "")))
        return reply
