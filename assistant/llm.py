"""LLM layer (Contract 1).

One small interface, `LLMClient.generate(...)`, with four implementations picked by
`get_llm(mode)` / env `LLM_MODE`:

* ``live``   - GeminiClient: real calls to the Gemini REST API (needs GEMINI_API_KEY)
* ``record`` - CassetteLLM wrapping GeminiClient: real call, response saved to ``cassettes/``
* ``replay`` - CassetteLLM without a real client: loads saved responses, raises CassetteMiss
* ``fake``   - FakeLLM: scripted responses, no network at all (unit tests)

Message format used everywhere in this repo (provider-neutral)::

    {"role": "user", "content": "hi"}
    {"role": "assistant", "content": "", "tool_calls": [{"name": "set_fan_speed", "args": {...}}]}
    {"role": "tool", "name": "set_fan_speed", "content": "{\"ok\": true}"}
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Union

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Pricing in USD per 1M tokens (Gemini flash-lite class). The ONLY place prices live.
PRICE_PER_M_INPUT = 0.10
PRICE_PER_M_OUTPUT = 0.40

# Gemini 3 models want the "thought signature" of a function call echoed back. When we
# have none (e.g. history created by FakeLLM) Google documents this dummy value.
DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    """Dollar cost of one or more LLM calls."""
    return round(tokens_in * PRICE_PER_M_INPUT / 1e6 + tokens_out * PRICE_PER_M_OUTPUT / 1e6, 8)


@dataclass
class ToolCall:
    """A function call requested by the model."""

    name: str
    args: dict
    # Opaque Gemini "thoughtSignature"; not part of equality so tests can compare simply.
    signature: str | None = field(default=None, compare=False, repr=False)


@dataclass
class LLMResponse:
    """What the model returned: text and/or tool calls, plus token usage."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


class LLMClient(Protocol):
    """Anything with this `generate` method can be used by the assistant and the judge."""

    def generate(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 0.0) -> LLMResponse: ...


class LLMError(Exception):
    """The LLM could not produce a response (after retries and fallback)."""

    def __init__(self, status: int, msg: str):
        super().__init__(f"{status}: {msg}")
        self.status = status
        self.msg = msg


class CassetteMiss(LLMError):
    """Replay mode: no saved response exists for this exact request."""

    def __init__(self, key: str):
        super().__init__(404, f"no cassette for request {key}")
        self.key = key


# --------------------------------------------------------------------------- Gemini (live)
def to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Convert our provider-neutral messages into Gemini `contents`.

    user -> role "user" text part; assistant -> role "model" with text and/or functionCall
    parts; tool -> role "user" functionResponse part. Consecutive turns with the same role are
    merged (Gemini expects all function responses of one round in one turn).
    """
    contents: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            g_role, parts = "user", [{"text": m["content"]}]
        elif role == "assistant":
            g_role, parts = "model", []
            if m.get("content"):
                parts.append({"text": m["content"]})
            for tc in m.get("tool_calls") or []:
                parts.append({"functionCall": {"name": tc["name"], "args": tc.get("args") or {}},
                              "thoughtSignature": tc.get("signature") or DUMMY_THOUGHT_SIGNATURE})
            if not parts:
                parts.append({"text": ""})
        elif role == "tool":
            try:
                payload = json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                payload = {"result": m["content"]}
            if not isinstance(payload, dict):
                payload = {"result": payload}
            g_role, parts = "user", [{"functionResponse": {"name": m.get("name", "tool"),
                                                            "response": payload}}]
        else:
            raise ValueError(f"unknown role {role!r}")
        if contents and contents[-1]["role"] == g_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": g_role, "parts": parts})
    return contents


def parse_gemini_response(data: dict) -> LLMResponse:
    """Turn a Gemini generateContent JSON response into an LLMResponse."""
    candidates = data.get("candidates") or []
    parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in parts:
        if "functionCall" in part:
            fc = part["functionCall"]
            calls.append(ToolCall(name=fc["name"], args=fc.get("args") or {},
                                  signature=part.get("thoughtSignature")))
        elif "text" in part and not part.get("thought"):
            texts.append(part["text"])
    usage = data.get("usageMetadata") or {}
    tokens_out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return LLMResponse(text="".join(texts).strip(), tool_calls=calls,
                       tokens_in=usage.get("promptTokenCount", 0), tokens_out=tokens_out)


class GeminiClient:
    """Live Gemini client over plain REST (httpx), with retries and model fallback."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 fallback_model: str | None = FALLBACK_MODEL, max_retries: int = 2,
                 timeout: float = 60.0, sleep: Callable[[float], None] = time.sleep):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise LLMError(401, "GEMINI_API_KEY is not set")
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        self.fallback_model = fallback_model if fallback_model != self.model else None
        self.max_retries = max_retries
        self.timeout = timeout
        self.sleep = sleep

    def build_payload(self, system: str, messages: list[dict], tools: list[dict] | None,
                      temperature: float) -> dict:
        """The JSON body for generateContent."""
        payload: dict = {"contents": to_gemini_contents(messages),
                         "generationConfig": {"temperature": temperature}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]
        return payload

    def _post(self, model: str, payload: dict) -> dict:
        """One HTTP call; raises LLMError with the HTTP status on failure."""
        try:
            resp = httpx.post(API_URL.format(model=model), json=payload, timeout=self.timeout,
                              headers={"x-goog-api-key": self.api_key})
        except httpx.HTTPError as exc:
            raise LLMError(503, f"network error: {exc.__class__.__name__}") from exc
        if resp.status_code != 200:
            raise LLMError(resp.status_code, resp.text[:300])
        return resp.json()

    def generate(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        """Call Gemini. Retries 429/5xx with backoff, then tries the fallback model."""
        payload = self.build_payload(system, messages, tools, temperature)
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        last_error = LLMError(503, "no attempt made")
        for model in models:
            for attempt in range(self.max_retries + 1):
                try:
                    return parse_gemini_response(self._post(model, payload))
                except LLMError as exc:
                    last_error = exc
                    if exc.status not in RETRYABLE_STATUS:
                        raise
                    if attempt < self.max_retries:
                        self.sleep(2 ** attempt)   # 1s, 2s, ...
        raise last_error


# --------------------------------------------------------------------------- cassettes
def cassette_key(system: str, messages: list[dict], tools: list[dict] | None, model: str) -> str:
    """sha256 of the canonical JSON of (system, messages, tools, model)."""
    canonical = json.dumps({"system": system, "messages": messages, "tools": tools,
                            "model": model}, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def response_from_dict(data: dict) -> LLMResponse:
    """Rebuild an LLMResponse saved with dataclasses.asdict."""
    return LLMResponse(text=data.get("text", ""),
                       tool_calls=[ToolCall(**tc) for tc in data.get("tool_calls", [])],
                       tokens_in=data.get("tokens_in", 0), tokens_out=data.get("tokens_out", 0))


class CassetteLLM:
    """Record/replay wrapper. Record: call `inner` and save. Replay: load or raise CassetteMiss."""

    def __init__(self, mode: str, inner: LLMClient | None = None,
                 cassette_dir: str | Path | None = None, model: str | None = None):
        if mode not in ("record", "replay"):
            raise ValueError("mode must be 'record' or 'replay'")
        if mode == "record" and inner is None:
            raise ValueError("record mode needs a real client")
        self.mode = mode
        self.inner = inner
        self.dir = Path(cassette_dir or os.getenv("CASSETTE_DIR") or ROOT / "cassettes")
        self.model = model or getattr(inner, "model", None) or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL

    def path_for(self, system: str, messages: list[dict], tools: list[dict] | None) -> Path:
        """Where the cassette for this request lives."""
        return self.dir / f"{cassette_key(system, messages, tools, self.model)}.json"

    def generate(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        """Replay a saved response, or record a new one."""
        path = self.path_for(system, messages, tools)
        if self.mode == "replay":
            if not path.exists():
                raise CassetteMiss(path.stem)
            return response_from_dict(json.loads(path.read_text(encoding="utf-8"))["response"])
        response = self.inner.generate(system, messages, tools, temperature)
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "request": {"system": system, "messages": messages,
                                                   "tools": tools},
                  "response": asdict(response)}
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return response


# --------------------------------------------------------------------------- fake
ScriptItem = Union[LLMResponse, str, Exception]
Responder = Callable[[str, list[dict], Union[list[dict], None]], LLMResponse]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def default_fake_responder(system: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    """A tiny rule-based 'model' so the API/UI work offline with no script.

    * after a tool result -> confirms what the tool returned
    * "speed N" + fan words -> calls set_fan_speed on fan-living (or fan-bedroom)
    * otherwise -> quotes the first retrieved manual chunk from the system prompt
    """
    last = messages[-1] if messages else {"role": "user", "content": ""}
    tokens_in = _estimate_tokens(system + json.dumps(messages))
    if last["role"] == "tool":
        return LLMResponse(text=f"Done. {last.get('name', 'tool')} result: {last['content']}",
                           tokens_in=tokens_in, tokens_out=20)
    text = last.get("content", "").lower()
    speed = re.search(r"speed\s*(?:to\s*)?(\d)", text)
    if tools and speed and ("fan" in text or "pankha" in text):
        device = "fan-bedroom" if "bedroom" in text else "fan-living"
        return LLMResponse(text="", tool_calls=[ToolCall("set_fan_speed",
                                                         {"device_id": device, "speed": int(speed.group(1))})],
                           tokens_in=tokens_in, tokens_out=10)
    match = re.search(r"\[([a-z]+/[a-z0-9#-]+)\][^\n]*\n(.+?)(?:\n\n|\Z)", system, re.S)
    if match:
        answer = f"According to the manual: {match.group(2).strip()[:400]}"
    else:
        answer = "I don't know based on the Hawa manuals. I can raise a service ticket for you."
    return LLMResponse(text=answer, tokens_in=tokens_in, tokens_out=_estimate_tokens(answer))


class FakeLLM:
    """Scripted LLM for tests.

    `script` can be a list (LLMResponse, plain str, or an Exception to raise), consumed one per
    `generate` call, or a callable `(system, messages, tools) -> LLMResponse`. When the list is
    empty, `default_fake_responder` is used. Every call is saved in `self.calls`.
    """

    def __init__(self, script: list[ScriptItem] | Responder | None = None):
        self.calls: list[dict] = []
        self.responder: Responder | None = script if callable(script) else None
        self.queue: list[ScriptItem] = [] if callable(script) else list(script or [])

    def generate(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        """Return the next scripted response."""
        self.calls.append({"system": system, "messages": json.loads(json.dumps(messages)),
                           "tools": tools, "temperature": temperature})
        if self.responder:
            return self.responder(system, messages, tools)
        if not self.queue:
            return default_fake_responder(system, messages, tools)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return LLMResponse(text=item, tokens_in=_estimate_tokens(system), tokens_out=_estimate_tokens(item))
        return item


# --------------------------------------------------------------------------- factory
def current_mode(mode: str | None = None) -> str:
    """Resolve the LLM mode: explicit arg, else env LLM_MODE, else 'fake' (never spends quota)."""
    return (mode or os.getenv("LLM_MODE") or "fake").strip().lower()


def get_llm(mode: str | None = None) -> LLMClient:
    """Build the LLM client for live | record | replay | fake."""
    mode = current_mode(mode)
    if mode == "live":
        return GeminiClient()
    if mode == "record":
        return CassetteLLM("record", inner=GeminiClient())
    if mode == "replay":
        return CassetteLLM("replay")
    if mode == "fake":
        return FakeLLM()
    raise ValueError(f"unknown LLM_MODE {mode!r}; use live|record|replay|fake")
