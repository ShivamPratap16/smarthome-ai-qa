# The assistant (system under test)

"Hawa Assist" is a customer-support chatbot for a fictional smart-home brand, **Hawa Home**
(Hawa Aero fan, Hawa Lock S1 door lock, Hawa Pure RO purifier). It answers questions from the
product manuals and can control devices through tools. It is the thing our QA system tests.

## How one message is handled

```
user message
   │
   ▼
1. Guardrails (check_input) ──blocked──► friendly refusal (no LLM call), refused=true
   │ allowed
   ▼
2. Retrieval: BM25 search over manuals/*.md → top 4 sections
   │
   ▼
3. LLM (Gemini) with system prompt (rules + devices + manual sections) + tools
   │   ├─ model asks for a tool → we run it on DeviceCloud → send result back → repeat
   │   │   (max 4 tool rounds; a 503 from a device is retried once, then we apologise
   │   │    and offer a service ticket)
   │   └─ model gives final text
   ▼
4. mask_pii on the final text (phones, emails, Aadhaar-like numbers)
   │
   ▼
5. Log user + bot messages, tool calls and tickets in SQLite (already masked)
   │
   ▼
BotReply {text, tool_calls, contexts, context_ids, refused, blocked_by, latency_ms,
          tokens_in, tokens_out, cost_usd, prompt_version}
```

## Files

| File | What it does |
|---|---|
| `manuals/*.md` | The "knowledge base". Each `##` heading is one chunk. `## Troubleshooting: Beeping` in `aero.md` gets id `aero/troubleshooting#beeping` (the `:` becomes `#`). |
| `assistant/retrieval.py` | Splits manuals into chunks and ranks them with BM25 (`rank_bm25`), a classic keyword-search formula. `retrieve(query, k=4)`. |
| `assistant/devices.py` | `DeviceCloud`: a fake in-memory device backend with 4 seeded devices and 4 tools. `fail_next={"set_fan_speed": 503}` injects an error to test recovery. |
| `assistant/guardrails.py` | Regex rules: prompt injection, blocked topics (DIY wiring, bypassing a lock, medical claims about water), sensitive IDs. `mask_pii` hides personal data. |
| `assistant/llm.py` | One interface `generate(system, messages, tools)` with 4 modes (below). Talks to Gemini over plain REST with `httpx`. |
| `assistant/prompts/v1.txt`, `v2.txt` | v1 = naive prompt. v2 = grounded prompt: answer only from context, say "I don't know" + offer a ticket, safety refusals, Hinglish replies. We compare them in evals. |
| `assistant/bot.py` | `SupportAssistant.chat()` — the flow above, plus per-session history in memory. |
| `assistant/store.py` | SQLite chat logs (`conversations`, `messages`, `tool_calls`, `tickets`). |
| `assistant/api.py` | FastAPI app: `POST /v1/chat`, `POST /v1/sessions/{id}/reset`, `GET /v1/devices`, `GET /v1/devices/{id}`, `GET /health`, `GET /` (chat UI). |
| `web/index.html` | Single-file chat UI (vanilla JS) with `data-testid`s for UI tests. |

## LLM modes (`LLM_MODE`)

| Mode | What happens | Used for |
|---|---|---|
| `live` | Real Gemini call | Demo, manual checks |
| `record` | Real call, response saved in `cassettes/<sha256>.json` | One-time recording run |
| `replay` | Load saved response; error `CassetteMiss` if not recorded | CI (free, deterministic) |
| `fake` (default) | `FakeLLM` returns scripted answers | Unit tests |

The cassette file name is a hash of (system prompt, messages, tools, model), so the same
request always maps to the same file. This is why CI needs no API key: the free Gemini quota is
tiny, so we record once and replay forever.

**How tool calling works with Gemini:** we send our tools as `functionDeclarations`. If the model
wants a tool, its reply has a `functionCall` part (name + args). We run the tool, then send the
result back as a `functionResponse` part, and call the model again. Gemini 3 also returns a
`thoughtSignature` with each function call that must be sent back unchanged — we keep it in
`ToolCall.signature`.

Reliability: on HTTP 429/5xx the client retries with backoff (1s, 2s), then switches to a
fallback model. If everything fails the API returns `503 {"error": "llm_unavailable"}`.

## Device tools (seed data for `demo-user`)

| Device | State |
|---|---|
| `fan-living` | Hawa Aero, online, speed 2 |
| `fan-bedroom` | Hawa Aero, offline (set speed → 409) |
| `lock-main` | Hawa Lock S1, battery 15% |
| `purifier-kitchen` | Hawa Pure RO, filter life 8% |

Warranty serials: `HA-AERO-1001` (till 2027-03-31), `HA-LOCK-2001` (expired 2026-01-15),
`HA-PURE-3001` (till 2028-06-30). Tickets get ids `HW-100001`, `HW-100002`, ...

## Run it

```bash
.venv/Scripts/python -m pytest tests/unit -q                     # no API key needed
LLM_MODE=live .venv/Scripts/python -m uvicorn assistant.api:app --port 8000
# open http://localhost:8000
```

Env vars: `LLM_MODE` (default `fake`), `PROMPT_VERSION` (default `v2`), `CHAT_DB`
(default `data/chatlogs.db`), `GEMINI_API_KEY`, `GEMINI_MODEL`, `CASSETTE_DIR` (default `cassettes/`).

## Using it in tests

```python
from assistant.bot import SupportAssistant
from assistant.devices import DeviceCloud
from assistant.llm import FakeLLM, LLMResponse, ToolCall
from assistant.store import ChatStore

bot = SupportAssistant(
    llm=FakeLLM([LLMResponse("", [ToolCall("set_fan_speed", {"device_id": "fan-living", "speed": 3})]),
                 "Done, speed 3."]),
    devices=DeviceCloud(fail_next={"set_fan_speed": 503}),
    store=ChatStore(":memory:"))
reply = bot.chat("s1", "set living room fan to speed 3")
```

## Interview talking points
- **Why RAG?** The model only knows facts we put in the prompt, so answers stay grounded in the
  manual and we can measure faithfulness / hallucination against the retrieved chunks.
- **Why guardrails before the LLM?** Cheap, deterministic, and testable; dangerous requests never
  reach the model. The v2 prompt is a second safety layer.
- **Why mask PII before logging?** Logs are read by many people/tools; phone numbers must not leak.
- **Why record/replay?** Deterministic, free, fast CI; live calls only when we re-record.
- **Why fault injection?** To test that the bot recovers (retry once, then apologise + offer a
  ticket) instead of lying that the fan speed changed.
