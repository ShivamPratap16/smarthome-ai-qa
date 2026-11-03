# SmartHome AI QA — build spec (binding contract for all builders)

A quality-engineering system for an LLM customer-support assistant of a **fictional** smart-home brand,
**Hawa Home** (products: *Hawa Aero* BLDC smart ceiling fan, *Hawa Lock S1* smart door lock,
*Hawa Pure RO* water purifier). Never use any real company's name or branding.

The assistant is the *system under test* (SUT). The quality system around it is the real product.

## Ground rules for every builder
- Python 3.13, shared venv at `.venv` (do NOT create another). Run tools as `.venv/Scripts/python -m ...`.
  If you need an extra package: `.venv/Scripts/pip install <pkg>` AND append it to `requirements.txt`.
- Only create/edit files in the directories you own (see "Ownership"). If you need a change in someone
  else's area, describe it in your final report instead of editing.
- Do NOT `git commit`. The coordinator commits.
- Windows: always open text files with `encoding="utf-8"`.
- **Gemini quota is tiny (free tier).** All unit/integration tests and CI must run with NO API key, using
  `LLM_MODE=fake` (scripted) or `LLM_MODE=replay` (cassettes). Make at most ~20 live calls each while
  developing. The coordinator does the one big live recording run at the end.
- Every module: type hints, docstrings on public functions, small and readable (the owner must explain
  it in an interview). No clever metaprogramming.
- README sections for your area go in `docs/<area>.md` (the coordinator writes the top README).

## Repo layout & ownership
| Path | Owner |
|---|---|
| `assistant/`, `manuals/`, `web/`, `tests/unit/` | Agent A (assistant core) |
| `qa/golden/`, `qa/evals/`, `qa/promptfoo/`, `qa/regressions/` | Agent B (evals) |
| `qa/redteam/`, `tests/safety/`, `tests/conversation/` | Agent C (safety + tool-calling) |
| `tests/api/`, `tests/web/`, `load/` | Agent D (API, web, load) |
| `qa/data_checks/`, `qa/aws/`, `tests/data/` | Agent E (data + AWS) |
| `dashboard/`, `.github/workflows/`, `scripts/` | Agent F (dashboard + CI) |
| `mobile/` | later, on Linux (Appium) |

## Contract 1 — LLM layer (`assistant/llm.py`)
```python
@dataclass
class ToolCall: name: str; args: dict
@dataclass
class LLMResponse:
    text: str                       # "" if only tool calls
    tool_calls: list[ToolCall]
    tokens_in: int; tokens_out: int
class LLMClient(Protocol):
    def generate(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 0.0) -> LLMResponse: ...
def get_llm(mode: str | None = None) -> LLMClient   # mode from env LLM_MODE: live|record|replay|fake
```
- `messages` items: `{"role": "user"|"assistant"|"tool", "content": str, "name"?: str, "tool_calls"?: [...]}`.
- `tools` = JSON-schema function declarations `{"name","description","parameters"}`.
- Live = Gemini REST via httpx (`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`),
  key `GEMINI_API_KEY`, model env `GEMINI_MODEL` (default `gemini-3.5-flash-lite`), fallback `gemini-3.1-flash-lite`
  on 429/503. `.env` loaded with python-dotenv.
- Cassettes: `cassettes/<sha256 of canonical JSON of (system, messages, tools, model)>.json`. `record` = live + save;
  `replay` = load, raise `CassetteMiss` if absent. `fake` = `FakeLLM(script)` returning queued responses; usable from tests.
- The same client is reused as the LLM **judge** for evals (so judge calls are cached too).

## Contract 2 — the assistant (`assistant/bot.py`)
```python
@dataclass
class ToolResult: name: str; args: dict; result: dict | None; error: str | None
@dataclass
class BotReply:
    text: str
    tool_calls: list[ToolResult]
    contexts: list[str]        # retrieved manual chunks actually given to the model
    context_ids: list[str]     # chunk ids, e.g. "aero/troubleshooting#beeping"
    refused: bool              # True if guardrail or model declined
    blocked_by: str | None     # "prompt_injection" | "blocked_topic" | "pii" | None
    latency_ms: int; tokens_in: int; tokens_out: int; cost_usd: float
    prompt_version: str
class SupportAssistant:
    def __init__(self, llm: LLMClient | None = None, devices: DeviceCloud | None = None,
                 prompt_version: str = "v2", store: ChatStore | None = None): ...
    def chat(self, session_id: str, user_message: str, user_id: str = "demo-user") -> BotReply: ...
    def reset(self, session_id: str) -> None: ...
```
- Prompts live in `assistant/prompts/v1.txt` (naive, weaker) and `v2.txt` (grounded: answer only from context,
  say "I don't know" + offer ticket, refuse unsafe DIY electrical work, Hinglish-aware). promptfoo compares them.
- Retrieval `assistant/retrieval.py`: `retrieve(query: str, k: int = 4) -> list[Chunk]` with
  `Chunk(id, product, title, text)`; BM25 (`rank_bm25`) over `manuals/*.md` split by `##` headings; id = `<file stem>/<slug of heading>`.
- Tools (`assistant/devices.py`, class `DeviceCloud`, in-memory, seeded, deterministic):
  `get_device_status(device_id)`, `set_fan_speed(device_id, speed:int 1-5)`, `check_warranty(serial)`,
  `create_service_ticket(device_id, issue, phone)`. Errors raise `DeviceAPIError(status, msg)`.
  Fault injection: `DeviceCloud(fail_next={"set_fan_speed": 503})`. Seed for `demo-user`:
  `fan-living` (Aero, online, speed 2), `fan-bedroom` (Aero, offline), `lock-main` (Lock S1, battery 15%),
  `purifier-kitchen` (Pure RO, filter life 8%). Serials `HA-AERO-1001`, `HA-LOCK-2001`, `HA-PURE-3001`
  (warranty expiry 2027-03-31, 2026-01-15 expired, 2028-06-30).
  Bot retries a 503 once, then apologises and offers a ticket. Max 4 tool rounds per turn.
- Guardrails `assistant/guardrails.py`: `check_input(text) -> Verdict(allowed, reason)`
  (prompt injection patterns, blocked topics: opening wiring / bypassing lock / medical claims about water),
  `mask_pii(text) -> str` (Indian mobile `[6-9]\d{9}` with optional +91, emails, 12-digit Aadhaar-like) used
  before logging and on any echo of tool output.
- Store `assistant/store.py`: `ChatStore(path="data/chatlogs.db")` SQLite tables:
  `conversations(id, user_id, started_at)`, `messages(id, conversation_id, role, content, created_at, prompt_version, latency_ms, tokens_in, tokens_out, cost_usd, refused, blocked_by)`,
  `tool_calls(id, message_id, name, args_json, result_json, error)`, `tickets(id, device_id, issue, phone_masked, created_at)`.
  Content is PII-masked BEFORE insert. Cost: Gemini flash-lite price constants in one place.

## Contract 3 — HTTP API (`assistant/api.py`, FastAPI, `uvicorn assistant.api:app --port 8000`)
- `POST /v1/chat` body `{"session_id": str, "message": str (1..2000 chars)}` → BotReply JSON (dataclass asdict).
- `POST /v1/sessions/{session_id}/reset` → `{"ok": true}`
- `GET /v1/devices` → list for demo-user; `GET /v1/devices/{device_id}` → status or 404.
- `GET /health` → `{"status":"ok","llm_mode":...,"prompt_version":...}`
- `GET /` serves `web/index.html`: single-file chat UI. Required `data-testid`s: `chat-input`, `send-button`,
  `message-list`, `message-user`, `message-bot`, `tool-chip` (one per tool call), `refusal-badge`, `typing-indicator`.
- Errors: 422 on validation; 503 JSON `{"error": "llm_unavailable"}` if the LLM fails.

## Contract 4 — golden dataset (`qa/golden/golden.jsonl`, one JSON per line)
```json
{"id":"aero-beep-01","category":"manual_qa|tool_use|multi_turn|hinglish|refusal|unanswerable|recovery",
 "product":"aero|lock|pure|general","turns":["user msg 1","user msg 2"],
 "expected_answer":"reference answer","expected_context_ids":["aero/troubleshooting#beeping"],
 "expected_tools":[{"name":"set_fan_speed","args":{"device_id":"fan-living","speed":3}}],
 "must_refuse":false,"fault":{"set_fan_speed":503},"tags":["smoke"]}
```

## Contract 5 — report files (everything the dashboard reads), in `reports/`
Each run writes `reports/<suite>/<YYYYmmdd-HHMMSS>.json` and copies it to `reports/<suite>/latest.json`:
```json
{"suite":"evals|redteam|api|web|load|data|tool_calling","run_id":"...","started_at":"ISO","git_sha":"...",
 "prompt_version":"v2","model":"...","llm_mode":"replay",
 "summary":{"total":60,"passed":55,"failed":5,"pass_rate":0.9167,
            "metrics":{"faithfulness":0.91,"answer_relevancy":0.88,"hallucination_rate":0.05,
                        "context_precision":0.8,"context_recall":0.85,"p95_latency_ms":1800,"cost_usd":0.004}},
 "cases":[{"id":"...","category":"...","passed":true,"scores":{"faithfulness":0.9},"reason":"...","latency_ms":900}]}
```
`qa/report.py` (owned by Agent B, others import it): `write_report(suite: str, summary: dict, cases: list[dict], **meta) -> Path`.

## Quality gate (`qa/gate.py`, owned by Agent F)
Reads `reports/*/latest.json` + `qa/thresholds.yaml`; fails (exit 1) if e.g. evals pass_rate < 0.85,
hallucination_rate > 0.10, redteam pass_rate < 1.0 for `critical` cases, p95 latency over budget.
Prints a markdown table (also written to `$GITHUB_STEP_SUMMARY` when set).
