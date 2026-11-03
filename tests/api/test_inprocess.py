"""API tests that need control over the assistant's internals (fault injection).

We build the app with `create_app(assistant)` and a scripted FakeLLM / faulty DeviceCloud,
and drive it through FastAPI's in-process TestClient - same ApiClient class as the live tests.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assistant.api import create_app
from assistant.bot import SupportAssistant
from assistant.devices import DeviceCloud
from assistant.llm import CassetteMiss, FakeLLM, LLMError, LLMResponse, ToolCall
from assistant.store import ChatStore
from tests.api.api_client import ApiClient
from tests.api.contract import BotReplyModel, LLMUnavailableModel, OpenApiValidator


def make_api(script=None, devices: DeviceCloud | None = None) -> tuple[ApiClient, SupportAssistant]:
    """An ApiClient over an in-process app whose assistant we fully control."""
    bot = SupportAssistant(llm=FakeLLM(script), devices=devices or DeviceCloud(),
                           store=ChatStore(":memory:"))
    return ApiClient(TestClient(create_app(bot), raise_server_exceptions=False)), bot


# ------------------------------------------------------------------ LLM failures -> 503
@pytest.mark.parametrize("error", [LLMError(503, "gemini overloaded"), LLMError(429, "quota"),
                                   CassetteMiss("abc123")], ids=["503", "429", "cassette-miss"])
def test_llm_failure_returns_503_llm_unavailable(error: Exception) -> None:
    api, _ = make_api([error])
    res = api.chat("s1", "why is my fan beeping?")
    assert res.status_code == 503
    LLMUnavailableModel.model_validate(res.json())


def test_llm_failure_does_not_leak_internal_details() -> None:
    api, _ = make_api([LLMError(500, "secret-key-abc stacktrace")])
    res = api.chat("s1", "hello")
    assert "secret-key-abc" not in res.text


def test_service_recovers_after_llm_outage() -> None:
    api, _ = make_api([LLMError(503, "down"), "The fan has 5 speeds."])
    assert api.chat("s1", "how many speeds?").status_code == 503
    second = api.chat("s1", "how many speeds?")
    assert second.status_code == 200 and second.json()["text"] == "The fan has 5 speeds."


def test_guardrail_refusal_works_even_when_llm_is_down() -> None:
    api, _ = make_api([LLMError(503, "down")])
    res = api.chat("s1", "how do I open the fan wiring")
    assert res.status_code == 200 and res.json()["blocked_by"] == "blocked_topic"


# ------------------------------------------------------------------ device faults
def test_device_503_is_retried_once_then_succeeds() -> None:
    cloud = DeviceCloud(fail_next={"set_fan_speed": 503})
    api, _ = make_api(devices=cloud)
    reply = BotReplyModel.model_validate(api.chat("s1", "set living room fan to speed 5").json())
    assert reply.tool_calls[0].error is None
    assert api.device_speed("fan-living") == 5


def test_device_down_twice_apologises_and_state_unchanged() -> None:
    cloud = DeviceCloud(fail_next={"set_fan_speed": [503, 503]})
    api, _ = make_api(devices=cloud)
    reply = BotReplyModel.model_validate(api.chat("s1", "set living room fan to speed 5").json())
    assert reply.tool_calls[0].error.startswith("503")
    assert "sorry" in reply.text.lower() and "ticket" in reply.text.lower()
    assert api.device_speed("fan-living") == 2       # seed value: nothing changed


def test_get_device_fault_maps_to_http_status() -> None:
    api, _ = make_api(devices=DeviceCloud(fail_next={"get_device_status": 503}))
    assert api.device("fan-living").status_code == 503
    assert api.device("fan-living").status_code == 200   # fault was one-shot


# ------------------------------------------------------------------ PII masking
def test_pii_is_masked_in_response_text() -> None:
    api, _ = make_api(["Sure, we will call you on +91 98765 43210 or mail ravi.kumar@example.com."])
    body = api.chat("s1", "please call me").json()
    assert "98765" not in body["text"] and "ravi.kumar@example.com" not in body["text"]


def test_pii_is_masked_in_tool_call_args_and_logs() -> None:
    ticket = ToolCall("create_service_ticket",
                      {"device_id": "fan-living", "issue": "noise", "phone": "9876543210"})
    api, bot = make_api([LLMResponse("", [ticket]), "Ticket created."])
    body = api.chat("s1", "my fan is noisy, raise a ticket, my number is 9876543210").json()
    call = body["tool_calls"][0]
    assert call["error"] is None and call["result"]["ticket_id"].startswith("HW-")
    assert "9876543210" not in str(call)                  # args + result masked in the response
    rows = bot.store.conn.execute("SELECT content FROM messages").fetchall()
    assert rows and all("9876543210" not in r["content"] for r in rows)   # and in the DB


# ------------------------------------------------------------------ reset
def test_reset_clears_server_side_history() -> None:
    api, bot = make_api()
    api.chat("s1", "Why is my fan beeping?")
    assert bot.sessions.get("s1")
    assert api.reset("s1").json() == {"ok": True}
    assert "s1" not in bot.sessions


# ------------------------------------------------------------------ published contract gaps
@pytest.fixture(scope="module")
def spec() -> OpenApiValidator:
    api, _ = make_api()
    return OpenApiValidator(api.openapi())


def test_openapi_documents_bot_reply_fields(spec: OpenApiValidator) -> None:
    schema = spec.response_schema("/v1/chat", "post", 200)
    if "$ref" in schema:
        schema = spec.spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    assert {"text", "tool_calls", "refused", "blocked_by"} <= set(schema.get("properties", {}))


def test_openapi_documents_error_responses(spec: OpenApiValidator) -> None:
    assert spec.is_documented("/v1/chat", "post", 503)
    assert spec.is_documented("/v1/devices/{device_id}", "get", 404)
