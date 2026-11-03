"""LLM layer: Gemini wire format, cassette record/replay, FakeLLM, get_llm."""

import pytest

from assistant.llm import (
    CassetteLLM,
    CassetteMiss,
    FakeLLM,
    GeminiClient,
    LLMError,
    LLMResponse,
    ToolCall,
    get_llm,
    parse_gemini_response,
    to_gemini_contents,
)

TOOLS = [{"name": "set_fan_speed", "description": "x",
          "parameters": {"type": "object", "properties": {"speed": {"type": "integer"}}}}]


def test_messages_convert_to_gemini_contents() -> None:
    contents = to_gemini_contents([
        {"role": "user", "content": "fan to 3"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "set_fan_speed", "args": {"speed": 3}, "signature": "sig1"},
            {"name": "get_device_status", "args": {"device_id": "fan-living"}}]},
        {"role": "tool", "name": "set_fan_speed", "content": '{"ok": true}'},
        {"role": "tool", "name": "get_device_status", "content": "not json"},
    ])
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    model_parts = contents[1]["parts"]
    assert model_parts[0] == {"functionCall": {"name": "set_fan_speed", "args": {"speed": 3}},
                              "thoughtSignature": "sig1"}
    assert model_parts[1]["thoughtSignature"] == "skip_thought_signature_validator"
    responses = contents[2]["parts"]   # both tool results merged into one turn
    assert responses[0]["functionResponse"] == {"name": "set_fan_speed", "response": {"ok": True}}
    assert responses[1]["functionResponse"]["response"] == {"result": "not json"}


def test_parse_gemini_response() -> None:
    data = {"candidates": [{"content": {"role": "model", "parts": [
        {"text": "thinking...", "thought": True},
        {"text": "Setting it."},
        {"functionCall": {"name": "set_fan_speed", "args": {"speed": 3}}, "thoughtSignature": "abc"}]}}],
        "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 7, "thoughtsTokenCount": 3}}
    resp = parse_gemini_response(data)
    assert resp.text == "Setting it."
    assert resp.tool_calls == [ToolCall("set_fan_speed", {"speed": 3})]
    assert resp.tool_calls[0].signature == "abc"
    assert (resp.tokens_in, resp.tokens_out) == (50, 10)


def test_gemini_retries_then_falls_back(monkeypatch) -> None:
    client = GeminiClient(api_key="test", model="m1", fallback_model="m2", max_retries=1,
                          sleep=lambda s: None)
    seen = []

    def fake_post(model, payload):
        seen.append(model)
        if model == "m1":
            raise LLMError(429, "quota")
        return {"candidates": [{"content": {"parts": [{"text": "hi"}]}}], "usageMetadata": {}}

    monkeypatch.setattr(client, "_post", fake_post)
    assert client.generate("sys", [{"role": "user", "content": "x"}], TOOLS).text == "hi"
    assert seen == ["m1", "m1", "m2"]


def test_gemini_does_not_retry_400(monkeypatch) -> None:
    client = GeminiClient(api_key="test", model="m1", sleep=lambda s: None)
    calls = []
    monkeypatch.setattr(client, "_post", lambda m, p: calls.append(m) or (_ for _ in ()).throw(LLMError(400, "bad")))
    with pytest.raises(LLMError):
        client.generate("sys", [{"role": "user", "content": "x"}])
    assert calls == ["m1"]


def test_payload_has_function_declarations() -> None:
    payload = GeminiClient(api_key="test").build_payload("sys", [{"role": "user", "content": "x"}], TOOLS, 0.0)
    assert payload["tools"] == [{"functionDeclarations": TOOLS}]
    assert payload["systemInstruction"]["parts"][0]["text"] == "sys"


def test_cassette_record_then_replay(tmp_path) -> None:
    inner = FakeLLM([LLMResponse("", [ToolCall("set_fan_speed", {"speed": 3}, signature="s")], 10, 2)])
    inner.model = "test-model"
    messages = [{"role": "user", "content": "fan to 3"}]
    recorded = CassetteLLM("record", inner=inner, cassette_dir=tmp_path).generate("sys", messages, TOOLS)
    assert len(list(tmp_path.glob("*.json"))) == 1

    replay = CassetteLLM("replay", cassette_dir=tmp_path, model="test-model")
    again = replay.generate("sys", messages, TOOLS)
    assert again == recorded and again.tool_calls[0].signature == "s"
    with pytest.raises(CassetteMiss):
        replay.generate("sys", [{"role": "user", "content": "something else"}], TOOLS)


def test_fake_llm_script_and_callable() -> None:
    fake = FakeLLM(["hello", LLMResponse("bye", [], 1, 1)])
    assert fake.generate("s", [{"role": "user", "content": "a"}]).text == "hello"
    assert fake.generate("s", [{"role": "user", "content": "b"}]).text == "bye"
    assert len(fake.calls) == 2
    fn = FakeLLM(lambda system, messages, tools: LLMResponse(messages[-1]["content"].upper(), [], 0, 0))
    assert fn.generate("s", [{"role": "user", "content": "abc"}]).text == "ABC"


def test_get_llm_modes(monkeypatch) -> None:
    assert isinstance(get_llm("fake"), FakeLLM)
    assert isinstance(get_llm("replay"), CassetteLLM)
    monkeypatch.setenv("LLM_MODE", "fake")
    assert isinstance(get_llm(), FakeLLM)
    with pytest.raises(LLMError):
        get_llm("live")   # no key in unit tests
    with pytest.raises(ValueError):
        get_llm("nope")
