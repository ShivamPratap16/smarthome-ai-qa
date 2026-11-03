"""Failure recovery: device 503s, retries, the tool-round cap and history consistency."""

from __future__ import annotations

import pytest

from assistant.bot import (
    DEVICE_DOWN_MESSAGE,
    MAX_TOOL_ROUNDS,
    TICKET_DOWN_MESSAGE,
    TOO_MANY_ROUNDS_MESSAGE,
)
from assistant.devices import DeviceCloud
from tests.conversation.framework import Conversation, call, calls, say


def test_single_503_is_retried_and_recovers() -> None:
    devices = DeviceCloud(fail_next={"set_fan_speed": [503]})
    (Conversation(devices=devices)
        .user("living room fan to 3",
              model=[call("set_fan_speed", device_id="fan-living", speed=3), say("Done, speed 3.")])
        .expect_tool_count(1)                      # retry is invisible: one logical call
        .expect(lambda r: r.tool_calls[0].error is None, "retry succeeded")
        .expect_device("fan-living", speed=3)
        .expect_text("Done, speed 3."))
    assert devices.calls.count(("set_fan_speed", {"device_id": "fan-living", "speed": 3})) == 2


def test_double_503_gives_fixed_apology_and_offers_ticket() -> None:
    devices = DeviceCloud(fail_next={"set_fan_speed": [503, 503]})
    (Conversation(devices=devices)
        .user("living room fan to 3", model=[call("set_fan_speed", device_id="fan-living", speed=3)])
        .expect_tool_error("set_fan_speed", 503)
        .expect_text(DEVICE_DOWN_MESSAGE)
        .expect_model_calls(1)                     # no second LLM call: code answers directly
        .expect_device("fan-living", speed=2))


def test_ticket_double_503_gives_ticket_down_message() -> None:
    devices = DeviceCloud(fail_next={"create_service_ticket": [503, 503]})
    (Conversation(devices=devices)
        .user("ticket for fan-living, noise, 9876543210",
              model=[call("create_service_ticket", device_id="fan-living", issue="noise", phone="9876543210")])
        .expect_text(TICKET_DOWN_MESSAGE)
        .expect_no_pii())


def test_user_can_retry_after_device_outage() -> None:
    devices = DeviceCloud(fail_next={"set_fan_speed": [503, 503]})
    (Conversation(devices=devices)
        .user("fan-living to 4", model=[call("set_fan_speed", device_id="fan-living", speed=4)])
        .expect_text(DEVICE_DOWN_MESSAGE)
        .user("try again please",
              model=[call("set_fan_speed", device_id="fan-living", speed=4), say("Done, speed 4 now.")])
        .expect_model_saw(DEVICE_DOWN_MESSAGE)     # the model knows the last attempt failed
        .expect_device("fan-living", speed=4))


def test_non_503_errors_are_not_retried() -> None:
    devices = DeviceCloud(fail_next={"get_device_status": [500]})
    (Conversation(devices=devices)
        .user("status of fan-living",
              model=[call("get_device_status", device_id="fan-living"), say("Something went wrong.")])
        .expect_tool_error("get_device_status", 500))
    assert len(devices.calls) == 1


def test_max_four_tool_rounds_then_polite_stop() -> None:
    loop = [call("get_device_status", device_id="lock-main")] * (MAX_TOOL_ROUNDS + 1)
    (Conversation()
        .user("keep checking my lock", model=loop)
        .expect_tool_count(MAX_TOOL_ROUNDS)
        .expect_model_calls(MAX_TOOL_ROUNDS + 1)
        .expect_text(TOO_MANY_ROUNDS_MESSAGE))


def test_four_rounds_then_answer_is_allowed() -> None:
    script = [call("get_device_status", device_id="lock-main")] * MAX_TOOL_ROUNDS + [say("Battery is 15%.")]
    (Conversation()
        .user("check my lock a few times", model=script)
        .expect_tool_count(MAX_TOOL_ROUNDS)
        .expect_text("Battery is 15%."))


def test_empty_model_answer_falls_back_to_i_dont_know() -> None:
    (Conversation()
        .user("something odd", model=[say("")])
        .expect_text_contains("I don't know")
        .expect_text_contains("service ticket"))


def unanswered_tool_calls(messages: list[dict]) -> list[str]:
    """Tool calls in the history that never got a tool result (Gemini rejects such histories)."""
    missing = []
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            answered = []
            for nxt in messages[i + 1:]:
                if nxt["role"] != "tool":
                    break
                answered.append(nxt.get("name"))
            for tc in msg["tool_calls"]:
                if tc["name"] in answered:
                    answered.remove(tc["name"])
                else:
                    missing.append(tc["name"])
    return missing


def test_history_is_consistent_after_normal_tool_turn() -> None:
    conv = (Conversation()
            .user("fan-living to 3", model=[call("set_fan_speed", device_id="fan-living", speed=3), say("ok")])
            .user("thanks", model=[say("You're welcome!")]))
    assert unanswered_tool_calls(conv.model_calls_this_turn()[0]["messages"]) == []


def test_history_is_consistent_after_503_in_parallel_calls() -> None:
    devices = DeviceCloud(fail_next={"set_fan_speed": [503, 503]})
    conv = (Conversation(devices=devices)
            .user("set living fan to 3 and tell me the lock battery",
                  model=[calls(("set_fan_speed", {"device_id": "fan-living", "speed": 3}),
                               ("get_device_status", {"device_id": "lock-main"}))])
            .expect_text(DEVICE_DOWN_MESSAGE)
            .user("ok, just the lock battery then", model=[say("Battery is 15%.")]))
    assert unanswered_tool_calls(conv.model_calls_this_turn()[0]["messages"]) == []
