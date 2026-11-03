"""Tool selection and argument validation: the right tool, the right args, clean error handling."""

from __future__ import annotations

import json

import pytest

from assistant.devices import TOOLS
from tests.conversation.framework import Conversation, call, say


def test_set_speed_with_exact_args() -> None:
    (Conversation()
        .user("Set the living room fan to speed 3",
              model=[call("set_fan_speed", device_id="fan-living", speed=3), say("Done, speed 3.")])
        .expect_tool_sequence("set_fan_speed")
        .expect_tool("set_fan_speed", device_id="fan-living", speed=3)
        .expect_device("fan-living", speed=3)
        .expect_model_calls(2))


@pytest.mark.parametrize("speed", [0, 6, 7, -1, 2.5, True])
def test_invalid_speed_rejected_and_device_unchanged(speed) -> None:
    (Conversation()
        .user(f"set the living room fan to {speed}",
              model=[call("set_fan_speed", device_id="fan-living", speed=speed),
                     say("Sorry, speed must be between 1 and 5.")])
        .expect_tool_error("set_fan_speed", 400)
        .expect_model_saw("speed must be an integer from 1 to 5")   # the error is fed back to the model
        .expect_device("fan-living", speed=2))


def test_float_speed_that_is_a_whole_number_is_accepted() -> None:
    """Gemini sometimes sends 3.0 for an integer field."""
    (Conversation()
        .user("speed 3 please", model=[call("set_fan_speed", device_id="fan-living", speed=3.0), say("ok")])
        .expect_device("fan-living", speed=3))


def test_offline_device_returns_409_and_bot_explains() -> None:
    (Conversation()
        .user("Set the bedroom fan to 3",
              model=[call("set_fan_speed", device_id="fan-bedroom", speed=3),
                     say("Your bedroom fan is offline. Please check its power and Wi-Fi.")])
        .expect_tool_error("set_fan_speed", 409)
        .expect_text_contains("offline")
        .expect_device("fan-bedroom", online=False, speed=0))


def test_unknown_device_returns_404() -> None:
    (Conversation()
        .user("Set the garage fan to 2",
              model=[call("set_fan_speed", device_id="fan-garage", speed=2),
                     say("I couldn't find a garage fan in your home.")])
        .expect_tool_error("set_fan_speed", 404))


def test_wrong_device_type_returns_400() -> None:
    (Conversation()
        .user("Set my lock to speed 2",
              model=[call("set_fan_speed", device_id="lock-main", speed=2), say("That is not a fan.")])
        .expect_tool_error("set_fan_speed", 400))


def test_missing_argument_is_a_clean_400_not_a_crash() -> None:
    (Conversation()
        .user("speed up the fan", model=[call("set_fan_speed", device_id="fan-living"), say("Which speed?")])
        .expect_tool_error("set_fan_speed", 400)
        .expect_text("Which speed?"))


def test_status_then_action_in_one_turn() -> None:
    (Conversation()
        .user("If the living room fan is on, put it on 5",
              model=[call("get_device_status", device_id="fan-living"),
                     call("set_fan_speed", device_id="fan-living", speed=5),
                     say("It was on 2, now on 5.")])
        .expect_tool_sequence("get_device_status", "set_fan_speed")
        .expect_device("fan-living", speed=5))


@pytest.mark.parametrize("serial, status", [("HA-AERO-1001", "active"), ("ha-lock-2001", "expired")])
def test_warranty_lookup(serial, status) -> None:
    conv = Conversation().user(f"warranty for {serial}?",
                               model=[call("check_warranty", serial=serial), say(f"Warranty is {status}.")])
    conv.expect_tool("check_warranty", serial=serial)
    assert conv.last.tool_calls[0].result["status"] == status


def test_ticket_needs_phone_model_asks_first() -> None:
    (Conversation()
        .user("Raise a ticket, my bedroom fan is not working",
              model=[say("Sure. Please share your 10-digit mobile number for the ticket.")])
        .expect_no_tool("create_service_ticket")
        .expect_text_contains("mobile number")
        .user("9876543210",
              model=[call("create_service_ticket", device_id="fan-bedroom", issue="fan not working",
                          phone="9876543210"),
                     say("Ticket HW-100001 raised. A technician will call your registered number.")])
        .expect_tool("create_service_ticket", device_id="fan-bedroom", phone="[PHONE]")
        .expect_text_contains("HW-100001"))


def test_ticket_with_invalid_phone_is_rejected() -> None:
    (Conversation()
        .user("ticket for fan-living, noise, phone 12345",
              model=[call("create_service_ticket", device_id="fan-living", issue="noise", phone="12345"),
                     say("That number looks wrong, please share a 10-digit mobile number.")])
        .expect_tool_error("create_service_ticket", 400))


def test_ticket_creation_masks_phone_in_logs_and_reply() -> None:
    conv = (Conversation()
            .user("Ticket for fan-living, it wobbles. Number +91 98765 43210",
                  model=[call("create_service_ticket", device_id="fan-living", issue="fan wobbles",
                              phone="+91 98765 43210"),
                         say("Done! We will call +91 98765 43210.")])
            .expect_tool("create_service_ticket", phone="[PHONE]")
            .expect_text_not_contains("98765")
            .expect_no_pii())
    ticket = conv.store.query("SELECT * FROM tickets")[0]
    assert ticket["phone_masked"] == "[PHONE]" and ticket["id"] == "HW-100001"
    args = json.loads(conv.store.query("SELECT args_json FROM tool_calls")[0]["args_json"])
    assert args["phone"] == "[PHONE]"


def test_tool_declarations_match_the_device_api() -> None:
    """The schema the model sees must list the same required args the API needs."""
    by_name = {t["name"]: t for t in TOOLS}
    assert set(by_name) == {"get_device_status", "set_fan_speed", "check_warranty", "create_service_ticket"}
    assert by_name["set_fan_speed"]["parameters"]["required"] == ["device_id", "speed"]
    assert by_name["create_service_ticket"]["parameters"]["required"] == ["device_id", "issue", "phone"]
