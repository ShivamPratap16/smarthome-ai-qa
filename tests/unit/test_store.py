"""Chat store writes masked data."""

from assistant.store import ChatStore


def test_messages_and_tool_calls_are_masked(tmp_path) -> None:
    store = ChatStore(tmp_path / "logs.db")
    store.ensure_conversation("c1", "demo-user")
    store.ensure_conversation("c1", "demo-user")   # idempotent
    msg_id = store.add_message("c1", "user", "my number is +91 9876543210, mail a@b.com",
                               prompt_version="v2", refused=False)
    store.add_tool_call(msg_id, "create_service_ticket", {"phone": "9876543210"},
                        {"ticket_id": "HW-100001"}, None)
    store.add_ticket("HW-100001", "fan-living", "noise", "9876543210")

    assert store.query("SELECT COUNT(*) AS n FROM conversations")[0]["n"] == 1
    row = store.messages("c1")[0]
    assert row["content"] == "my number is [PHONE], mail [EMAIL]"
    assert row["prompt_version"] == "v2" and row["refused"] == 0
    tool = store.query("SELECT * FROM tool_calls")[0]
    assert "9876543210" not in tool["args_json"]
    assert store.query("SELECT phone_masked FROM tickets")[0]["phone_masked"] == "[PHONE]"
    store.close()


def test_bot_logs_both_sides(make_bot, store) -> None:
    bot = make_bot(["Hi! Call 9123456789 for help."])
    bot.chat("sess-9", "hello, I am at 9876543210")
    rows = store.messages("sess-9")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert all("[PHONE]" in r["content"] for r in rows)
    assert rows[1]["tokens_in"] > 0 and rows[1]["cost_usd"] > 0
