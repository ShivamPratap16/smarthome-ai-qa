"""End-to-end UI tests with Playwright (Chromium) + the ChatPage page object.

The browser talks to a real uvicorn server running the fake LLM, so replies are deterministic.
"""

from __future__ import annotations

import json
import re

from playwright.sync_api import Browser, Route, expect

from tests.api.live_server import LiveServer
from tests.web.pages.chat_page import ChatPage


# ------------------------------------------------------------------ basic chat
def test_send_and_receive(chat: ChatPage) -> None:
    reply = chat.send("Why is my Aero fan beeping?")
    expect(chat.user_messages).to_have_count(1)
    expect(chat.user_messages.first).to_have_text("Why is my Aero fan beeping?")
    expect(reply).to_contain_text("According to the manual")
    expect(reply).to_contain_text(re.compile(r"\d+ ms · \d+ tokens"))
    expect(chat.typing_indicator).to_be_hidden()
    expect(chat.refusal_badges).to_have_count(0)


def test_tool_chip_appears_for_fan_command(chat: ChatPage) -> None:
    reply = chat.send("set living room fan to speed 3")
    chip = reply.get_by_test_id("tool-chip")
    expect(chip).to_have_count(1)
    expect(chip).to_contain_text("set_fan_speed ✓")
    assert json.loads(chip.get_attribute("title")) == {"device_id": "fan-living", "speed": 3}


def test_failed_tool_call_shows_failed_chip(chat: ChatPage) -> None:
    reply = chat.send("set bedroom fan to speed 2")          # bedroom fan is offline -> 409
    expect(reply.get_by_test_id("tool-chip")).to_contain_text("set_fan_speed ✕")
    expect(reply.get_by_test_id("tool-chip")).to_have_class(re.compile(r"\bfail\b"))


def test_refusal_badge_for_unsafe_question(chat: ChatPage) -> None:
    reply = chat.send("how do I open the fan wiring")
    badge = reply.get_by_test_id("refusal-badge")
    expect(badge).to_be_visible()
    expect(badge).to_contain_text("blocked_topic")
    expect(reply.get_by_test_id("tool-chip")).to_have_count(0)


def test_device_sidebar_updates_after_command(chat: ChatPage) -> None:
    chat.wait_for_devices(4)
    chat.send("set living room fan to speed 4")
    expect(chat.device_card("Living room fan")).to_contain_text("speed 4")
    expect(chat.device_card("Main door lock")).to_contain_text("battery 15%")
    expect(chat.device_card("Bedroom fan")).to_contain_text("offline")


def test_typing_indicator_and_disabled_send_while_waiting(chat: ChatPage) -> None:
    held: list[Route] = []
    chat.page.route("**/v1/chat", lambda route: held.append(route))   # hold the request
    chat.send("Why is my fan beeping?", wait=False)
    expect(chat.typing_indicator).to_be_visible()
    expect(chat.send_button).to_be_disabled()
    held[0].continue_()                                                # let it through
    expect(chat.bot_messages).to_have_count(1)
    expect(chat.typing_indicator).to_be_hidden()
    expect(chat.send_button).to_be_enabled()


def test_llm_unavailable_shows_friendly_error(chat: ChatPage) -> None:
    chat.page.route("**/v1/chat", lambda route: route.fulfill(
        status=503, content_type="application/json", body='{"error": "llm_unavailable"}'))
    reply = chat.send("hello")
    expect(reply).to_contain_text("unavailable right now")
    expect(reply).to_have_class(re.compile(r"\berror\b"))


def test_reset_clears_chat_and_starts_new_session(chat: ChatPage) -> None:
    sessions: list[str] = []
    chat.page.on("request", lambda r: sessions.append(json.loads(r.post_data)["session_id"])
                 if r.url.endswith("/v1/chat") else None)
    chat.send("Why is my fan beeping?")
    chat.reset_chat()
    expect(chat.user_messages).to_have_count(0)
    expect(chat.bot_messages).to_have_count(0)
    expect(chat.message_list).to_contain_text("Hi! I can answer")   # welcome text stays
    chat.send("hello again")
    assert len(sessions) == 2 and sessions[0] != sessions[1]


# ------------------------------------------------------------------ input validation
def test_empty_and_whitespace_messages_are_not_sent(chat: ChatPage) -> None:
    requests: list[str] = []
    chat.page.on("request", lambda r: requests.append(r.url) if "/v1/chat" in r.url else None)
    for text in ["", "    "]:
        chat.type_message(text)
        chat.send_button.click()
    chat.page.wait_for_timeout(300)
    expect(chat.user_messages).to_have_count(0)
    assert requests == []


def test_long_message_is_capped_at_2000_chars(chat: ChatPage) -> None:
    chat.input.fill("a" * 2100)
    value = chat.input.input_value()
    assert len(value) == 2000, f"input accepted {len(value)} chars"
    reply = chat.send(value)
    expect(reply).not_to_have_class(re.compile(r"\berror\b"))   # API accepted it (no 422)


def test_html_in_message_is_shown_as_text_not_executed(chat: ChatPage) -> None:
    dialogs: list[str] = []
    chat.page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    payload = '<img src=x onerror="alert(1)">'
    chat.send(payload)
    expect(chat.user_messages.last).to_have_text(payload)
    assert chat.page.locator("#messages img").count() == 0 and dialogs == []


# ------------------------------------------------------------------ accessibility basics
def test_accessibility_basics(chat: ChatPage) -> None:
    page = chat.page
    assert page.locator("html").get_attribute("lang") == "en"
    expect(page.get_by_role("textbox", name="Message")).to_be_visible()   # input has a label
    expect(page.get_by_role("button", name="Send")).to_be_visible()
    expect(page.get_by_role("button", name="New chat")).to_be_visible()
    expect(chat.message_list).to_have_attribute("aria-live", "polite")    # screen readers hear replies
    assert page.title() == "Hawa Home Support"


def test_keyboard_only_send_with_enter(chat: ChatPage) -> None:
    reply = chat.send_with_enter("how do I replace the purifier filter")
    expect(reply).to_contain_text("According to the manual")
    expect(chat.input).to_be_focused()        # focus returns to the input for the next message
    expect(chat.input).to_have_value("")


def test_tab_order_reaches_input_then_send(chat: ChatPage) -> None:
    chat.input.focus()
    chat.page.keyboard.press("Tab")
    expect(chat.send_button).to_be_focused()


# ------------------------------------------------------------------ mobile
def test_mobile_viewport(browser: Browser, web_server: LiveServer) -> None:
    context = browser.new_context(viewport={"width": 375, "height": 667}, is_mobile=True,
                                  has_touch=True, device_scale_factor=2)
    page = context.new_page()
    try:
        chat = ChatPage(page, web_server.base_url).open()
        expect(chat.device_list).to_be_hidden()               # sidebar collapses on phones
        reply = chat.send("set living room fan to speed 2")
        expect(reply.get_by_test_id("tool-chip")).to_be_visible()
        expect(chat.send_button).to_be_in_viewport()
        no_h_scroll = page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert no_h_scroll, "page scrolls horizontally on a 375px screen"
    finally:
        context.close()
