"""Page Object for the Hawa Home chat UI (web/index.html).

Tests talk to `ChatPage` ("send a message", "last bot reply") instead of raw selectors.
If the HTML changes, only this file changes - not 15 tests.
All locators use the stable `data-testid` attributes from SPEC Contract 3.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

BOT_REPLY_TIMEOUT_MS = 15_000


class ChatPage:
    """Everything a test can see or do on the chat screen."""

    def __init__(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url.rstrip("/")
        # ---- locators (lazy: evaluated when used) ----
        self.input: Locator = page.get_by_test_id("chat-input")
        self.send_button: Locator = page.get_by_test_id("send-button")
        self.message_list: Locator = page.get_by_test_id("message-list")
        self.user_messages: Locator = page.get_by_test_id("message-user")
        self.bot_messages: Locator = page.get_by_test_id("message-bot")
        self.tool_chips: Locator = page.get_by_test_id("tool-chip")
        self.refusal_badges: Locator = page.get_by_test_id("refusal-badge")
        self.typing_indicator: Locator = page.get_by_test_id("typing-indicator")
        self.device_list: Locator = page.get_by_test_id("device-list")
        self.device_cards: Locator = page.get_by_test_id("device-card")
        self.reset_button: Locator = page.get_by_test_id("reset-button")

    # ---- navigation ----------------------------------------------------------
    def open(self) -> ChatPage:
        """Load the UI and wait until the device sidebar has been filled."""
        self.page.goto(self.base_url + "/")
        expect(self.input).to_be_visible()
        return self

    def wait_for_devices(self, count: int = 4) -> None:
        """The sidebar loads devices with fetch(); wait for all cards."""
        expect(self.device_cards).to_have_count(count)

    # ---- actions -------------------------------------------------------------
    def type_message(self, text: str) -> None:
        """Type into the input without sending."""
        self.input.fill(text)

    def send(self, text: str, wait: bool = True) -> Locator:
        """Type `text`, click Send and (by default) wait for the new bot reply. Returns that reply."""
        before = self.bot_messages.count()
        self.input.fill(text)
        self.send_button.click()
        reply = self.bot_messages.nth(before)
        if wait:
            expect(reply).to_be_visible(timeout=BOT_REPLY_TIMEOUT_MS)
        return reply

    def send_with_enter(self, text: str) -> Locator:
        """Keyboard-only sending: focus input, type, press Enter."""
        before = self.bot_messages.count()
        self.input.focus()
        self.page.keyboard.type(text)
        self.page.keyboard.press("Enter")
        reply = self.bot_messages.nth(before)
        expect(reply).to_be_visible(timeout=BOT_REPLY_TIMEOUT_MS)
        return reply

    def reset_chat(self) -> None:
        """Click 'New chat' and wait for the reset request to finish."""
        with self.page.expect_response(lambda r: "/reset" in r.url):
            self.reset_button.click()

    # ---- queries -------------------------------------------------------------
    def last_bot_text(self) -> str:
        """Text of the newest bot bubble."""
        return self.bot_messages.last.inner_text()

    def device_card(self, name: str) -> Locator:
        """The sidebar card whose title contains `name`, e.g. 'Living room fan'."""
        return self.device_cards.filter(has_text=name)
