"""Safety tests run fully offline: fake LLM, no API key, no network."""

from __future__ import annotations

import pytest

from assistant.bot import SupportAssistant
from assistant.devices import DeviceCloud
from assistant.llm import FakeLLM
from assistant.store import ChatStore


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force fake mode and hide the real key so no safety test can spend Gemini quota."""
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def store() -> ChatStore:
    """A throw-away in-memory chat-log database."""
    return ChatStore(":memory:")


@pytest.fixture
def make_bot(store: ChatStore):
    """make_bot(script, devices=None) -> (bot, fake_llm, devices)."""
    def _make(script=None, devices: DeviceCloud | None = None):
        llm = FakeLLM(script)
        devices = devices or DeviceCloud()
        return SupportAssistant(llm=llm, devices=devices, store=store), llm, devices
    return _make
