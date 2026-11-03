"""Shared fixtures for unit tests. No API key and no network are ever needed."""

from __future__ import annotations

import pytest

from assistant.bot import SupportAssistant
from assistant.devices import DeviceCloud
from assistant.llm import FakeLLM
from assistant.store import ChatStore


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force fake mode and hide the real key so no test can call Gemini by accident."""
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def store() -> ChatStore:
    return ChatStore(":memory:")


@pytest.fixture
def make_bot(store: ChatStore):
    """Factory: make_bot(script, fail_next=None, prompt_version="v2") -> SupportAssistant."""
    def _make(script=None, fail_next=None, prompt_version: str = "v2") -> SupportAssistant:
        return SupportAssistant(llm=FakeLLM(script), devices=DeviceCloud(fail_next=fail_next),
                                prompt_version=prompt_version, store=store)
    return _make
