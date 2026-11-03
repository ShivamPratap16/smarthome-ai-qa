"""HTTP API for the assistant (Contract 3).

Run:  .venv/Scripts/python -m uvicorn assistant.api:app --port 8000
Env:  LLM_MODE (live|record|replay|fake, default fake), PROMPT_VERSION (default v2),
      CHAT_DB (default data/chatlogs.db).
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from assistant.bot import SupportAssistant
from assistant.devices import DeviceAPIError
from assistant.llm import LLMError, current_mode
from assistant.store import ChatStore

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


class ChatRequest(BaseModel):
    """Body of POST /v1/chat."""

    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject whitespace-only messages (422) and strip the rest."""
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class ToolResultModel(BaseModel):
    """One executed tool call (mirrors bot.ToolResult)."""

    name: str
    args: dict
    result: dict | None
    error: str | None


class BotReplyModel(BaseModel):
    """Response of POST /v1/chat (mirrors bot.BotReply, same field names)."""

    text: str
    tool_calls: list[ToolResultModel]
    contexts: list[str]
    context_ids: list[str]
    refused: bool
    blocked_by: str | None
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    prompt_version: str


class ErrorModel(BaseModel):
    """503 body when the LLM is unavailable."""

    error: str


class NotFoundModel(BaseModel):
    """404 body."""

    detail: str


def create_app(assistant: SupportAssistant | None = None) -> FastAPI:
    """Build the FastAPI app. Pass an assistant in tests (e.g. one using FakeLLM)."""
    app = FastAPI(title="Hawa Home Support Assistant", version="1.0.0")
    state: dict[str, SupportAssistant] = {}
    if assistant is not None:
        state["assistant"] = assistant

    def get_assistant() -> SupportAssistant:
        """Create the assistant lazily on first use (so importing the app never calls the LLM)."""
        if "assistant" not in state:
            state["assistant"] = SupportAssistant(
                prompt_version=os.getenv("PROMPT_VERSION", "v2"),
                store=ChatStore(os.getenv("CHAT_DB", "data/chatlogs.db")))
        return state["assistant"]

    app.state.get_assistant = get_assistant

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "llm_unavailable"})

    @app.get("/health")
    def health() -> dict:
        """Liveness check plus the active LLM mode and prompt version."""
        bot = state.get("assistant")
        return {"status": "ok", "llm_mode": current_mode(),
                "prompt_version": bot.prompt_version if bot else os.getenv("PROMPT_VERSION", "v2")}

    @app.post("/v1/chat", response_model=BotReplyModel,
              responses={503: {"model": ErrorModel, "description": "LLM unavailable"}})
    def chat(body: ChatRequest) -> dict:
        """Send one message; returns the BotReply as JSON."""
        return asdict(get_assistant().chat(body.session_id, body.message))

    @app.post("/v1/sessions/{session_id}/reset")
    def reset(session_id: str) -> dict:
        """Clear a session's conversation history."""
        get_assistant().reset(session_id)
        return {"ok": True}

    @app.get("/v1/devices")
    def list_devices() -> list[dict]:
        """All devices of the demo user."""
        return get_assistant().devices.list_devices("demo-user")

    @app.get("/v1/devices/{device_id}",
             responses={404: {"model": NotFoundModel, "description": "Unknown device"}})
    def device_status(device_id: str) -> dict:
        """Status of one device, 404 if unknown."""
        devices = get_assistant().devices
        if device_id not in devices.devices:
            raise HTTPException(status_code=404, detail=f"device '{device_id}' not found")
        try:
            return devices.get_device_status(device_id)
        except DeviceAPIError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.msg) from exc

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """The single-file chat UI."""
        return FileResponse(WEB_INDEX, media_type="text/html")

    return app


app = create_app()
