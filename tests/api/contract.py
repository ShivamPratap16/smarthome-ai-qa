"""The API contract, written down as code.

Two independent checks for every response body:

1. `OpenApiValidator` - validates against the schema the app PUBLISHES at /openapi.json
   (what FastAPI generates; what a Swagger UI / client generator would rely on).
2. Pydantic models below - the STRICT contract from SPEC.md (Contract 2 + 3). They are
   stricter than the published schema (exact field names, types, no extra fields).

If (1) passes but (2) fails, the code broke the contract. If both pass but they disagree
with each other, the published docs are too loose (see docs/api-web-load.md).
"""

from __future__ import annotations

from typing import Any, Literal

import jsonschema_rs
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------- strict contract (SPEC)
class Strict(BaseModel):
    """Base: unknown fields are an error, so new/renamed fields are caught."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ToolResultModel(Strict):
    name: str
    args: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None


class BotReplyModel(Strict):
    text: str = Field(min_length=1)
    tool_calls: list[ToolResultModel]
    contexts: list[str]
    context_ids: list[str]
    refused: bool
    blocked_by: Literal["prompt_injection", "blocked_topic", "pii"] | None
    latency_ms: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    prompt_version: str


class HealthModel(Strict):
    status: Literal["ok"]
    llm_mode: Literal["live", "record", "replay", "fake"]
    prompt_version: str


class DeviceModel(BaseModel):
    """Devices have type-specific fields (speed / battery_pct / filter_life_pct), so extras are allowed."""

    model_config = ConfigDict(extra="allow")
    device_id: str
    name: str
    room: str
    product: str
    type: Literal["fan", "lock", "purifier"]
    online: bool
    serial: str


class OkModel(Strict):
    ok: Literal[True]


class LLMUnavailableModel(Strict):
    error: Literal["llm_unavailable"]


class ValidationErrorModel(BaseModel):
    """FastAPI's 422 body: {"detail": [{"loc": [...], "msg": "...", "type": "..."}]}"""

    detail: list[dict[str, Any]] = Field(min_length=1)


# ---------------------------------------------------------------- published OpenAPI schema
class OpenApiValidator:
    """Validate a response body against the response schema in the app's OpenAPI document."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec

    def response_schema(self, path: str, method: str, status: int) -> dict[str, Any]:
        """The JSON schema documented for (path, method, status). KeyError if undocumented."""
        responses = self.spec["paths"][path][method.lower()]["responses"]
        entry = responses.get(str(status)) or responses["default"]
        return entry["content"]["application/json"]["schema"]

    def validate(self, path: str, method: str, status: int, body: Any) -> None:
        """Raise jsonschema_rs.ValidationError if `body` does not match the documented schema."""
        schema = dict(self.response_schema(path, method, status))
        # "$ref": "#/components/schemas/X" is resolved against this root document.
        schema["components"] = self.spec.get("components", {})
        jsonschema_rs.Draft202012Validator(schema).validate(body)

    def is_documented(self, path: str, method: str, status: int) -> bool:
        """True if the spec lists this status code for the operation."""
        responses = self.spec["paths"].get(path, {}).get(method.lower(), {}).get("responses", {})
        return str(status) in responses
