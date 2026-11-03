"""A small, reusable client for the Hawa Home Support API (one method per endpoint).

Tests call `api.chat("s1", "hello")` instead of repeating URLs and JSON bodies everywhere.
It wraps any `httpx.Client`, so the SAME class works against:

* a live uvicorn server:   ApiClient.for_url("http://127.0.0.1:8000")
* an in-process app:       ApiClient(TestClient(create_app(bot)))   (TestClient is an httpx.Client)
"""

from __future__ import annotations

from typing import Any

import httpx


class ApiClient:
    """Thin wrapper: builds requests, returns the raw `httpx.Response` (tests assert on it)."""

    def __init__(self, http: httpx.Client):
        self.http = http

    @classmethod
    def for_url(cls, base_url: str, timeout: float = 30.0) -> ApiClient:
        """Client for a server reachable over the network."""
        return cls(httpx.Client(base_url=base_url, timeout=timeout))

    # ---- endpoints ---------------------------------------------------------
    def health(self) -> httpx.Response:
        """GET /health"""
        return self.http.get("/health")

    def openapi(self) -> dict[str, Any]:
        """GET /openapi.json (the app's own Swagger/OpenAPI document)."""
        res = self.http.get("/openapi.json")
        res.raise_for_status()
        return res.json()

    def chat(self, session_id: str, message: str) -> httpx.Response:
        """POST /v1/chat"""
        return self.http.post("/v1/chat", json={"session_id": session_id, "message": message})

    def chat_raw(self, body: Any = None, content: str | None = None,
                 headers: dict[str, str] | None = None) -> httpx.Response:
        """POST /v1/chat with any body (for negative tests: missing fields, bad JSON...)."""
        if content is not None:
            return self.http.post("/v1/chat", content=content, headers=headers or {})
        return self.http.post("/v1/chat", json=body)

    def reset(self, session_id: str) -> httpx.Response:
        """POST /v1/sessions/{session_id}/reset"""
        return self.http.post(f"/v1/sessions/{session_id}/reset")

    def devices(self) -> httpx.Response:
        """GET /v1/devices"""
        return self.http.get("/v1/devices")

    def device(self, device_id: str) -> httpx.Response:
        """GET /v1/devices/{device_id}"""
        return self.http.get(f"/v1/devices/{device_id}")

    # ---- helpers -----------------------------------------------------------
    def device_speed(self, device_id: str) -> int:
        """Current fan speed, read through the public API."""
        res = self.device(device_id)
        res.raise_for_status()
        return res.json()["speed"]

    def close(self) -> None:
        """Close the underlying connection pool."""
        self.http.close()
