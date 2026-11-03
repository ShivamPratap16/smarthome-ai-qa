"""Start the real API server (uvicorn) in a background process for tests and load runs.

Used by tests/api, tests/web and load/run_load.py so every suite talks to the app exactly
like a real client would: over HTTP, on a real port.

    with LiveServer() as server:          # fake LLM, temp SQLite db, random free port
        httpx.get(server.base_url + "/health")
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, Self

import httpx

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    """Ask the OS for a free TCP port (bind to port 0, read the port, close)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LiveServer:
    """A uvicorn process running `assistant.api:app` with its own temp database."""

    def __init__(self, llm_mode: str = "fake", prompt_version: str = "v2",
                 port: int | None = None, extra_env: dict[str, str] | None = None,
                 startup_timeout: float = 30.0):
        self.llm_mode = llm_mode
        self.prompt_version = prompt_version
        self.port = port or free_port()
        self.extra_env = extra_env or {}
        self.startup_timeout = startup_timeout
        self.tmpdir: tempfile.TemporaryDirectory | None = None
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self._log_file: IO[str] | None = None

    @property
    def base_url(self) -> str:
        """Root URL of the running server, e.g. http://127.0.0.1:54321."""
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> LiveServer:
        """Launch uvicorn and wait until GET /health answers 200."""
        self.tmpdir = tempfile.TemporaryDirectory(prefix="smarthome-live-", ignore_cleanup_errors=True)
        tmp = Path(self.tmpdir.name)
        env = {**os.environ,
               "LLM_MODE": self.llm_mode,
               "PROMPT_VERSION": self.prompt_version,
               "CHAT_DB": str(tmp / "chatlogs.db"),
               "PYTHONUTF8": "1",
               **self.extra_env}
        if self.llm_mode == "fake":
            env["CASSETTE_DIR"] = str(tmp / "cassettes")   # never touch the real cassettes
        self.log_path = tmp / "server.log"
        self._log_file = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115 (closed in stop())
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "assistant.api:app", "--host", "127.0.0.1",
             "--port", str(self.port), "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=self._log_file, stderr=subprocess.STDOUT)
        self._wait_until_healthy()
        return self

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"server exited early:\n{self.logs()}")
            try:
                if httpx.get(self.base_url + "/health", timeout=1.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        self.stop()
        raise RuntimeError(f"server did not become healthy in {self.startup_timeout}s")

    def logs(self) -> str:
        """Everything uvicorn printed so far (useful when a test fails)."""
        if self.log_path and self.log_path.exists():
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        return ""

    def stop(self) -> None:
        """Terminate the server and delete its temp folder."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        if self.tmpdir:
            self.tmpdir.cleanup()
            self.tmpdir = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
