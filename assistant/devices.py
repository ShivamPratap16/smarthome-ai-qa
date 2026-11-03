"""A fake, in-memory "device cloud" that the assistant controls through tools.

It is seeded with the same devices every time (deterministic), so tests can
rely on exact values. Faults can be injected to test error handling:

    DeviceCloud(fail_next={"set_fan_speed": 503})          # next call fails once
    DeviceCloud(fail_next={"set_fan_speed": [503, 503]})   # next two calls fail
"""

from __future__ import annotations

import copy
import os
import re
from datetime import date

from assistant.guardrails import mask_pii


class DeviceAPIError(Exception):
    """Error returned by the device cloud, with an HTTP-like status code."""

    def __init__(self, status: int, msg: str):
        super().__init__(f"{status}: {msg}")
        self.status = status
        self.msg = msg


SEED_DEVICES: dict[str, dict] = {
    "fan-living": {"device_id": "fan-living", "name": "Living room fan", "room": "living room",
                   "product": "Hawa Aero", "type": "fan", "online": True, "speed": 2,
                   "serial": "HA-AERO-1001"},
    "fan-bedroom": {"device_id": "fan-bedroom", "name": "Bedroom fan", "room": "bedroom",
                    "product": "Hawa Aero", "type": "fan", "online": False, "speed": 0,
                    "serial": "HA-AERO-1002"},
    "lock-main": {"device_id": "lock-main", "name": "Main door lock", "room": "main door",
                  "product": "Hawa Lock S1", "type": "lock", "online": True, "locked": True,
                  "battery_pct": 15, "serial": "HA-LOCK-2001"},
    "purifier-kitchen": {"device_id": "purifier-kitchen", "name": "Kitchen purifier",
                         "room": "kitchen", "product": "Hawa Pure RO", "type": "purifier",
                         "online": True, "filter_life_pct": 8, "tds_ppm": 80,
                         "serial": "HA-PURE-3001"},
}

SEED_WARRANTIES: dict[str, dict] = {
    "HA-AERO-1001": {"product": "Hawa Aero", "expiry": "2027-03-31"},
    "HA-LOCK-2001": {"product": "Hawa Lock S1", "expiry": "2026-01-15"},
    "HA-PURE-3001": {"product": "Hawa Pure RO", "expiry": "2028-06-30"},
}

PHONE_RE = re.compile(r"^(\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}$")

# Tool declarations given to the LLM (JSON schema, Contract 1 format).
TOOLS: list[dict] = [
    {"name": "get_device_status",
     "description": "Get the live status of one of the user's devices (online, speed, battery, filter life).",
     "parameters": {"type": "object",
                    "properties": {"device_id": {"type": "string", "description": "e.g. fan-living"}},
                    "required": ["device_id"]}},
    {"name": "set_fan_speed",
     "description": "Set the speed of a Hawa Aero fan. Speed must be an integer from 1 to 5.",
     "parameters": {"type": "object",
                    "properties": {"device_id": {"type": "string", "description": "e.g. fan-living"},
                                   "speed": {"type": "integer", "description": "1 (lowest) to 5 (highest)"}},
                    "required": ["device_id", "speed"]}},
    {"name": "check_warranty",
     "description": "Check warranty status for a product serial number like HA-AERO-1001.",
     "parameters": {"type": "object",
                    "properties": {"serial": {"type": "string"}},
                    "required": ["serial"]}},
    {"name": "create_service_ticket",
     "description": "Raise a service ticket for a technician visit. Needs the device, a short issue "
                    "description and the user's 10-digit Indian mobile number.",
     "parameters": {"type": "object",
                    "properties": {"device_id": {"type": "string"},
                                   "issue": {"type": "string"},
                                   "phone": {"type": "string"}},
                    "required": ["device_id", "issue", "phone"]}},
]


class DeviceCloud:
    """In-memory device cloud for one demo user."""

    def __init__(self, fail_next: dict[str, int | list[int]] | None = None,
                 today: date | None = None):
        self.devices = copy.deepcopy(SEED_DEVICES)
        self.warranties = copy.deepcopy(SEED_WARRANTIES)
        self.tickets: list[dict] = []
        self.today = today
        self.ticket_counter = 100000   # last used ticket number; see reserve_ticket_numbers()
        self.calls: list[tuple[str, dict]] = []   # every tool call, for test assertions
        self.fail_next: dict[str, list[int]] = {}
        for name, status in (fail_next or {}).items():
            self.fail_next[name] = list(status) if isinstance(status, (list, tuple)) else [status]

    # ---- helpers -------------------------------------------------------
    def _maybe_fail(self, name: str) -> None:
        """Raise an injected fault if one is queued for this tool."""
        queue = self.fail_next.get(name)
        if queue:
            status = queue.pop(0)
            raise DeviceAPIError(status, "device cloud temporarily unavailable"
                                 if status >= 500 else "injected error")

    def reserve_ticket_numbers(self, last_used: int) -> None:
        """Continue ticket numbering after `last_used` (e.g. the highest id already in the DB)."""
        self.ticket_counter = max(self.ticket_counter, last_used)

    def current_date(self) -> date:
        """`today` if given, else env FIXED_TODAY (ISO date, keeps cassettes stable), else today."""
        if self.today:
            return self.today
        fixed = os.getenv("FIXED_TODAY")
        return date.fromisoformat(fixed) if fixed else date.today()

    def _device(self, device_id: str) -> dict:
        if device_id not in self.devices:
            raise DeviceAPIError(404, f"device '{device_id}' not found")
        return self.devices[device_id]

    def list_devices(self, user_id: str = "demo-user") -> list[dict]:
        """All devices of the (single) demo user."""
        return [copy.deepcopy(d) for d in self.devices.values()]

    # ---- tools ---------------------------------------------------------
    def get_device_status(self, device_id: str) -> dict:
        """Current status of a device."""
        self._maybe_fail("get_device_status")
        return copy.deepcopy(self._device(device_id))

    def set_fan_speed(self, device_id: str, speed: int) -> dict:
        """Set a fan's speed (1-5). Fails if the device is not a fan or is offline."""
        self._maybe_fail("set_fan_speed")
        device = self._device(device_id)
        if device["type"] != "fan":
            raise DeviceAPIError(400, f"{device_id} is not a fan")
        if isinstance(speed, float) and speed.is_integer():
            speed = int(speed)
        if not isinstance(speed, int) or isinstance(speed, bool) or not 1 <= speed <= 5:
            raise DeviceAPIError(400, "speed must be an integer from 1 to 5")
        if not device["online"]:
            raise DeviceAPIError(409, f"{device_id} is offline")
        previous = device["speed"]
        device["speed"] = speed
        return {"device_id": device_id, "previous_speed": previous, "speed": speed, "ok": True}

    def check_warranty(self, serial: str) -> dict:
        """Warranty status for a serial number."""
        self._maybe_fail("check_warranty")
        serial = serial.strip().upper()
        if serial not in self.warranties:
            raise DeviceAPIError(404, f"serial '{serial}' not found")
        info = self.warranties[serial]
        today = self.current_date()
        expiry = date.fromisoformat(info["expiry"])
        days_left = (expiry - today).days
        return {"serial": serial, "product": info["product"], "expiry": info["expiry"],
                "status": "active" if days_left >= 0 else "expired",
                "days_left": max(days_left, 0)}

    def create_service_ticket(self, device_id: str, issue: str, phone: str) -> dict:
        """Create a ticket. The phone number is validated and only returned masked."""
        self._maybe_fail("create_service_ticket")
        self._device(device_id)
        if not issue or not issue.strip():
            raise DeviceAPIError(400, "issue description is required")
        if not PHONE_RE.match(str(phone).strip()):
            raise DeviceAPIError(400, "phone must be a valid 10-digit Indian mobile number")
        self.ticket_counter += 1
        ticket = {"ticket_id": f"HW-{self.ticket_counter}", "device_id": device_id,
                  "issue": issue.strip(), "phone_masked": mask_pii(str(phone).strip()),
                  "status": "open", "eta": "within 48 hours (metro) / 72 hours (other cities)"}
        self.tickets.append(ticket)
        return dict(ticket)

    def call_tool(self, name: str, args: dict) -> dict:
        """Dispatch a tool call by name. Unknown tools / bad args raise DeviceAPIError(400)."""
        self.calls.append((name, dict(args)))
        handlers = {"get_device_status": self.get_device_status,
                    "set_fan_speed": self.set_fan_speed,
                    "check_warranty": self.check_warranty,
                    "create_service_ticket": self.create_service_ticket}
        if name not in handlers:
            raise DeviceAPIError(400, f"unknown tool '{name}'")
        try:
            return handlers[name](**args)
        except TypeError as exc:
            raise DeviceAPIError(400, f"bad arguments for {name}: {exc}") from exc
