"""Device cloud: seed data, validation and fault injection."""

from datetime import date

import pytest

from assistant.devices import DeviceAPIError, DeviceCloud


def test_seed_devices() -> None:
    cloud = DeviceCloud()
    ids = [d["device_id"] for d in cloud.list_devices()]
    assert ids == ["fan-living", "fan-bedroom", "lock-main", "purifier-kitchen"]
    assert cloud.get_device_status("fan-living")["speed"] == 2
    assert cloud.get_device_status("fan-bedroom")["online"] is False
    assert cloud.get_device_status("lock-main")["battery_pct"] == 15
    assert cloud.get_device_status("purifier-kitchen")["filter_life_pct"] == 8


def test_set_fan_speed_ok_and_errors() -> None:
    cloud = DeviceCloud()
    assert cloud.set_fan_speed("fan-living", 3) == {"device_id": "fan-living", "previous_speed": 2,
                                                    "speed": 3, "ok": True}
    for device, speed, status in [("fan-living", 6, 400), ("fan-living", 0, 400),
                                  ("lock-main", 2, 400), ("fan-bedroom", 2, 409), ("fan-x", 2, 404)]:
        with pytest.raises(DeviceAPIError) as err:
            cloud.set_fan_speed(device, speed)
        assert err.value.status == status


def test_warranty_status() -> None:
    cloud = DeviceCloud(today=date(2026, 11, 3))
    assert cloud.check_warranty("HA-AERO-1001")["status"] == "active"
    lock = cloud.check_warranty("ha-lock-2001")
    assert lock["status"] == "expired" and lock["expiry"] == "2026-01-15"
    assert cloud.check_warranty("HA-PURE-3001")["expiry"] == "2028-06-30"
    with pytest.raises(DeviceAPIError) as err:
        cloud.check_warranty("HA-NOPE-0000")
    assert err.value.status == 404


def test_ticket_masks_phone_and_validates() -> None:
    cloud = DeviceCloud()
    ticket = cloud.create_service_ticket("fan-living", "noisy", "+91 9876543210")
    assert ticket["ticket_id"] == "HW-100001"
    assert "9876543210" not in str(ticket)
    with pytest.raises(DeviceAPIError):
        cloud.create_service_ticket("fan-living", "noisy", "12345")


def test_fault_injection_fails_once_then_recovers() -> None:
    cloud = DeviceCloud(fail_next={"set_fan_speed": 503})
    with pytest.raises(DeviceAPIError) as err:
        cloud.set_fan_speed("fan-living", 3)
    assert err.value.status == 503
    assert cloud.set_fan_speed("fan-living", 3)["ok"] is True


def test_fault_injection_list_and_call_tool() -> None:
    cloud = DeviceCloud(fail_next={"get_device_status": [503, 500]})
    for status in (503, 500):
        with pytest.raises(DeviceAPIError) as err:
            cloud.call_tool("get_device_status", {"device_id": "lock-main"})
        assert err.value.status == status
    assert cloud.call_tool("get_device_status", {"device_id": "lock-main"})["device_id"] == "lock-main"
    with pytest.raises(DeviceAPIError):
        cloud.call_tool("open_door", {})
