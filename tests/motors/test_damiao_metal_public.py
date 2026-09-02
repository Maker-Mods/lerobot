from unittest.mock import MagicMock

import can

from lerobot.motors import Motor
from lerobot.motors.damiao import DamiaoMotorsBus, damiao as damiao_module


def test_sync_write_metal_delegates():
    bus = DamiaoMotorsBus.__new__(DamiaoMotorsBus)  # skip __init__/hardware
    bus._mit_control_batch = MagicMock()
    cmds = {"shoulder_pan": (0.0, 0.5, 10.0, 0.0, 1.2)}
    bus.sync_write_metal(cmds)
    bus._mit_control_batch.assert_called_once_with(cmds)


# ── Reply freshness (`last_update_ts`) ────────────────────────────────────


def _bus(monkeypatch):
    """A bus built without touching hardware, on a clock the test drives."""
    motors = {
        "shoulder_pan": Motor(id=0x01, model="metal_jlo", norm_mode="degrees", recv_id=0x11),
        "gripper": Motor(id=0x07, model="metal_jhi", norm_mode="degrees", recv_id=0x17),
    }
    for motor in motors.values():
        motor.motor_type_str = motor.model
    now = [0.0]
    monkeypatch.setattr(damiao_module.time, "perf_counter", lambda: now[0])
    return DamiaoMotorsBus(port="can0", motors=motors), now


def _reply(recv_id=0x11):
    """Any well-formed 8-byte MIT state payload; the decoded values do not matter here."""
    return can.Message(
        arbitration_id=recv_id,
        data=bytes([recv_id, 0x80, 0x00, 0x80, 0x00, 0x00, 25, 30]),
        is_extended_id=False,
    )


def test_every_motor_starts_with_a_timestamp(monkeypatch):
    """A motor that has never answered still needs an entry, or a caller comparing against
    `max(...)` would raise KeyError on the very first tick instead of watching for silence."""
    bus, _now = _bus(monkeypatch)
    assert set(bus.last_update_ts) == {"shoulder_pan", "gripper"}


def test_last_update_ts_is_a_copy(monkeypatch):
    bus, _now = _bus(monkeypatch)
    bus.last_update_ts["shoulder_pan"] = 999.0
    assert bus.last_update_ts["shoulder_pan"] != 999.0


def test_a_decoded_reply_stamps_only_that_motor(monkeypatch):
    bus, now = _bus(monkeypatch)
    now[0] = 12.5
    bus._process_response("shoulder_pan", _reply())

    assert bus.last_update_ts["shoulder_pan"] == 12.5
    assert bus.last_update_ts["gripper"] == 0.0  # still silent, and the stamp says so


def test_an_undecodable_reply_does_not_stamp(monkeypatch):
    """A frame we could not decode says nothing about the motor being alive."""
    bus, now = _bus(monkeypatch)
    now[0] = 12.5
    bus._process_response("shoulder_pan", can.Message(arbitration_id=0x11, data=bytes([1, 2, 3])))

    assert bus.last_update_ts["shoulder_pan"] == 0.0


def test_reset_update_timestamps_rearms_every_motor(monkeypatch):
    bus, now = _bus(monkeypatch)
    now[0] = 40.0
    bus.reset_update_timestamps()

    assert bus.last_update_ts == {"shoulder_pan": 40.0, "gripper": 40.0}


def test_reset_leaves_the_cached_state_alone(monkeypatch):
    """Freshness and the cached values are separate: re-arming the watchdog must not invent
    positions, and must not throw away the last real ones either."""
    bus, now = _bus(monkeypatch)
    bus._process_response("shoulder_pan", _reply())
    cached = bus._last_known_states["shoulder_pan"].copy()

    now[0] = 40.0
    bus.reset_update_timestamps()

    assert bus._last_known_states["shoulder_pan"] == cached
