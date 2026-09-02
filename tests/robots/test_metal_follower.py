#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors.damiao.tables import MIT_KD_RANGE, MIT_KP_RANGE, MOTOR_LIMIT_PARAMS
from lerobot.robots.bi_metal_follower import BiMetalFollower, BiMetalFollowerConfig
from lerobot.robots.metal_follower import metal_follower as metal_follower_module
from lerobot.robots.metal_follower.config_metal_follower import (
    MetalFollowerConfig,
    MetalFollowerConfigBase,
)
from lerobot.robots.metal_follower.metal_follower import MetalFollower
from lerobot.utils.errors import DeviceNotConnectedError

# The follower resolves its gains from the config defaults, so read them from there.
DEFAULT_GAINS = MetalFollowerConfig(port="can0").gains
DEFAULT_CAN_IDS = MetalFollowerConfig(port="can0").motor_can_ids


@pytest.fixture
def follower():
    # startup slow-sync off so these tests exercise soft limits / gains directly (a dedicated
    # test covers the sync behaviour). Velocity feedforward is also off for the legacy mock
    # harness; packet-level feedforward tests below use the real Damiao encoder.
    r = MetalFollower(
        MetalFollowerConfig(port="can0", startup_sync_speed_deg=None, velocity_feedforward=False)
    )
    r.bus = MagicMock()
    r.bus.sync_read.return_value = dict.fromkeys(r._joint_motor_names, 0.0)
    return r


def _make_wired_follower(**config_overrides):
    config = MetalFollowerConfig(port="can0", startup_sync_speed_deg=None, **config_overrides)
    follower = MetalFollower(config)
    wire = MagicMock()
    follower.bus.canbus = wire
    follower.bus._recv_all_responses = MagicMock(return_value={})

    def connect_bus():
        follower.bus._is_connected = True

    follower.bus.connect = MagicMock(side_effect=connect_bus)
    follower.bus.enable_torque = MagicMock()
    follower.connect(calibrate=False)
    wire.reset_mock()
    return follower, wire


def _last_frame(wire):
    return wire.send.call_args.args[0]


def _decode_dq_deg_s(follower, frame, motor="shoulder_pan"):
    dq_uint = (frame.data[2] << 4) | (frame.data[3] >> 4)
    vmax = MOTOR_LIMIT_PARAMS[follower.bus._motor_types[motor]][1]
    return math.degrees(dq_uint / ((1 << 12) - 1) * (2.0 * vmax) - vmax)


def _decode_gain(frame, value_range, field):
    if field == "kp":
        value_uint = ((frame.data[3] & 0x0F) << 8) | frame.data[4]
    else:
        value_uint = (frame.data[5] << 4) | (frame.data[6] >> 4)
    return value_uint / ((1 << 12) - 1) * (value_range[1] - value_range[0]) + value_range[0]


def test_has_all_seven_motors(follower):
    assert follower._joint_motor_names == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    ]


def test_action_features_include_all_motors(follower):
    assert "shoulder_pan.pos" in follower.action_features
    assert "gripper.pos" in follower.action_features


def test_velocity_feedforward_config_defaults():
    config = MetalFollowerConfig()
    assert config.velocity_feedforward is True
    assert config.velocity_ff_alpha == 0.08
    assert config.velocity_ff_max_deg_s == 120.0


def test_send_action_writes_goal_and_clamps(follower):
    action = {f"{m}.pos": 999.0 for m in follower._joint_motor_names}
    out = follower.send_action(action)
    assert follower.bus.sync_write.called
    assert out["shoulder_pan.pos"] == 160.0  # clamped to shoulder_pan upper soft limit


def test_factory_builds_metal_follower():
    from lerobot.robots.metal_follower.config_metal_follower import MetalFollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    r = make_robot_from_config(MetalFollowerConfig(port="can0"))
    assert r.name == "metal_follower"
    assert type(r).__name__ == "MetalFollower"


def test_startup_slow_sync_clamps_then_syncs():
    r = MetalFollower(
        MetalFollowerConfig(
            port="can0",
            startup_sync_speed_deg=3.0,
            startup_sync_tolerance_deg=3.0,
            velocity_feedforward=False,
        )
    )
    r.bus = MagicMock()
    r.bus.sync_read.return_value = {"shoulder_pan": 0.0}
    out = r.send_action({"shoulder_pan.pos": 50.0})
    assert out["shoulder_pan.pos"] == 3.0  # clamped to +3 deg/step from present 0
    assert r._synced is False
    r.bus.sync_read.return_value = {"shoulder_pan": 48.0}
    r.send_action({"shoulder_pan.pos": 50.0})  # err 2 <= tolerance 3 -> synced
    assert r._synced is True


def test_connect_sets_follow_gains(follower):
    follower.cameras = {}
    follower.bus.is_connected = False  # so @check_if_already_connected doesn't trip
    follower.connect(calibrate=False)
    kp_call = {m: kp for m, (kp, kd) in DEFAULT_GAINS.items()}
    kd_call = {m: kd for m, (kp, kd) in DEFAULT_GAINS.items()}
    follower.bus.sync_write.assert_any_call("Kp", kp_call)
    follower.bus.sync_write.assert_any_call("Kd", kd_call)
    assert follower._resolved_gains == DEFAULT_GAINS


def test_moving_goals_encode_filtered_velocity_in_mit_frame():
    follower, wire = _make_wired_follower()

    with patch("lerobot.robots.metal_follower.metal_follower.time.perf_counter", side_effect=[1.0, 1.02]):
        follower.send_action({"shoulder_pan.pos": 0.0})
        follower.send_action({"shoulder_pan.pos": 10.0})

    frame = _last_frame(wire)
    # Raw velocity is 10 deg / 0.02 s = 500 deg/s, clipped to velocity_ff_max_deg_s. The previous
    # velocity is 0, so the first EMA update is just alpha * clipped. Derived from the config so
    # retuning the feedforward defaults doesn't break this.
    expected = follower.config.velocity_ff_alpha * follower.config.velocity_ff_max_deg_s
    assert _decode_dq_deg_s(follower, frame) == pytest.approx(expected, abs=0.3)
    dq_uint = (frame.data[2] << 4) | (frame.data[3] >> 4)
    assert dq_uint != ((1 << 12) - 1) // 2


def test_velocity_feedforward_off_is_byte_identical_to_legacy_goal_write():
    follower, wire = _make_wired_follower(velocity_feedforward=False)

    follower.send_action({"shoulder_pan.pos": 10.0})
    follower_frame = bytes(_last_frame(wire).data)
    wire.reset_mock()
    follower.bus.sync_write("Goal_Position", {"shoulder_pan": 10.0})
    legacy_frame = bytes(_last_frame(wire).data)

    assert follower_frame == legacy_frame
    dq_uint = (legacy_frame[2] << 4) | (legacy_frame[3] >> 4)
    assert dq_uint == ((1 << 12) - 1) // 2


def test_velocity_feedforward_is_clipped_to_configured_maximum():
    follower, wire = _make_wired_follower(velocity_ff_alpha=1.0, velocity_ff_max_deg_s=40.0)

    with patch("lerobot.robots.metal_follower.metal_follower.time.perf_counter", side_effect=[2.0, 2.01]):
        follower.send_action({"shoulder_pan.pos": 0.0})
        follower.send_action({"shoulder_pan.pos": 100.0})

    assert _decode_dq_deg_s(follower, _last_frame(wire)) == pytest.approx(40.0, abs=0.3)


def test_mit_frames_use_configured_follow_gains():
    gains = dict.fromkeys(DEFAULT_GAINS, (123.0, 1.25))
    follower, wire = _make_wired_follower(gains=gains)

    with patch("lerobot.robots.metal_follower.metal_follower.time.perf_counter", return_value=3.0):
        follower.send_action({"shoulder_pan.pos": 5.0})

    frame = _last_frame(wire)
    assert _decode_gain(frame, MIT_KP_RANGE, "kp") == pytest.approx(123.0, abs=0.13)
    assert _decode_gain(frame, MIT_KD_RANGE, "kd") == pytest.approx(1.25, abs=0.002)


def _decode_tau_nm(follower, frame, motor):
    tau_uint = ((frame.data[6] & 0x0F) << 8) | frame.data[7]
    tmax = MOTOR_LIMIT_PARAMS[follower.bus._motor_types[motor]][2]
    return tau_uint / ((1 << 12) - 1) * (2.0 * tmax) - tmax


def _frame_for(wire, send_id):
    for call in wire.send.call_args_list:
        if call.args[0].arbitration_id == send_id:
            return call.args[0]
    raise AssertionError(f"no frame sent to CAN id 0x{send_id:02X}")


def test_mit_frames_carry_no_feedforward_torque():
    follower, wire = _make_wired_follower()

    follower.send_action({"shoulder_lift.pos": -40.0})

    frame = _last_frame(wire)
    tau_uint = ((frame.data[6] & 0x0F) << 8) | frame.data[7]
    assert tau_uint == ((1 << 12) - 1) // 2  # exact 0.0 encoding


def test_velocity_estimator_resets_after_long_action_gap():
    follower, wire = _make_wired_follower(velocity_ff_alpha=1.0)

    with patch(
        "lerobot.robots.metal_follower.metal_follower.time.perf_counter",
        side_effect=[4.0, 4.1, 4.7],
    ):
        follower.send_action({"shoulder_pan.pos": 0.0})
        follower.send_action({"shoulder_pan.pos": 10.0})
        moving_velocity = _decode_dq_deg_s(follower, _last_frame(wire))
        follower.send_action({"shoulder_pan.pos": 20.0})

    assert moving_velocity == pytest.approx(100.0, abs=0.3)
    assert _decode_dq_deg_s(follower, _last_frame(wire)) == pytest.approx(0.0, abs=0.3)


# ── Startup-sync stall release ────────────────────────────────────────────


def _stuck_follower(monkeypatch, **overrides):
    """A follower mid-startup-sync whose `wrist_yaw` never moves, however hard it is driven."""
    r = MetalFollower(
        MetalFollowerConfig(
            port="can0",
            startup_sync_speed_deg=1.0,
            startup_sync_tolerance_deg=3.0,
            velocity_feedforward=False,
            **overrides,
        )
    )
    r.bus = MagicMock()
    r.bus.sync_read.return_value = dict.fromkeys(r._joint_motor_names, 0.0)
    now = [0.0]
    monkeypatch.setattr(metal_follower_module.time, "perf_counter", lambda: now[0])
    return r, now


def test_a_stalled_joint_is_released_to_the_release_rate_not_the_raw_target(monkeypatch):
    r, now = _stuck_follower(monkeypatch)
    assert r.config.startup_sync_release_speed_deg == 5.0

    # First tick: the ordinary 1 deg/step ramp.
    assert r.send_action({"wrist_yaw.pos": 40.0})["wrist_yaw.pos"] == pytest.approx(1.0)

    # The joint has not budged for longer than the stall window, so it is released -- but to
    # 5 deg/step, not to the 40 deg raw target it could not reach.
    now[0] = 2.0
    assert r.send_action({"wrist_yaw.pos": 40.0})["wrist_yaw.pos"] == pytest.approx(5.0)
    assert r._released_motors == {"wrist_yaw"}
    # A released joint stays rate-limited, tick after tick, until it converges.
    now[0] = 2.1
    assert r.send_action({"wrist_yaw.pos": 40.0})["wrist_yaw.pos"] == pytest.approx(5.0)
    assert r._synced is False


def test_a_released_joint_rejoins_the_synced_set_once_it_converges(monkeypatch):
    r, now = _stuck_follower(monkeypatch)
    r.send_action({"wrist_yaw.pos": 40.0})
    now[0] = 2.0
    r.send_action({"wrist_yaw.pos": 40.0})
    assert r._released_motors == {"wrist_yaw"}

    # Freed at last: within tolerance, so it tracks at full speed like any other joint.
    r.bus.sync_read.return_value = {**dict.fromkeys(r._joint_motor_names, 0.0), "wrist_yaw": 38.5}
    now[0] = 2.1
    assert r.send_action({"wrist_yaw.pos": 40.0})["wrist_yaw.pos"] == 40.0
    assert r._released_motors == set()
    assert r._synced is True


def test_release_speed_none_restores_the_unbounded_release(monkeypatch):
    r, now = _stuck_follower(monkeypatch, startup_sync_release_speed_deg=None)
    assert r.send_action({"wrist_yaw.pos": 40.0})["wrist_yaw.pos"] == pytest.approx(1.0)

    now[0] = 2.0
    assert r.send_action({"wrist_yaw.pos": 40.0})["wrist_yaw.pos"] == 40.0
    assert r._released_motors == set()
    assert r._synced is True


def test_a_stalled_joint_does_not_hold_back_the_joints_that_converge(monkeypatch):
    r, now = _stuck_follower(monkeypatch)
    action = {"wrist_yaw.pos": 40.0, "wrist_flex.pos": 20.0}
    out = r.send_action(action)
    assert out == {"wrist_yaw.pos": pytest.approx(1.0), "wrist_flex.pos": pytest.approx(1.0)}

    # wrist_flex has travelled and is within tolerance; wrist_yaw is still stuck at zero.
    r.bus.sync_read.return_value = {**dict.fromkeys(r._joint_motor_names, 0.0), "wrist_flex": 19.0}
    now[0] = 2.0
    out = r.send_action(action)
    assert out["wrist_flex.pos"] == 20.0  # raw target, exactly as before this change
    assert out["wrist_yaw.pos"] == pytest.approx(5.0)
    assert r._synced_motors == {"wrist_flex"}
    assert r._released_motors == {"wrist_yaw"}


def test_bimanual_propagates_the_stall_release_and_watchdog_settings():
    """`_arm_config` rebuilds the per-arm config field by field, so a safety knob left out of
    that list is silently swapped for its default on a bimanual rig."""
    robot = BiMetalFollower(
        BiMetalFollowerConfig(
            left_arm_config=MetalFollowerConfigBase(
                port="can0",
                startup_sync_release_speed_deg=2.5,
                stale_read_timeout_s=1.5,
            ),
            right_arm_config=MetalFollowerConfigBase(
                port="can1",
                startup_sync_release_speed_deg=None,
                stale_read_timeout_s=None,
            ),
        )
    )
    assert robot.left_arm.config.startup_sync_release_speed_deg == 2.5
    assert robot.left_arm.config.stale_read_timeout_s == 1.5
    assert robot.right_arm.config.startup_sync_release_speed_deg is None
    assert robot.right_arm.config.stale_read_timeout_s is None


# ── Stale-read watchdog ───────────────────────────────────────────────────


def _watchdog_follower(monkeypatch, **overrides):
    """A connected-looking follower whose bus freshness stamps the test drives by hand.

    `bus.last_update_ts` is a real dict shared with the caller, so a test scripts "this motor
    answered on this tick" by writing into it, exactly as `_process_response` would.
    """
    r = MetalFollower(
        MetalFollowerConfig(
            port="can0",
            startup_sync_speed_deg=None,
            velocity_feedforward=False,
            **overrides,
        )
    )
    r.bus = MagicMock()
    r.cameras = {}
    r.bus.sync_read.return_value = dict.fromkeys(r._joint_motor_names, 4.0)
    now = [0.0]
    monkeypatch.setattr(metal_follower_module.time, "perf_counter", lambda: now[0])
    stamps = dict.fromkeys(r._joint_motor_names, 0.0)
    r.bus.last_update_ts = stamps
    r.bus.reset_update_timestamps.side_effect = lambda: stamps.update(
        dict.fromkeys(r._joint_motor_names, now[0])
    )
    return r, now, stamps


def test_one_silent_joint_never_trips_the_watchdog(monkeypatch):
    """The normal packet-drop case: the bus loses one motor's reply, for a long time, while the
    arm is plainly alive. Serving that motor from the state cache must keep working."""
    r, now, stamps = _watchdog_follower(monkeypatch)
    answering = [m for m in r._joint_motor_names if m != "wrist_roll"]

    for _ in range(20):
        now[0] += 1.0
        stamps.update(dict.fromkeys(answering, now[0]))
        obs = r.get_observation()

    assert obs["wrist_roll.pos"] == 4.0
    assert now[0] - stamps["wrist_roll"] == pytest.approx(20.0)


def test_a_fully_silent_arm_raises_once_past_the_timeout(monkeypatch):
    r, now, _stamps = _watchdog_follower(monkeypatch)
    assert r.config.stale_read_timeout_s == 0.5

    # Inside the window this is still just a burst of drops; the cached pose is returned.
    now[0] = 0.4
    assert r.get_observation()["gripper.pos"] == 4.0

    now[0] = 0.6
    with pytest.raises(DeviceNotConnectedError, match="stopped answering"):
        r.get_observation()


def test_the_watchdog_message_names_the_arm_and_the_timeout(monkeypatch):
    r, now, _stamps = _watchdog_follower(monkeypatch)
    now[0] = 3.0
    with pytest.raises(DeviceNotConnectedError) as excinfo:
        r.get_observation()
    message = str(excinfo.value)
    assert "MetalFollower" in message
    assert "0.5" in message
    assert "power" in message and "CAN cable" in message


def test_a_successful_read_resets_the_stale_clock(monkeypatch):
    r, now, stamps = _watchdog_follower(monkeypatch)
    now[0] = 0.4
    r.get_observation()

    # The arm answers again, which re-arms the full window from here.
    now[0] = 0.45
    stamps.update(dict.fromkeys(r._joint_motor_names, now[0]))
    assert r.get_observation()["gripper.pos"] == 4.0

    now[0] = 0.8  # 0.8 s since the start, but only 0.35 s since the arm last answered
    assert r.get_observation()["gripper.pos"] == 4.0

    now[0] = 1.1  # 0.65 s of silence, past the timeout
    with pytest.raises(DeviceNotConnectedError):
        r.get_observation()


def test_the_watchdog_can_be_disabled(monkeypatch):
    r, now, _stamps = _watchdog_follower(monkeypatch, stale_read_timeout_s=None)
    now[0] = 600.0
    assert r.get_observation()["gripper.pos"] == 4.0


def test_connect_rearms_the_watchdog_window(monkeypatch):
    """Constructing a robot and connecting it can be seconds apart; that gap is not silence."""
    r, now, stamps = _watchdog_follower(monkeypatch)
    r.bus.is_connected = False  # so @check_if_already_connected doesn't trip

    now[0] = 30.0
    r.connect(calibrate=False)
    r.bus.is_connected = True  # the mock bus does not model the connect itself

    # 30 s passed before the first read, and none of it counts as the arm going silent.
    assert stamps == dict.fromkeys(r._joint_motor_names, 30.0)
    assert r.get_observation()["gripper.pos"] == 4.0


# ── Transport guard ───────────────────────────────────────────────────────


def test_slcan_is_accepted():
    """slcan is the only CAN transport available on macOS/Windows, where SocketCAN does not exist.

    Measured on a Metal arm over a CANable at the follower's 60 Hz: a full tick (7 state requests
    + 7 replies + 7 MIT writes + 7 replies) costs ~5 ms of a 16.7 ms period and drops nothing.
    """
    follower = MetalFollower(MetalFollowerConfig(port="/dev/ttyACM0", can_interface="slcan"))
    assert follower.bus.can_interface == "slcan"


def test_socketcan_is_accepted():
    follower = MetalFollower(MetalFollowerConfig(port="can0", can_interface="socketcan"))
    assert follower.bus.can_interface == "socketcan"


def test_unknown_can_interface_is_rejected():
    with pytest.raises(ValueError, match="socketcan"):
        MetalFollower(MetalFollowerConfig(port="can0", can_interface="pcan"))


def test_port_is_required():
    """`port` has no portable default: "can0" is meaningless on macOS/Windows, and silently
    defaulting to it would send a Mac user into a SocketCAN failure with no hint why."""
    with pytest.raises(ValueError, match="requires `port`"):
        MetalFollower(MetalFollowerConfig())


def test_defaults_to_slcan():
    """slcan is the default because it is the only transport that works on all three platforms
    and needs no privileged setup."""
    assert MetalFollowerConfig().can_interface == "slcan"
    assert MetalFollowerConfig().port is None
