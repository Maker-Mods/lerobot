#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors.robstride.tables import MOTOR_LIMIT_PARAMS, MotorType
from lerobot.robots.bi_maker_follower import BiMakerFollower, BiMakerFollowerConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.maker_follower import MakerFollower, MakerFollowerConfig, MakerFollowerConfigBase
from lerobot.robots.maker_follower.maker_follower import MOTOR_MODELS
from lerobot.robots.utils import make_robot_from_config

JOINTS = list(MakerFollowerConfig(port="x").motor_can_ids)


def set_positions(bus, positions):
    """Script the per-motor `bus.read("Present_Position", motor)` the follower uses."""
    bus.read.side_effect = lambda _data_name, motor: positions[motor]


def goal_writes(bus):
    return {c.args[1]: c.args[2] for c in bus.write.call_args_list if c.args[0] == "Goal_Position"}


@pytest.fixture
def follower(tmp_path):
    # Startup slow-sync off so these tests exercise soft limits / gains directly (a dedicated
    # test covers the sync behaviour).
    r = MakerFollower(
        MakerFollowerConfig(
            port="/dev/fake",
            calibration_dir=tmp_path,
            startup_sync_speed_deg=None,
        )
    )
    r.bus = MagicMock()
    set_positions(r.bus, dict.fromkeys(r._joint_motor_names, 0.0))
    r.bus.is_connected = True
    return r


def test_config_requires_port():
    with pytest.raises(ValueError, match="requires `port`"):
        MakerFollower(MakerFollowerConfig())


def test_config_rejects_unknown_can_interface():
    with pytest.raises(ValueError, match="can_interface"):
        MakerFollower(MakerFollowerConfig(port="/dev/fake", can_interface="pcan"))


def test_bus_is_built_for_classic_can_with_the_right_models():
    with patch("lerobot.robots.maker_follower.maker_follower.RobstrideMotorsBus") as bus_cls:
        MakerFollower(MakerFollowerConfig(port="/dev/fake", can_interface="socketcan"))
    kwargs = bus_cls.call_args.kwargs
    assert kwargs["port"] == "/dev/fake"
    assert kwargs["can_interface"] == "socketcan"
    assert kwargs["use_can_fd"] is False
    assert kwargs["bitrate"] == 1_000_000
    motors = kwargs["motors"]
    assert list(motors) == JOINTS
    for name, motor in motors.items():
        assert motor.model == MOTOR_MODELS[name]
        assert motor.motor_type_str == MOTOR_MODELS[name]
        assert motor.recv_id == motor.id
    # RS02 joints carry the wider torque/velocity table.
    assert MOTOR_MODELS["shoulder_lift"] == MOTOR_MODELS["elbow_flex"] == "O1"
    assert MOTOR_LIMIT_PARAMS[MotorType.O1][2] > MOTOR_LIMIT_PARAMS[MotorType.O0][2]


def test_features(follower):
    assert follower.action_features == {f"{j}.pos": float for j in JOINTS}
    assert follower.observation_features == follower.action_features


def test_defaults():
    cfg = MakerFollowerConfig(port="/dev/fake")
    assert cfg.can_interface == "slcan"
    assert cfg.disable_torque_on_disconnect is True
    assert set(cfg.joint_limits) == set(cfg.gains) == set(JOINTS)
    for lo, hi in cfg.joint_limits.values():
        assert lo < hi
    for kp, kd in cfg.gains.values():
        assert 0 < kp <= 500 and 0 < kd <= 5


def test_connect_pushes_gains_before_enabling(follower):
    follower.calibration = {"fake": object()}
    follower.bus.is_connected = False

    def _connect():
        follower.bus.is_connected = True

    follower.bus.connect.side_effect = _connect
    follower.connect(calibrate=False)
    calls = [c[0] for c in follower.bus.method_calls]
    assert calls.index("sync_write") < calls.index("enable_torque")
    follower.bus.sync_write.assert_any_call("Kp", {j: kp for j, (kp, _) in follower.config.gains.items()})
    follower.bus.sync_write.assert_any_call("Kd", {j: kd for j, (_, kd) in follower.config.gains.items()})


def test_connect_failure_releases_the_bus(follower):
    follower.bus.is_connected = False
    follower.bus.enable_torque.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        follower.connect(calibrate=False)
    follower.bus.disconnect.assert_called_once_with(True)


def test_send_action_clips_to_soft_limits(follower):
    lo, hi = follower.config.joint_limits["gripper"]
    out = follower.send_action({"gripper.pos": hi + 50.0, "shoulder_pan.pos": -1000.0})
    assert out["gripper.pos"] == hi
    assert out["shoulder_pan.pos"] == follower.config.joint_limits["shoulder_pan"][0]
    assert goal_writes(follower.bus) == {
        "gripper": hi,
        "shoulder_pan": follower.config.joint_limits["shoulder_pan"][0],
    }


def test_startup_sync_caps_the_first_steps(tmp_path):
    r = MakerFollower(
        MakerFollowerConfig(port="/dev/fake", calibration_dir=tmp_path, startup_sync_speed_deg=1.0)
    )
    r.bus = MagicMock()
    r.bus.is_connected = True
    set_positions(r.bus, dict.fromkeys(JOINTS, 0.0))
    out = r.send_action({"wrist_yaw.pos": 40.0})
    assert out["wrist_yaw.pos"] == pytest.approx(1.0)
    assert r._synced is False
    # Once within tolerance the joint is released and the arm tracks at full speed.
    set_positions(r.bus, {**dict.fromkeys(JOINTS, 0.0), "wrist_yaw": 38.5})
    out = r.send_action({"wrist_yaw.pos": 40.0})
    assert out["wrist_yaw.pos"] == 40.0
    assert r._synced is True


def test_max_relative_target_caps_motion(tmp_path):
    r = MakerFollower(
        MakerFollowerConfig(
            port="/dev/fake",
            calibration_dir=tmp_path,
            startup_sync_speed_deg=None,
            max_relative_target=5.0,
        )
    )
    r.bus = MagicMock()
    r.bus.is_connected = True
    set_positions(r.bus, dict.fromkeys(JOINTS, 0.0))
    out = r.send_action({"wrist_flex.pos": 30.0})
    assert out["wrist_flex.pos"] == pytest.approx(5.0)


def test_calibrate_zeroes_and_saves(follower, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    follower.calibrate()
    follower.bus.disable_torque.assert_called_once()
    follower.bus.set_zero_position.assert_called_once()
    assert set(follower.calibration) == set(JOINTS)
    cal = follower.calibration["elbow_flex"]
    assert cal.id == follower.config.motor_can_ids["elbow_flex"]
    assert (cal.range_min, cal.range_max) == tuple(int(v) for v in follower.config.joint_limits["elbow_flex"])
    assert follower.calibration_fpath.exists()


def test_disconnect_passes_the_torque_flag(follower):
    follower.disconnect()
    follower.bus.disconnect.assert_called_once_with(True)


def test_registered_types_resolve():
    assert RobotConfig.get_choice_class("maker_follower") is MakerFollowerConfig
    assert RobotConfig.get_choice_class("bi_maker_follower") is BiMakerFollowerConfig
    with patch("lerobot.robots.maker_follower.maker_follower.RobstrideMotorsBus"):
        assert isinstance(make_robot_from_config(MakerFollowerConfig(port="/dev/fake")), MakerFollower)


def test_bimanual_prefixes_keys_and_splits_buses():
    with patch("lerobot.robots.maker_follower.maker_follower.RobstrideMotorsBus") as bus_cls:
        robot = BiMakerFollower(
            BiMakerFollowerConfig(
                left_arm_config=MakerFollowerConfigBase(port="/dev/left"),
                right_arm_config=MakerFollowerConfigBase(port="/dev/right"),
            )
        )
    ports = [c.kwargs["port"] for c in bus_cls.call_args_list]
    assert ports == ["/dev/left", "/dev/right"]
    assert set(robot.action_features) == {f"{side}_{j}.pos" for side in ("left", "right") for j in JOINTS}
    assert robot.left_arm.config.gains == robot.right_arm.config.gains == MakerFollowerConfigBase().gains


def _connected(follower):
    follower.bus.is_connected = False

    def _connect():
        follower.bus.is_connected = True

    follower.bus.connect.side_effect = _connect
    follower.calibration = {"fake": object()}
    return follower


def test_connect_corrects_a_full_turn_jump(follower, caplog):
    low, high = follower.config.joint_limits["shoulder_lift"]
    set_positions(follower.bus, {**dict.fromkeys(JOINTS, 0.0), "shoulder_lift": high + 360.0 - 10.0})
    _connected(follower).connect(calibrate=False)
    assert follower._turn_offset["shoulder_lift"] == -360.0
    obs = follower.get_observation()
    assert obs["shoulder_lift.pos"] == pytest.approx(high - 10.0)
    out = follower.send_action({"shoulder_lift.pos": high - 20.0})
    assert out["shoulder_lift.pos"] == high - 20.0
    assert goal_writes(follower.bus)["shoulder_lift"] == pytest.approx(high - 20.0 + 360.0)


def test_stale_zero_blocks_motion_until_recalibrated(follower, monkeypatch):
    low, high = follower.config.joint_limits["wrist_yaw"]
    set_positions(follower.bus, {**dict.fromkeys(JOINTS, 0.0), "wrist_yaw": high + 90.0})
    # connect() must still succeed, because `lerobot-calibrate` connects before it calibrates.
    _connected(follower).connect(calibrate=False)
    with pytest.raises(RuntimeError, match="lerobot-calibrate"):
        follower.send_action({"wrist_yaw.pos": 0.0})
    assert goal_writes(follower.bus) == {}

    # Answer "c" to the reuse prompt so a fresh zero is written.
    monkeypatch.setattr("builtins.input", lambda *_: "c")
    follower.calibrate()
    set_positions(follower.bus, dict.fromkeys(JOINTS, 0.0))
    follower.send_action({"wrist_yaw.pos": 0.0})
    assert goal_writes(follower.bus) == {"wrist_yaw": 0.0}


def test_a_missed_reply_keeps_the_previous_position(follower):
    set_positions(follower.bus, dict.fromkeys(JOINTS, 4.0))
    assert follower._read_raw_positions()["wrist_roll"] == 4.0

    def _raise(_data_name, motor):
        if motor == "wrist_roll":
            raise ConnectionError("no response")
        return 9.0

    follower.bus.read.side_effect = _raise
    positions = follower._read_raw_positions()
    assert positions["wrist_roll"] == 4.0
    assert positions["wrist_yaw"] == 9.0
