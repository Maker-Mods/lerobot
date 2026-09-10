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

import draccus
import pytest

from lerobot.robots.maker_follower.config_maker_follower import MakerFollowerConfig
from lerobot.teleoperators.bi_rebot_102_leader import (
    BiRebot102Leader,
    BiRebot102LeaderMakerConfig,
    BiRebot102LeaderMakerTriggerConfig,
)
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.rebot_102_leader import (
    RebotArm102Leader,
    RebotArm102LeaderConfig,
    RebotArm102LeaderMakerConfig,
    RebotArm102LeaderMakerTeleopConfig,
    RebotArm102LeaderMakerTriggerConfig,
    RebotArm102LeaderMakerTriggerTeleopConfig,
)
from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader_maker import (
    MAKER_TRIGGER_GRIPPER_TRAVEL_DEG,
    MAKER_TRIGGER_GRIPPER_USABLE_TRAVEL_DEG,
)
from lerobot.teleoperators.utils import make_teleoperator_from_config

_MODULE = "lerobot.teleoperators.rebot_102_leader.rebot_102_leader"


def _make_bus_mock(raw_deg: float) -> MagicMock:
    bus = MagicMock(name="FashionStarServoMock")
    bus.ping.return_value = True

    def _sync_monitor(ids):
        monitors = {}
        for servo_id in ids:
            monitor = MagicMock()
            monitor.angle_deg = raw_deg
            monitors[servo_id] = monitor
        return monitors

    bus.sync_monitor.side_effect = _sync_monitor
    return bus


def test_maker_ranges_match_follower_limits():
    """The preset may never emit a target the follower would have to clip.

    `joint_ranges` are ints, `joint_limits` floats, so the check is containment with the
    rounding slack that implies, not equality.
    """
    ranges = RebotArm102LeaderMakerConfig(port="/dev/null").joint_ranges
    limits = MakerFollowerConfig(port="/dev/null").joint_limits
    assert set(ranges) == set(limits)
    for joint, (lo, hi) in ranges.items():
        lim_lo, lim_hi = limits[joint]
        assert lim_lo - 1.0 <= lo < hi <= lim_hi + 1.0, joint


def test_maker_preset_shares_hardware_fields_with_parent():
    base = RebotArm102LeaderConfig(port="/dev/null")
    maker = RebotArm102LeaderMakerConfig(port="/dev/null")
    assert maker.joint_ids == base.joint_ids
    assert maker.baudrate == base.baudrate
    assert set(maker.joint_directions) == set(base.joint_directions)
    # Every direction is a non-zero float scale.
    assert all(isinstance(d, float) and d != 0.0 for d in maker.joint_directions.values())


def test_maker_teleop_config_is_registered():
    cfg = TeleoperatorConfig.get_choice_class("rebot_102_leader_maker")
    assert cfg is RebotArm102LeaderMakerTeleopConfig
    assert TeleoperatorConfig.get_choice_class("bi_rebot_102_leader_maker") is BiRebot102LeaderMakerConfig


def test_factory_builds_the_stock_leader_for_the_maker_type():
    cfg = RebotArm102LeaderMakerTeleopConfig(port="/dev/null")
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        teleop = make_teleoperator_from_config(cfg)
    assert isinstance(teleop, RebotArm102Leader)
    assert teleop.config.joint_directions == cfg.joint_directions


def test_bimanual_factory_propagates_maker_mapping():
    cfg = BiRebot102LeaderMakerConfig(
        left_arm_config=RebotArm102LeaderMakerConfig(port="/dev/left"),
        right_arm_config=RebotArm102LeaderMakerConfig(port="/dev/right"),
    )
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        teleop = make_teleoperator_from_config(cfg)
    assert isinstance(teleop, BiRebot102Leader)
    expected = RebotArm102LeaderMakerConfig(port="x")
    for arm in (teleop.left_arm, teleop.right_arm):
        assert arm.config.joint_directions == expected.joint_directions
        assert arm.config.joint_ranges == expected.joint_ranges
    assert teleop.left_arm.config.port == "/dev/left"
    assert teleop.right_arm.config.port == "/dev/right"


def test_bimanual_cli_parse_only_needs_the_ports():
    cfg = draccus.parse(
        BiRebot102LeaderMakerConfig,
        args=["--left_arm_config.port=/dev/a", "--right_arm_config.port=/dev/b"],
    )
    assert cfg.left_arm_config.port == "/dev/a"
    assert cfg.right_arm_config.joint_directions == RebotArm102LeaderMakerConfig(port="x").joint_directions


@pytest.mark.parametrize("raw_deg", [5.0, -5.0])
def test_action_applies_scale_and_stays_inside_range(raw_deg):
    cfg = RebotArm102LeaderMakerTeleopConfig(port="/dev/null")
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.FashionStarServo", return_value=_make_bus_mock(raw_deg)),
    ):
        teleop = RebotArm102Leader(cfg)
        teleop.connect(calibrate=False)
        action = teleop.get_action()
        teleop.disconnect()
    for joint, (lo, hi) in cfg.joint_ranges.items():
        value = action[f"{joint}.pos"]
        assert lo <= value <= hi, joint


# --- the trigger-gripper variant -------------------------------------------------------------


def _action_for_raw(cfg, raw_deg: float) -> dict[str, float]:
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.FashionStarServo", return_value=_make_bus_mock(raw_deg)),
    ):
        teleop = RebotArm102Leader(cfg)
        teleop.connect(calibrate=False)
        try:
            return teleop.get_action()
        finally:
            teleop.disconnect()


def test_trigger_preset_differs_from_lever_only_on_the_gripper():
    lever = RebotArm102LeaderMakerConfig(port="/dev/null")
    trigger = RebotArm102LeaderMakerTriggerConfig(port="/dev/null")
    assert trigger.joint_ranges == lever.joint_ranges
    assert trigger.joint_ids == lever.joint_ids
    differing = {
        j for j in lever.joint_directions if trigger.joint_directions[j] != lever.joint_directions[j]
    }
    assert differing == {"gripper"}
    # The lever pulls the servo one way, the trigger the other.
    assert trigger.joint_directions["gripper"] > 0 > lever.joint_directions["gripper"]


def test_trigger_gripper_half_pull_opens_the_jaw_and_the_far_stop_clamps():
    """Raw 0 (the calibration stop) is the jaw at zero; HALF the pull is the jaw fully open,
    reached within a degree without relying on the clamp; the rest of the pull, down to the
    far hard stop, stays clamped there and never unwraps onto the other 360 deg branch."""
    cfg = RebotArm102LeaderMakerTriggerTeleopConfig(port="/dev/null")
    lo, hi = cfg.joint_ranges["gripper"]
    assert _action_for_raw(cfg, 0.0)["gripper.pos"] == pytest.approx(hi)
    raw_at_half = MAKER_TRIGGER_GRIPPER_USABLE_TRAVEL_DEG * cfg.joint_directions["gripper"]
    assert lo - 1.0 <= raw_at_half <= lo + 1.0
    assert _action_for_raw(cfg, MAKER_TRIGGER_GRIPPER_USABLE_TRAVEL_DEG)["gripper.pos"] == pytest.approx(
        lo, abs=1.0
    )
    assert _action_for_raw(cfg, MAKER_TRIGGER_GRIPPER_TRAVEL_DEG)["gripper.pos"] == pytest.approx(lo)
    # The far stop sits inside the unwrap window with margin (no branch flip).
    center = (lo + hi) / 2 / cfg.joint_directions["gripper"]
    window_low = center - 180
    assert window_low + 30 < MAKER_TRIGGER_GRIPPER_TRAVEL_DEG


def test_trigger_gripper_is_monotonic_across_the_pull_and_never_wraps():
    """The lever factor snapped the jaw closed-to-open near raw -150; the trigger factor must
    not, anywhere between the two stops (and a little past them)."""
    cfg = RebotArm102LeaderMakerTriggerTeleopConfig(port="/dev/null")
    raws = [5.0, 0.0, -20.0, -60.0, -100.0, -150.0, -160.0, -186.8, -195.0]
    values = [_action_for_raw(cfg, r)["gripper.pos"] for r in raws]
    assert values == sorted(values, reverse=True), list(zip(raws, values, strict=True))
    lo, hi = cfg.joint_ranges["gripper"]
    assert all(lo <= v <= hi for v in values)


def test_trigger_types_are_registered_and_build_the_shared_driver():
    assert TeleoperatorConfig.get_choice_class("rebot_102_leader_maker_trigger") is (
        RebotArm102LeaderMakerTriggerTeleopConfig
    )
    assert TeleoperatorConfig.get_choice_class("bi_rebot_102_leader_maker_trigger") is (
        BiRebot102LeaderMakerTriggerConfig
    )
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        single = make_teleoperator_from_config(RebotArm102LeaderMakerTriggerTeleopConfig(port="/dev/null"))
        bi = make_teleoperator_from_config(
            draccus.parse(
                BiRebot102LeaderMakerTriggerConfig,
                args=["--left_arm_config.port=/dev/a", "--right_arm_config.port=/dev/b"],
            )
        )
    assert isinstance(single, RebotArm102Leader)
    assert isinstance(bi, BiRebot102Leader)
    expected = RebotArm102LeaderMakerTriggerConfig(port="x").joint_directions
    assert single.config.joint_directions == expected
    assert bi.left_arm.config.joint_directions == expected
    assert bi.right_arm.config.joint_directions == expected
