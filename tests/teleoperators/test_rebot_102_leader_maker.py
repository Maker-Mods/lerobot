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
)
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.rebot_102_leader import (
    RebotArm102Leader,
    RebotArm102LeaderConfig,
    RebotArm102LeaderMakerConfig,
    RebotArm102LeaderMakerTeleopConfig,
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
