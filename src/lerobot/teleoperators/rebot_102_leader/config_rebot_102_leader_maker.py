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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig
from .config_rebot_102_leader import RebotArm102LeaderConfig


@dataclass
class RebotArm102LeaderMakerConfig(RebotArm102LeaderConfig):
    """The reBot Arm 102 / Star Arm 102 leader mapped onto the Maker arm follower.

    Same physical leader as `RebotArm102LeaderConfig`, same servo bus, same `joint_ids`. Only the
    joint mapping differs. The parent's defaults describe the reBot B601 follower; pointed at a
    Maker arm unchanged, most joints run the wrong way or saturate against the follower's soft
    limits and stop moving while teleop keeps reporting a healthy loop.

    `joint_ranges` is `MakerFollowerConfig.joint_limits` copied verbatim, so the leader can never
    emit a target outside the follower's envelope. `test_maker_ranges_match_follower_limits`
    enforces that identity. Update the follower's limits and that test tells you to update these.
    """

    # Sign flips travel; magnitude rescales a leader joint onto a follower joint whose travel
    # differs. These are the `direction * scale` factors from the vendor's hardware-captured
    # Star-to-Maker mapping (maker-arm SDK `profiles/star_to_maker_v1.json`, arm #02,
    # 2026-08-20): each Star joint was swept stop to stop against the Maker joint it drives.
    # The mapping is linear, so the factor carries over unchanged; the zero offset between the two
    # arms' calibration poses is absorbed by `joint_ranges` clamping, see the Maker docs.
    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 1.446501,
            "shoulder_lift": -0.859866,
            "elbow_flex": -0.857332,
            "wrist_flex": -0.85871,
            "wrist_yaw": -1.322941,
            "wrist_roll": 0.954213,
            "gripper": -1.983613,
        }
    )

    # Mirrors MakerFollowerConfig.joint_limits so leader output is bounded by the follower's own
    # soft limits before it ever reaches the bus. These bounds do more than clip:
    # `_round_to_valid_range` centres the multi-turn unwrap window on (min+max)/2, so a range
    # borrowed from the wrong follower can land a joint on the wrong 360 deg branch.
    joint_ranges: dict[str, list[int]] = field(
        default_factory=lambda: {
            "shoulder_pan": [-158, 156],
            "shoulder_lift": [-175, -3],
            "elbow_flex": [2, 236],
            "wrist_flex": [-65, 104],
            "wrist_yaw": [-97, 78],
            "wrist_roll": [-153, 152],
            "gripper": [-120, -2],
        }
    )


@TeleoperatorConfig.register_subclass("rebot_102_leader_maker")
@dataclass
class RebotArm102LeaderMakerTeleopConfig(TeleoperatorConfig, RebotArm102LeaderMakerConfig):
    """Registered configuration for the reBot Arm 102 leader driving a Maker arm follower."""

    pass


# Gripper factor for the trigger-style Star leader. Measured 2026-09-10 on the MakerMods trigger
# unit (servo id 6, multi-turn counter reset first, read_raw_angle): the trigger has a hard stop
# at both ends, 0.0 deg at the calibration stop and -186.8 deg at the far stop, repeatable to
# 0.1 deg across two runs. The follower's jaw runs 120 deg (-2.5 .. -120.1), so the factor is
# 120 / 186.8, and its sign is positive because the trigger travels the opposite way to the lever.
MAKER_TRIGGER_GRIPPER_TRAVEL_DEG = -186.8
MAKER_TRIGGER_GRIPPER_DIRECTION = round(120.0 / abs(MAKER_TRIGGER_GRIPPER_TRAVEL_DEG), 4)  # 0.6424


@dataclass
class RebotArm102LeaderMakerTriggerConfig(RebotArm102LeaderMakerConfig):
    """The Maker preset for a Star Arm 102 leader fitted with the MakerMods TRIGGER gripper.

    The stock Star Arm 102 works its gripper servo with a left-right lever that turns it about
    60 deg; MakerMods' revision replaces that with a trigger that turns the same servo about 187
    deg the other way, between two hard stops. Every other joint is untouched, so this preset
    inherits `RebotArm102LeaderMakerConfig` wholesale and overrides ONE entry, the gripper's
    `joint_directions` factor. `joint_ranges` stays the follower's envelope, unchanged.

    With the lever factor a trigger leader barely moves the jaw: its travel lies outside the
    band the lever mapped, so the jaw sits clamped at one limit, and the only motion is a snap
    from fully closed to fully open as the servo crosses the edge of the multi-turn unwrap window
    near -150 deg. The trigger factor puts the window at about -275 .. +85 raw degrees, 85 deg
    clear of either stop.

    Calibrate the trigger at the stop that matches the follower's zero-pose jaw state: the mapping
    scales, it does not offset, so raw 0 must be the jaw's zero-pose position on both arms.
    """

    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            **RebotArm102LeaderMakerConfig(port="").joint_directions,
            "gripper": MAKER_TRIGGER_GRIPPER_DIRECTION,
        }
    )


@TeleoperatorConfig.register_subclass("rebot_102_leader_maker_trigger")
@dataclass
class RebotArm102LeaderMakerTriggerTeleopConfig(TeleoperatorConfig, RebotArm102LeaderMakerTriggerConfig):
    """Registered configuration for a trigger-gripper reBot Arm 102 leader driving a Maker arm."""

    pass
