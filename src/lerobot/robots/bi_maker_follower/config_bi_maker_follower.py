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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..maker_follower import MakerFollowerConfigBase


@RobotConfig.register_subclass("bi_maker_follower")
@dataclass
class BiMakerFollowerConfig(RobotConfig):
    """Configuration for a bimanual Maker follower on two CAN buses."""

    # Using the unregistered base avoids a recursive draccus choice tree.
    left_arm_config: MakerFollowerConfigBase
    right_arm_config: MakerFollowerConfigBase

    # Keep top-level camera names stable so datasets can address a shared scene view
    # without assigning that camera to either arm.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
