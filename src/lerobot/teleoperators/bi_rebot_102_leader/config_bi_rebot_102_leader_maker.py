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

from dataclasses import dataclass

from ..config import TeleoperatorConfig
from ..rebot_102_leader import RebotArm102LeaderMakerConfig


@TeleoperatorConfig.register_subclass("bi_rebot_102_leader_maker")
@dataclass
class BiRebot102LeaderMakerConfig(TeleoperatorConfig):
    """Two reBot Arm 102 leaders driving a bimanual Maker arm follower.

    Same hardware and driver as `bi_rebot_102_leader`; each arm just defaults to the Maker joint
    mapping (`RebotArm102LeaderMakerConfig`) so the CLI only needs the two ports:

        --teleop.type=bi_rebot_102_leader_maker \\
        --teleop.left_arm_config.port=/dev/ttyUSB0 \\
        --teleop.right_arm_config.port=/dev/ttyUSB1
    """

    left_arm_config: RebotArm102LeaderMakerConfig
    right_arm_config: RebotArm102LeaderMakerConfig
