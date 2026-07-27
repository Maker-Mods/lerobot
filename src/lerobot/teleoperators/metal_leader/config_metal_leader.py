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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("metal_leader")
@dataclass
class MetalLeaderConfig(TeleoperatorConfig):
    """
    Configuration for the Metal arm leader teleoperator: 7 Damiao motors (6 joints + a permanent
    gripper) over classic CAN, driven via the stock `DamiaoMotorsBus`.

    Unlike OpenArms, the gripper is not optional (no `arm_end_type`): the leader always exposes all
    7 motors, all normalized in degrees. A background thread streams Pinocchio gravity-compensation
    MIT torque so the arm feels weightless while a human guides it; the gripper is left backdrivable
    (kp=0, torque=0) so it can be squeezed freely.
    """

    # CAN interface (e.g. "can1"). Linux: "can0", "can1", etc. The leader lives on the second bus.
    port: str = "can1"

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # Metal uses classic CAN @ 1 Mbps (not CAN FD)
    can_bitrate: int = 1_000_000
    use_can_fd: bool = False
    can_data_bitrate: int | None = None

    # Path to the URDF describing the arm. Defaults to the bundled metal_with_gripper.urdf
    # (resolved lazily in MetalLeader.__init__ if left empty).
    urdf_path: str = ""

    # Background gravity-compensation thread rate (Hz).
    gravity_hz: int = 200

    # MIT damping gain while gravity-compensated (kp is always 0 so the human can freely position
    # the arm; kd supplies velocity damping / feel). kd is also the brake against friction-
    # feedforward runaway — don't drive it to 0 while raising friction_scale unless you have tested
    # stability at your control rate. Accepts a single float (all joints) or a per-joint dict
    # {motor_name: kd}; motors absent from the dict get 0. The vendor uses kd=0 (uniform) + full
    # friction feedforward; its per-joint feel comes from the per-joint viscous coefficients.
    leader_kd: float | dict[str, float] = 0.0

    # Friction/coriolis feedforward: fed the measured joint velocity to cancel the arm's own
    # gearbox friction so the leader feels transparent (lighter to move). Accepts a single float
    # (all joints) or a per-joint dict {motor_name: scale}. 0 = gravity only. Higher = lighter, but
    # too high a joint RUNS AWAY. Per-joint because this arm's real friction differs per joint from
    # the vendor's viscous coefficients (tuned on hardware; the vendor's uniform 1.0 did not fit).
    use_velocity_feedforward: bool = True
    friction_scale: float | dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 1.4,
            "shoulder_lift": 3.3,
            "elbow_flex": 1.1,
            "wrist_flex": 0.7,
            "wrist_yaw": 0.3,
            "wrist_roll": 0.7,
        }
    )
    velocity_deadzone_rad_s: float = 0.05

    # Scales the gripper friction feedforward (vendor GripperTorqueCompensation) so the leader
    # gripper is easy to squeeze. 0 disables it (gripper left at torque=0). Tune on hardware.
    gripper_friction_scale: float = 1.0

    # On disconnect (teleop end), stop gravity compensation and hold the current pose with these
    # MIT gains so the arm freezes in place (stays up, no longer weightless) instead of drifting.
    # Set hold_kp_on_disconnect=0 to leave the arm limp/backdrivable instead.
    hold_kp_on_disconnect: float = 50.0
    hold_kd_on_disconnect: float = 1.0
