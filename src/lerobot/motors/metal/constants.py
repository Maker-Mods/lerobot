"""Metal arm constants (CAN ids, motor types, joint limits). Pure data; no hardware, no ROS."""

# Joint names follow the reBot convention (shoulder_pan / shoulder_lift / elbow_flex /
# wrist_flex / wrist_yaw / wrist_roll) so leader-follower key matching is automatic and
# config files can be shared across arm types.
METAL_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
METAL_GRIPPER_NAME = "gripper"
GRIPPER_MAX_MM = 80.0

# name -> (send_id, recv_id, motor_type_str)
METAL_MOTOR_CONFIG: dict[str, tuple[int, int, str]] = {
    "shoulder_pan":  (0x01, 0x11, "metal_jlo"),
    "shoulder_lift": (0x02, 0x12, "metal_j2"),
    "elbow_flex":    (0x03, 0x13, "metal_jlo"),
    "wrist_flex":    (0x04, 0x14, "metal_jlo"),
    "wrist_yaw":     (0x05, 0x15, "metal_jhi"),
    "wrist_roll":    (0x06, 0x16, "metal_jhi"),
    "gripper":       (0x07, 0x17, "metal_jhi"),
}

# degrees, from vendor motor_config.cpp position_min/max
METAL_JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan":  (-160.0, 160.0),
    "shoulder_lift": (-180.0, 0.0),
    "elbow_flex":    (0.0, 180.0),
    "wrist_flex":    (-123.0, 81.0),
    "wrist_yaw":     (-85.0, 85.0),
    "wrist_roll":    (-145.0, 145.0),
}

# Follower MIT gains (kp, kd) from vendor motor_config.cpp follow_mit_kp/follow_mit_kd.
# lerobot's MIT_KP_RANGE=(0,500) / MIT_KD_RANGE=(0,5) match the vendor exactly, so these
# copy over with identical meaning. Bus defaults (kp=10) are far too soft to hold the arm
# against gravity, so the follower sets these at connect for firm, responsive tracking.
METAL_FOLLOWER_GAINS: dict[str, tuple[float, float]] = {
    "shoulder_pan":  (200.0, 3.0),
    "shoulder_lift": (500.0, 5.0),
    "elbow_flex":    (400.0, 5.0),
    "wrist_flex":    (200.0, 2.0),
    "wrist_yaw":     (20.0, 0.1),
    "wrist_roll":    (20.0, 0.1),
    "gripper":       (20.0, 0.1),
}
