"""Tune the REAL MetalLeader (the exact code path teleop uses) with per-joint friction from the
CLI, so what you feel here is what teleop gives. The standalone gravity debug loop can feel
different (timing/threading), so tune here.

Usage:
  python docs/metal/tests/leader_tune.py "f1,f2,f3,f4,f5,f6" [gripper_scale] [can1]
  e.g. python docs/metal/tests/leader_tune.py "0.9,1.7,0.8,0.7,0.6,0.6"

Move the arm by hand. Dial each friction DOWN until the joint is transparent but NOT
over-compensated (no self-motion / push). gripper_scale defaults to 0 (gripper off) so you can
isolate the arm. Report the final 6 values and I'll bake them into MetalLeaderConfig."""
import sys
import time

from lerobot.motors.metal import METAL_JOINT_NAMES
from lerobot.teleoperators.metal_leader.config_metal_leader import MetalLeaderConfig
from lerobot.teleoperators.metal_leader.metal_leader import MetalLeader

fr = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1.0] * 6
assert len(fr) == 6, "need 6 comma-separated friction values"
grip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
port = sys.argv[3] if len(sys.argv) > 3 else "can1"

cfg = MetalLeaderConfig(
    port=port,
    leader_kd=0.0,
    friction_scale={METAL_JOINT_NAMES[i]: fr[i] for i in range(6)},
    gripper_friction_scale=grip,
)
t = MetalLeader(cfg)
t.connect()
print(f"REAL MetalLeader on {port}: friction={fr}, gripper={grip}. Move by hand. Ctrl-C to stop.")
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    t.disconnect()
    print("\nstopped.")
