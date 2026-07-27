"""Diagnostic: leader gravity comp at 200 Hz with tunable MIT damping (kd) and friction
feedforward scale, to find a transparent hand feel.
Usage: python docs/metal/tests/leader_gravity_debug.py [can1] [kd] [friction_scale]
  e.g. python docs/metal/tests/leader_gravity_debug.py can1 0.15 0.6

- kd (0..5): our added velocity damping. Lower = freer to move, but too low = won't settle
  when tapped. Current production default is 0.3.
- friction_scale (0..~1.2): how much of the vendor viscous-friction/coriolis feedforward to add.
  0 = pure gravity (current behaviour). Higher = the arm feels lighter / more transparent, but
  too high a joint can RUN AWAY (self-accelerate) — back off immediately if that happens.
Sweep both to find "light to move + still settles + no runaway", then report the two values."""
import math
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.metal import METAL_MOTOR_CONFIG, METAL_JOINT_NAMES
from lerobot.motors.metal.gravity import MetalGravityModel

PORT = sys.argv[1] if len(sys.argv) > 1 else "can1"
KD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
# friction: a single number (all 6 joints) OR 6 comma-separated per-joint values
# e.g. "1.2,1.5,0.5,1.2,1.0,1.0"
_raw = sys.argv[3] if len(sys.argv) > 3 else "0"
FRICTION = [float(x) for x in _raw.split(",")] if "," in _raw else [float(_raw)] * 6
assert len(FRICTION) == 6, "friction must be 1 value or 6 comma-separated"
DEADZONE_RAD_S = 0.05
URDF = "src/lerobot/motors/metal/urdf/metal_with_gripper.urdf"

motors = {}
for name, (send, recv, typ) in METAL_MOTOR_CONFIG.items():
    m = Motor(send, typ, MotorNormMode.DEGREES)
    m.recv_id = recv
    m.motor_type_str = typ
    motors[name] = m

bus = DamiaoMotorsBus(port=PORT, motors=motors, use_can_fd=False, bitrate=1_000_000, can_interface="socketcan")
bus.connect()
bus.enable_torque()
gm = MetalGravityModel(URDF)
print(f"leader debug on {PORT}: kd={KD}, friction(per-joint)={FRICTION}, 200 Hz. "
      f"Move the arm by hand. Watch for runaway. Ctrl-C to stop.\n")
period = 1.0 / 200.0
n = 0
try:
    while True:
        t0 = time.perf_counter()
        st = bus.sync_read_all_states()
        q = [math.radians(st[m]["position"]) for m in METAL_JOINT_NAMES]
        tau_grav = gm.feedforward_torque(q, [0.0] * 6)          # gravity only
        if any(f > 0.0 for f in FRICTION):
            dq = []
            for m in METAL_JOINT_NAMES:
                v = math.radians(st[m]["velocity"])            # deg/s -> rad/s
                dq.append(0.0 if abs(v) < DEADZONE_RAD_S else v)
            tau_full = gm.feedforward_torque(q, dq)             # gravity + coriolis + friction
            tau = [tau_grav[i] + FRICTION[i] * (tau_full[i] - tau_grav[i]) for i in range(6)]
        else:
            tau = tau_grav
        cmds = {}
        for i, m in enumerate(METAL_JOINT_NAMES):
            cmds[m] = (0.0, KD, st[m]["position"], 0.0, tau[i])
        cmds["gripper"] = (0.0, KD, st["gripper"]["position"], 0.0, 0.0)
        bus.sync_write_mit(cmds)
        n += 1
        if n % 40 == 0:  # ~5 prints/sec
            print("  ".join(f"{m}:{st[m]['position']:+6.1f}deg tau={tau[i]:+5.2f}"
                            for i, m in enumerate(METAL_JOINT_NAMES)))
        dt = period - (time.perf_counter() - t0)
        if dt > 0:
            time.sleep(dt)
except KeyboardInterrupt:
    pass
finally:
    bus.disconnect(disable_torque=False)
    print("\nstopped.")
