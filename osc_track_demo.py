"""OSC absolute-pose tracking demo.

    uv run python osc_track_demo.py                 # position targets, gripper-down
    uv run python osc_track_demo.py --rand-rot      # also randomize target orientation
    uv run python osc_track_demo.py --weld          # keep the grasped peg

Samples a random workspace pose every --hold seconds and lets the (fixed) OSC
drive the end-effector there. Draws the TARGET pose and the EE pose as RGB axes
(x=red, y=green, z=blue; the axes origin is the position). Watch the EE axes
snap onto the target axes — that's the absolute base-frame OSC tracking.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import types

import numpy as np
import jax.numpy as jnp
import warp as wp

import newton
from controllers import OSCController, rotvec_to_mat
from peg_env import PegEnv
import peg_scene_newton as scene

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hold", type=float, default=2.0, help="seconds per target")
ap.add_argument("--weld", action="store_true", help="keep the grasped peg")
ap.add_argument("--rand-rot", action="store_true", help="randomize target orientation too")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
CENTER = np.array([0.5, 0.0, 0.35], np.float32)      # workspace box center (reachable, in front of base)
HALF = np.array([0.11, 0.15, 0.11], np.float32)      # half-extents

ctrl = OSCController()
env = PegEnv(controller=ctrl, episode_length=10**9, weld=args.weld)
env.reset()
# Drive the OSC target directly (bypass the action->target mapping) so we can
# command arbitrary absolute workspace poses for the demo.
ctrl.set_action = types.MethodType(lambda self, env, action: None, ctrl)
nominal = np.array(ctrl.nominal_R)
dt = scene.SUBSTEPS * scene.SIM_DT
hold_steps = int(round(args.hold / dt))
zero = np.zeros(env.act_dim, np.float32)


def new_target():
    p = (CENTER + (rng.random(3) * 2 - 1) * HALF).astype(np.float32)
    R = nominal
    if args.rand_rot:
        rv = ((rng.random(3) * 2 - 1) * 0.4).astype(np.float32)   # +-0.4 rad about each axis
        R = (np.array(rotvec_to_mat(jnp.asarray(rv)[None])[0]) @ nominal).astype(np.float32)
    ctrl.target_pos = jnp.asarray(p[None])
    ctrl.target_R = jnp.asarray(R)
    return p, R


_AXES = ((0, (1.0, 0.2, 0.2)), (1, (0.2, 1.0, 0.2)), (2, (0.35, 0.5, 1.0)))  # x=red, y=green, z=blue


def draw_pose(viewer, name, pos, R, L):
    """Draw a pose as 3 RGB axis lines (origin = position)."""
    p = np.asarray(pos, np.float32).reshape(3)
    Rf = np.asarray(R, np.float32).reshape(3, 3)
    for ax, col in _AXES:
        s = wp.array(p.reshape(1, 3), dtype=wp.vec3)
        e = wp.array((p + L * Rf[:, ax]).reshape(1, 3), dtype=wp.vec3)
        viewer.log_lines(f"{name}_{ax}", s, e, col)


target_pos, target_R = new_target()

viewer = newton.viewer.ViewerGL(paused=False)
viewer.set_model(env.model)
viewer.set_camera(pos=wp.vec3(1.5, -1.1, 0.9), pitch=-22.0, yaw=130.0)
print(f"[demo] OSC tracking random workspace poses every {args.hold}s. "
      f"Target = long RGB axes, EE = short RGB axes. Close the window to stop.")

i = 0
t = 0.0
while viewer.is_running():
    if viewer.should_step():
        if i % hold_steps == 0:
            target_pos, target_R = new_target()
            print(f"[t={i * dt:6.1f}s] new target pos={np.round(target_pos, 3)}  "
                  f"err={ctrl.last_pos_err * 1000:.1f}mm")
        env.step(zero)
        i += 1
        t += 1 / 60
    viewer.begin_frame(t)
    viewer.log_state(env.state_0)
    draw_pose(viewer, "target", target_pos, target_R, 0.09)       # target: long axes
    if ctrl.last_ft_R is not None:
        draw_pose(viewer, "ee", ctrl.last_ft_pos, ctrl.last_ft_R, 0.05)   # EE: short axes
    viewer.end_frame()
viewer.close()
