"""Interactive, PAUSED-start viewer for the peg env. You drive it.

    uv run python env_view.py                       # OSC, hold (zero action), paused
    uv run python env_view.py --action down         # command -z each step
    uv run python env_view.py --action up
    uv run python env_view.py --controller jointpos
    uv run python env_view.py --no-weld             # peg unwelded (falls away)

Starts PAUSED. Use the viewer's play/step controls to advance. Each control
frame it prints hand_z / peg_z / OSC target so you can report what happens.
"""
import argparse

import numpy as np
import warp as wp

import newton
from controllers import JointPositionController, OSCController
from peg_env import PegEnv

ap = argparse.ArgumentParser()
ap.add_argument("--controller", choices=["osc", "jointpos"], default="osc")
ap.add_argument("--action", choices=["hold", "down", "up"], default="hold")
ap.add_argument("--no-weld", dest="weld", action="store_false")
ap.add_argument("--no-rot", dest="rot", action="store_false", help="OSC: drop orientation control")
ap.add_argument("--no-null", dest="null", action="store_false", help="OSC: drop nullspace term")
ap.add_argument("--no-bias", dest="bias", action="store_false", help="OSC: drop gravity comp")
args = ap.parse_args()

ctrl = OSCController() if args.controller == "osc" else JointPositionController()
if args.controller == "osc":
    ctrl.use_rot = args.rot
    ctrl.use_null = args.null
    ctrl.use_bias = args.bias
env = PegEnv(controller=ctrl, episode_length=10**9, weld=args.weld)
env.reset()

action = np.zeros(env.act_dim, np.float32)
if args.action == "down":
    action[2] = -1.0
elif args.action == "up":
    action[2] = 1.0

HAND, PEG = 8, 12

viewer = newton.viewer.ViewerGL(paused=True)          # <-- start paused
viewer.set_model(env.model)
viewer.set_camera(pos=wp.vec3(1.3, -0.9, 0.7), pitch=-20.0, yaw=130.0)

print(f"[view] controller={args.controller} action={args.action} weld={args.weld} "
      f"act_dim={env.act_dim} — PAUSED; use GUI play/step to advance.")
step_i = 0
t = 0.0
while viewer.is_running():
    if viewer.should_step():                          # advances only when you step in the GUI
        env.step(action)
        bq = env.state_0.body_q.numpy()
        msg = f"[step {step_i:4d}] hand_z={bq[HAND][2]:.3f} peg_z={bq[PEG][2]:.3f}"
        if args.controller == "osc":
            tp = np.asarray(env.controller.target_pos[0])
            msg += (f"  osc_target_z={tp[2]:.3f}"
                    f"  pos_err={env.controller.last_pos_err:.3f}"
                    f"  rot_err={env.controller.last_rot_err:.3f}")
        print(msg)
        step_i += 1
        t += 1 / 60
    viewer.begin_frame(t)
    viewer.log_state(env.state_0)
    viewer.end_frame()
viewer.close()
