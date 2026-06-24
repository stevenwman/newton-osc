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
import itertools

import numpy as np
import warp as wp

import newton
import controllers as C
from controllers import JointPositionController, OSCController
from peg_env import PegEnv


def _box_edges(center, half):
    """12 wireframe edges of an axis-aligned box (center ± half) as (starts, ends)."""
    c = np.asarray(center, np.float32)
    h = np.asarray(half, np.float32)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)), np.float32)  # (8,3)
    corners = c + signs * h
    s, e = [], []
    for a in range(8):
        for b in range(a + 1, 8):
            if int(np.sum(signs[a] != signs[b])) == 1:        # neighbours differ in one axis
                s.append(corners[a]); e.append(corners[b])
    return np.asarray(s, np.float32), np.asarray(e, np.float32)


def _draw_osc_debug(viewer, env):
    """Clip box (green), EE point fed to OSC (yellow), OSC target (red), bore-top anchor (blue)."""
    anchor = np.asarray(env.hole_pos, np.float32) + np.array([0.0, 0.0, C.ASSET_HEIGHT], np.float32)
    bs, be = _box_edges(anchor, np.asarray(env.controller.pos_bounds, np.float32))
    viewer.log_lines("clip_box", wp.array(bs, dtype=wp.vec3), wp.array(be, dtype=wp.vec3), (0.1, 0.9, 0.1))
    pts, cols = [], []
    if env.controller.last_ft_pos is not None:
        pts.append(env.controller.last_ft_pos); cols.append((1.0, 1.0, 0.0))     # EE input -> yellow
    pts.append(np.asarray(env.controller.target_pos[0], np.float32)); cols.append((1.0, 0.0, 0.0))  # target -> red
    pts.append(anchor); cols.append((0.0, 0.6, 1.0))                              # bore-top anchor -> blue
    viewer.log_points("osc_dbg", wp.array(np.asarray(pts, np.float32), dtype=wp.vec3),
                      0.01, wp.array(np.asarray(cols, np.float32), dtype=wp.vec3))

ap = argparse.ArgumentParser()
ap.add_argument("--controller", choices=["osc", "jointpos"], default="osc")
ap.add_argument("--action", choices=["hold", "down", "up"], default="hold")
ap.add_argument("--no-weld", dest="weld", action="store_false")
ap.add_argument("--no-rot", dest="rot", action="store_false", help="OSC: drop orientation control")
ap.add_argument("--no-null", dest="null", action="store_false", help="OSC: drop nullspace term")
ap.add_argument("--no-bias", dest="bias", action="store_false", help="OSC: drop gravity comp")
ap.add_argument("--box-scale", type=float, default=1.0, help="OSC: scale the clip-box half-extents")
ap.add_argument("--hide-hole", action="store_true", help="move the bore far away (observe weld without contact)")
ap.add_argument("--weld-stiff", action="store_true",
                help="match jax_rl weld stiffness (solref 0.001 1, solimp 0.999 0.9999 0.001) vs Newton's soft default")
ap.add_argument("--freeze-target", action="store_true",
                help="OSC: freeze the target at the reset pose (no clip/EMA) — clean STATIC hold to observe the weld")
args = ap.parse_args()

ctrl = OSCController() if args.controller == "osc" else JointPositionController()
if args.controller == "osc":
    ctrl.use_rot = args.rot
    ctrl.use_null = args.null
    ctrl.use_bias = args.bias
    ctrl.pos_bounds = C.POS_BOUNDS * args.box_scale
env = PegEnv(controller=ctrl, episode_length=10**9, weld=args.weld)
if args.weld_stiff and args.weld:
    # Match jax_rl's deliberately-stiffened grasp weld. Newton re-added the weld
    # with its SOFT default (solref=0.02); jax_rl uses solref=0.001/solimp dimp->1
    # so the weld behaves rigidly. Weld = equality index 1 (index 0 = finger coupling).
    M = env.solver.mjw_model
    sr = M.eq_solref.numpy().copy(); sr[..., 1, :] = (0.001, 1.0); M.eq_solref.assign(sr)
    si = M.eq_solimp.numpy().copy(); si[..., 1, :3] = (0.999, 0.9999, 0.001); M.eq_solimp.assign(si)
    print("[view] weld stiffened to jax_rl spec (solref=0.001 1, solimp=0.999 0.9999 0.001)")
env.reset()
if args.hide_hole:
    # Move the bore BODY away (no contact) but KEEP the OSC clip anchor at the real
    # bore — otherwise the anchor follows the bore to z=-5 and the OSC chases the EE
    # to an unreachable point near the base.
    real_anchor = env.hole_pos.copy()
    env._set_hole(np.array([0.6, 0.0, -5.0], np.float32))
    env.hole_pos = real_anchor
    print("[view] bore body moved away (no contact); OSC anchor kept at", real_anchor)
if args.controller == "osc" and args.freeze_target:
    import types
    env.controller.set_action = types.MethodType(lambda self, env, action: None, env.controller)
    print("[view] target FROZEN at reset pose (no clip/EMA) — static hold")

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
    if args.controller == "osc":
        _draw_osc_debug(viewer, env)
    viewer.end_frame()
viewer.close()
