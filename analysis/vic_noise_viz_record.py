"""Record square-VIC trials under obs noise, visualizing the policy's NOISED
perception: RGB-XYZ axis triads at the perceived peg pose + perceived goal pose
(un-normalized from the SAME noised obs fed to the actor, so the wobble is the real
corrupted perception), the compliance ellipsoid at the TRUE EE, and a translucent
hole. One mp4 per trial, filename tagged with controller / sigma / SUCCESS|FAIL.

    PYTHONPATH=. DISPLAY=:1 .venv/bin/python vic_noise_viz_record.py \
        --gain-mode axis --ckpt runs/sq_vic_axis/best_actor.pkl --trials 1 --out runs/sq_vic_noiseviz
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import argparse, pickle, time
import numpy as np, jax, jax.numpy as jnp, optax, warp as wp
import newton, imageio.v2 as iio
from jax_rl.algos.ppo import PPO
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.utils import normalization as nrm
from controllers import OSCController
from peg_env_square import PegEnv, PEG_BODY_LOCAL
from vic_noise_eval import field_idx

ap = argparse.ArgumentParser()
ap.add_argument("--gain-mode", choices=["fixed", "single", "axis"], required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sigma", type=float, default=2.0)
ap.add_argument("--target", default="both")
ap.add_argument("--trials", type=int, default=1)
ap.add_argument("--episode-length", type=int, default=450)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--axis-len", type=float, default=0.04)
ap.add_argument("--r0", type=float, default=0.05, help="compliance ellipsoid base radius")
args = ap.parse_args()


def quat_xyzw_to_R(q):
    q = q / (np.linalg.norm(q) + 1e-9)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float32)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def axes_lines(pos, R, L):
    """3 RGB axis segments (x=red, y=green, z=blue) at pose (pos,R)."""
    starts = np.tile(pos, (3, 1)).astype(np.float32)
    ends = (pos + (R * L).T).astype(np.float32)              # columns of R are axes
    cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    return starts, ends, cols


def sphere_wire(n_lon=14, n_lat=7, n_seg=20):
    S, E = [], []
    for j in range(n_lon):
        phi = 2 * np.pi * j / n_lon; th = np.linspace(0, np.pi, n_seg + 1)
        p = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], 1)
        S.append(p[:-1]); E.append(p[1:])
    for i in range(1, n_lat):
        th = np.pi * i / n_lat; ph = np.linspace(0, 2 * np.pi, n_seg + 1)
        p = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.full_like(ph, np.cos(th))], 1)
        S.append(p[:-1]); E.append(p[1:])
    return np.concatenate(S).astype(np.float32), np.concatenate(E).astype(np.float32)


WS, WE = sphere_wire()

# Hole prisms (local hole_base frame), from square_insert/scene.xml: (center, half).
HOLE_BOXES = [
    ((-0.0198, 0.0302, 0.015), (0.0302, 0.0198, 0.020)),   # slab_top
    ((0.0302, 0.0198, 0.015), (0.0198, 0.0302, 0.020)),    # slab_right
    ((0.0198, -0.0302, 0.015), (0.0302, 0.0198, 0.020)),   # slab_bottom
    ((-0.0302, -0.0198, 0.015), (0.0198, 0.0302, 0.020)),  # slab_left
    ((0.0, 0.0, -0.003), (0.012, 0.012, 0.001)),           # bore_floor
]
_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
          if np.sum(np.abs(_CORNERS[a] - _CORNERS[b])) == 2]   # 12 edges (differ in 1 axis)


def hole_wire(pos, R):
    """Wireframe edges of the 5 hole prisms, transformed to world by (pos,R)."""
    S, E = [], []
    for c, h in HOLE_BOXES:
        corners = (np.array(c) + _CORNERS * np.array(h)) @ R.T + pos   # (8,3) world
        for a, b in _EDGES:
            S.append(corners[a]); E.append(corners[b])
    return np.array(S, np.float32), np.array(E, np.float32)

ctrl = OSCController(); ctrl.action_mode = "delta"; ctrl.gain_mode = args.gain_mode
env = PegEnv(controller=ctrl, episode_length=args.episode_length, weld=True, num_envs=1)
A = env.act_dim
idx = field_idx(A)[args.target]
mask = np.zeros((1, env.obs_dim), np.float32); mask[0, idx] = 1.0
mask = jnp.asarray(mask)
GY = env.obs_dim - 2                                          # goal_yaw [cos,sin] start

cfg = PPOConfig(policy_hidden_dim=(256, 256), value_hidden_dim=(256, 256),
                activation="swish", squash=True, state_dependent_std=False)
cfg.num_envs = 1; cfg.minibatch_size = 1
cfg.encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=cfg.policy_hidden_dim, activation=cfg.activation)
cfg.critic_encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=cfg.value_hidden_dim, activation=cfg.activation)
cfg.policy_head = PolicyHeadConfig(action_dim=A, squash=True, state_dependent_std=False)
ppo = PPO(cfg, env.obs_dim, A, optax.adam(3e-4), optax.adam(3e-4))
with open(args.ckpt, "rb") as f:
    loaded = pickle.load(f)
actor_params, norm_state = loaded["actor_params"], loaded["norm_state"]

# See-through hole: clear the VISIBLE flag on the hole shapes (keep COLLIDE for
# physics) so the solid slab isn't rendered; we draw it as a wireframe instead.
hole_b = [l.split('/')[-1] for l in env.model.body_label].index("hole_base")
sb = env.model.shape_body.numpy()
hole_shapes = [i for i in range(len(sb)) if sb[i] == hole_b]
flags = env.model.shape_flags.numpy()
flags[hole_shapes] &= ~int(newton.ShapeFlags.VISIBLE)
env.model.shape_flags.assign(flags)
print(f"[viz] hid {len(hole_shapes)} solid hole shapes (-> wireframe); {len(idx)} noised obs dims")

viewer = newton.viewer.ViewerGL(headless=True)
viewer.set_model(env.model)
viewer.show_collision = False                                # don't render collide-only (hidden) shapes
viewer.set_camera(pos=wp.vec3(0.78, -0.42, 0.30), pitch=-26.0, yaw=118.0)
key = jax.random.PRNGKey(0)
os.makedirs(args.out, exist_ok=True)

for trial in range(args.trials):
    obs = env.reset(); frames = []; t = 0.0; succ = False
    t_step = t_render = 0.0
    for i in range(args.episode_length):
        normed = nrm.normalize(norm_state, jnp.asarray(obs))
        key, k = jax.random.split(key)
        noised = normed + args.sigma * mask * jax.random.normal(k, normed.shape)
        a = ppo.select_action_eval(actor_params, noised)
        # perceived (noised) world poses via un-normalize of the SAME vector
        raw = np.asarray(nrm.unnormalize(norm_state, noised))[0]
        hole_n = raw[21:24]
        peg_pos_n = hole_n + raw[14:17]                      # peg_rel = peg - hole
        peg_R_n = quat_xyzw_to_R(raw[17:21])
        yaw_n = float(np.arctan2(raw[GY + 1], raw[GY]))
        ps, pe, pc = axes_lines(peg_pos_n, peg_R_n, args.axis_len)
        gs, ge, gc = axes_lines(hole_n, Rz(yaw_n), args.axis_len)
        # compliance ellipsoid at TRUE ee (unnoised)
        ft = np.asarray(ctrl._fingertip(env.solver.mjw_data)[0][0])
        kp_s = np.asarray(ctrl.kp_scale[0])
        radii = np.clip(args.r0 / np.maximum(kp_s[:3], 1e-3), 0.01, 0.15)
        es = (WS * radii + ft).astype(np.float32); ee = (WE * radii + ft).astype(np.float32)

        ts = time.time()
        obs, rew, done, info = env.step(np.asarray(a))
        t_step += time.time() - ts
        succ = succ or bool(np.asarray(info["success"])[0])

        # hole wireframe at current hole world pose (state_0 body_q: pos + quat xyzw)
        hb = np.asarray(wp.to_jax(env.state_0.body_q).reshape(-1, 7)[hole_b])
        hws, hwe = hole_wire(hb[:3], quat_xyzw_to_R(hb[3:7]))

        tr = time.time()
        viewer.begin_frame(t); viewer.log_state(env.state_0)
        viewer.log_lines("hole_wire", wp.array(hws, dtype=wp.vec3), wp.array(hwe, dtype=wp.vec3),
                         colors=(0.6, 0.6, 0.65), width=0.0015)
        viewer.log_lines("perceived_peg", wp.array(ps, dtype=wp.vec3), wp.array(pe, dtype=wp.vec3),
                         colors=wp.array(pc, dtype=wp.vec3), width=0.008)
        viewer.log_lines("perceived_goal", wp.array(gs, dtype=wp.vec3), wp.array(ge, dtype=wp.vec3),
                         colors=wp.array(gc, dtype=wp.vec3), width=0.008)
        viewer.log_lines("compliance", wp.array(es, dtype=wp.vec3), wp.array(ee, dtype=wp.vec3),
                         colors=(0.2, 0.85, 1.0), width=0.0015)
        viewer.end_frame()
        img = np.asarray(viewer.get_frame().numpy())
        if img.dtype != np.uint8: img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[-1] == 4: img = img[..., :3]
        frames.append(img); t += 1.0 / args.fps; t_render += time.time() - tr
        if done: break

    tag = "SUCCESS" if succ else "FAIL"
    fn = f"{args.out}/{args.gain_mode}_{args.target}_s{args.sigma:g}_{tag}_t{trial}.mp4"
    iio.mimwrite(fn, frames, fps=args.fps, macro_block_size=2)
    print(f"[viz] trial {trial}: {tag}  saved {fn}  ({len(frames)}f)  "
          f"step={t_step:.1f}s render={t_render:.1f}s  -> render is "
          f"{100*t_render/(t_step+t_render):.0f}% of loop")
viewer.close()
