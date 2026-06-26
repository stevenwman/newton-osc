"""Record a square-VIC PPO policy with a live COMPLIANCE ELLIPSOID at the EE — a
transparent wireframe blob whose radii ∝ 1/Kp (bulges in the soft/compliant
directions, pinches where stiff), so the learned impedance shape is visible.
Also logs kp_scale/zeta/force to traj.npz (reuse vic_plot.py).

    PYTHONPATH=. DISPLAY=:1 .venv/bin/python vic_ellipsoid_record.py \
        --gain-mode axis --ckpt runs/sq_vic_axis/best_actor.pkl --out runs/sq_vic_axis
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")

import argparse, pickle
import numpy as np, jax, jax.numpy as jnp, optax, warp as wp
import newton, imageio.v2 as iio
from jax_rl.algos.ppo import PPO
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.utils import normalization as nrm
from controllers import OSCController, KP_TASK
from peg_env_square import PegEnv, PEG_BODY_LOCAL

ap = argparse.ArgumentParser()
ap.add_argument("--gain-mode", choices=["fixed", "single", "axis"], required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--episode-length", type=int, default=450)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--r0", type=float, default=0.05, help="ellipsoid base radius (m) at Kp=base")
args = ap.parse_args()


def sphere_wire(n_lon=14, n_lat=7, n_seg=20):
    """Unit-sphere wireframe as line segments (meridians + parallels). Returns
    (starts, ends) float arrays (M,3) of unit-sphere points."""
    S, E = [], []
    for j in range(n_lon):                              # meridians
        phi = 2 * np.pi * j / n_lon
        th = np.linspace(0, np.pi, n_seg + 1)
        p = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], 1)
        S.append(p[:-1]); E.append(p[1:])
    for i in range(1, n_lat):                           # parallels
        th = np.pi * i / n_lat
        ph = np.linspace(0, 2 * np.pi, n_seg + 1)
        p = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.full_like(ph, np.cos(th))], 1)
        S.append(p[:-1]); E.append(p[1:])
    return np.concatenate(S, 0).astype(np.float32), np.concatenate(E, 0).astype(np.float32)


WS, WE = sphere_wire()                                   # unit wireframe segments

ctrl = OSCController(); ctrl.action_mode = "delta"; ctrl.gain_mode = args.gain_mode
env = PegEnv(controller=ctrl, episode_length=args.episode_length, weld=True, num_envs=1)

cfg = PPOConfig(policy_hidden_dim=(256, 256), value_hidden_dim=(256, 256),
                activation="swish", squash=True, state_dependent_std=False)
cfg.num_envs = 1; cfg.minibatch_size = 1
cfg.encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=cfg.policy_hidden_dim, activation=cfg.activation)
cfg.critic_encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=cfg.value_hidden_dim, activation=cfg.activation)
cfg.policy_head = PolicyHeadConfig(action_dim=env.act_dim, squash=True, state_dependent_std=False)
ppo = PPO(cfg, env.obs_dim, env.act_dim, optax.adam(3e-4), optax.adam(3e-4))
with open(args.ckpt, "rb") as f:
    loaded = pickle.load(f)
actor_params, norm_state = loaded["actor_params"], loaded["norm_state"]
print(f"[ellip] loaded {args.ckpt} gain_mode={args.gain_mode}")

viewer = newton.viewer.ViewerGL(headless=True)
viewer.set_model(env.model)
viewer.set_camera(pos=wp.vec3(0.78, -0.42, 0.30), pitch=-26.0, yaw=118.0)

frames, kp_log, zeta_log, force_log, pegz_log = [], [], [], [], []
t = 0.0
for ep in range(args.episodes):
    obs = env.reset()
    for i in range(args.episode_length):
        normed = nrm.normalize(norm_state, jnp.asarray(obs))
        a = ppo.select_action_eval(actor_params, normed)
        obs, rew, done, info = env.step(np.asarray(a[0]))
        kp_s = np.asarray(ctrl.kp_scale[0])              # (6,) per-axis Kp multiplier
        zeta = np.asarray(ctrl.zeta[0])
        ft_pos, _ = ctrl._fingertip(env.solver.mjw_data)
        ft = np.asarray(ft_pos[0])
        # COMPLIANCE ellipsoid: radius_i ∝ 1/Kp_scale_i (translational x,y,z) ->
        # bulges where soft, pinches where stiff. World-axis aligned (OSC stiffness
        # is in the world frame). Clamp for sane sizes.
        radii = np.clip(args.r0 / np.maximum(kp_s[:3], 1e-3), 0.01, 0.15)
        starts = (WS * radii + ft).astype(np.float32)
        ends = (WE * radii + ft).astype(np.float32)
        viewer.begin_frame(t)
        viewer.log_state(env.state_0)
        viewer.log_lines("compliance", wp.array(starts, dtype=wp.vec3),
                         wp.array(ends, dtype=wp.vec3), colors=(0.2, 0.85, 1.0), width=0.0015)
        viewer.end_frame()
        img = np.asarray(viewer.get_frame().numpy())
        if img.dtype != np.uint8: img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[-1] == 4: img = img[..., :3]
        frames.append(img); t += 1.0 / args.fps
        qfc = wp.to_jax(env.solver.mjw_data.qfrc_constraint).reshape(1, -1)
        force_log.append(np.asarray(qfc[0, 9:12]))
        peg = wp.to_jax(env.state_0.body_q).reshape(1, env.nbody, 7)[0, PEG_BODY_LOCAL]
        kp_log.append(kp_s); zeta_log.append(zeta); pegz_log.append(float(peg[2]))
        if done: break

os.makedirs(args.out, exist_ok=True)
np.savez(os.path.join(args.out, "traj.npz"), kp_scale=np.array(kp_log), zeta=np.array(zeta_log),
         force=np.array(force_log), peg_z=np.array(pegz_log), gain_mode=args.gain_mode)
viewer.close()
iio.mimwrite(os.path.join(args.out, "ellipsoid.mp4"), frames, fps=args.fps, macro_block_size=2)
print(f"[ellip] saved {args.out}/ellipsoid.mp4 ({len(frames)} frames) + traj.npz")
