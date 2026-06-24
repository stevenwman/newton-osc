"""Replay a trained OSC peg-insertion policy and record an mp4 (headless GL).

    uv run python replay_record.py --ckpt runs/osc_peg/best_actor.pkl \
        --episodes 3 --out runs/osc_peg/replay.mp4

Frame capture mirrors jax_rl/projects/mud_eval (viewer.get_frame() -> imageio).
best_actor.pkl holds the actor params; ckpt.pkl holds the full train_state.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import warp as wp
import optax
import imageio.v2 as iio

import newton
from jax_rl.algos.fast_sac import FastSAC
from jax_rl.configs.fast_sac_config import FastSACConfig
from controllers import OSCController
from peg_env import PegEnv
import peg_scene_newton as scene

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/osc_peg/best_actor.pkl")
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--episode-length", type=int, default=128)
ap.add_argument("--out", default="runs/osc_peg/replay.mp4")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
ap.add_argument("--action-mode", choices=["absolute", "delta"], default="delta",
                help="must match the action mode the checkpoint was trained with")
args = ap.parse_args()

ctrl = OSCController()
ctrl.action_mode = args.action_mode
env = PegEnv(controller=ctrl, episode_length=args.episode_length, weld=True)

# Rebuild the algo with the SAME cfg/dims used for training, just to get select_action.
cfg = FastSACConfig(
    hidden_dim=(256, 256), critic_hidden_dim=(512, 512),
    num_atoms=51, batch_size=256, min_buffer_size=2000,
    grad_updates_per_step=4, policy_delay=2,
)
algo = FastSAC(cfg, env.obs_dim, env.act_dim, optax.adamw(3e-4), optax.adam(cfg.alpha_lr), gamma=0.97)
with open(args.ckpt, "rb") as f:
    loaded = pickle.load(f)
actor_params = loaded.actor_params if hasattr(loaded, "actor_params") else loaded
print(f"[replay] loaded {args.ckpt} ({'train_state' if hasattr(loaded, 'actor_params') else 'actor_params'})")

viewer = newton.viewer.ViewerGL(headless=True)
viewer.set_model(env.model)
viewer.set_camera(pos=wp.vec3(1.05, -0.55, 0.45), pitch=-22.0, yaw=125.0)

key = jax.random.PRNGKey(0)
frames = []
t = 0.0
HOLE_TOP = 0.075   # bore opening z (peg seats when peg body z drops well below this)
for ep in range(args.episodes):
    obs = env.reset()
    ep_ret = 0.0
    peg_min_z = 1e9
    for i in range(args.episode_length):
        key, k = jax.random.split(key)
        a = algo.select_action(actor_params, jnp.asarray(obs)[None], k, deterministic=not args.stochastic)
        obs, rew, done, info = env.step(np.asarray(a[0]))
        ep_ret += float(np.asarray(rew).reshape(-1)[0])
        peg = env.state_0.body_q.numpy()[scene.PEG_BODY_IDX]
        peg_min_z = min(peg_min_z, float(peg[2]))
        viewer.begin_frame(t)
        viewer.log_state(env.state_0)
        viewer.end_frame()
        img = np.asarray(viewer.get_frame().numpy())
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[-1] == 4:
            img = img[..., :3]
        frames.append(img)
        t += 1.0 / args.fps
        if done:
            break
    seated = "SEATED" if peg_min_z < HOLE_TOP - 0.02 else "above"
    print(f"  ep {ep}: ep_ret={ep_ret:7.1f}  peg_min_z={peg_min_z:.3f}  ({seated}; bore top ~{HOLE_TOP})")

viewer.close()
iio.mimwrite(args.out, frames, fps=args.fps, macro_block_size=2)
print(f"[replay] saved {args.out} ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
