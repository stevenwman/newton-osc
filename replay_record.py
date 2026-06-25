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
from jax_rl.algos.flash_sac import FlashSAC
from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.algos.ppo import PPO
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.utils import normalization as nrm
from controllers import OSCController

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/osc_peg/best_actor.pkl")
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--episode-length", type=int, default=128)
ap.add_argument("--out", default="runs/osc_peg/replay.mp4")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
ap.add_argument("--action-mode", choices=["absolute", "delta"], default="delta",
                help="must match the action mode the checkpoint was trained with")
ap.add_argument("--algo", choices=["fastsac", "flashsac", "ppo"], default="fastsac",
                help="which algo produced the checkpoint")
ap.add_argument("--env", choices=["peg", "square"], default="peg")
ap.add_argument("--gain-mode", choices=["fixed", "single", "axis"], default="fixed")
args = ap.parse_args()

if args.env == "square":
    from peg_env_square import PegEnv
    import peg_scene_square as scene
else:
    from peg_env import PegEnv
    import peg_scene_newton as scene

ctrl = OSCController()
ctrl.action_mode = args.action_mode
ctrl.gain_mode = args.gain_mode
env = PegEnv(controller=ctrl, episode_length=args.episode_length, weld=True)

# Rebuild the algo with the SAME cfg/dims used for training, just to get select_action.
actor_bs = None      # FlashSAC needs BatchNorm running stats; FastSAC doesn't
ppo_norm = None      # PPO needs the obs normalizer
with open(args.ckpt, "rb") as f:
    loaded = pickle.load(f)
if args.algo == "ppo":
    cfg = PPOConfig(policy_hidden_dim=(256, 256), value_hidden_dim=(256, 256),
                    activation="swish", squash=True, state_dependent_std=False)
    cfg.num_envs = 1; cfg.minibatch_size = 1
    cfg.encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=cfg.policy_hidden_dim, activation=cfg.activation)
    cfg.critic_encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=cfg.value_hidden_dim, activation=cfg.activation)
    cfg.policy_head = PolicyHeadConfig(action_dim=env.act_dim, squash=True, state_dependent_std=False)
    algo = PPO(cfg, env.obs_dim, env.act_dim, optax.adam(3e-4), optax.adam(3e-4))
    actor_params, ppo_norm = loaded["actor_params"], loaded["norm_state"]
elif args.algo == "flashsac":
    cfg = FlashSACConfig(num_blocks=2, actor_hidden_dim=128, critic_hidden_dim=256,
                         expansion=4, num_atoms=101, v_min=-5.0, v_max=5.0)
    algo = FlashSAC(cfg, env.obs_dim, env.act_dim, optax.adamw(3e-4), optax.adam(3e-4),
                    gamma=0.99, critic_obs_dim=env.obs_dim, num_envs=1)
    # ckpt is {"actor_params":..., "actor_batch_stats":...} (best_actor) or a TrainingState.
    if isinstance(loaded, dict) and "actor_params" in loaded:
        actor_params, actor_bs = loaded["actor_params"], loaded["actor_batch_stats"]
    else:
        actor_params, actor_bs = loaded.actor_params, loaded.actor_batch_stats
else:
    cfg = FastSACConfig(
        hidden_dim=(256, 256), critic_hidden_dim=(512, 512),
        num_atoms=51, batch_size=256, min_buffer_size=2000,
        grad_updates_per_step=4, policy_delay=2,
    )
    algo = FastSAC(cfg, env.obs_dim, env.act_dim, optax.adamw(3e-4), optax.adam(cfg.alpha_lr), gamma=0.97)
    actor_params = loaded.actor_params if hasattr(loaded, "actor_params") else loaded
print(f"[replay] loaded {args.ckpt} (algo={args.algo})")

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
        if args.algo == "ppo":
            normed = nrm.normalize(ppo_norm, jnp.asarray(obs)[None])
            a = algo.select_action_eval(actor_params, normed)   # deterministic mean (tanh-squashed)
        else:
            _kw = {"actor_batch_stats": actor_bs} if args.algo == "flashsac" else {}
            a = algo.select_action(actor_params, jnp.asarray(obs)[None], k,
                                   deterministic=not args.stochastic, **_kw)
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
