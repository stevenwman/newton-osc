"""Decoupled recording benchmark: (A) headless graph-captured warp sim storing the
body_q trajectory, then (B) offline render at VIDEO fps (subsampled) — vs the old
"rawdog" method (jax OSC + render every control step). Single env, no noise.

    PYTHONPATH=. DISPLAY=:1 .venv/bin/python vic_decouple_record.py --gain-mode axis
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
import argparse, pickle, time
import numpy as np, jax.numpy as jnp, optax, warp as wp
import newton, imageio.v2 as iio
from jax_rl.algos.ppo import PPO
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.utils import normalization as nrm
from controllers import OSCController
from controllers_warp import WarpOSCController
from peg_env_square import PegEnv, PEG_BODY_LOCAL
import peg_scene_square as scene

ap = argparse.ArgumentParser()
ap.add_argument("--gain-mode", default="axis")
ap.add_argument("--episode-length", type=int, default=450)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--out", default="runs/sq_vic_decouple")
args = ap.parse_args()
CKPT = f"runs/sq_vic_{args.gain_mode}/best_actor.pkl"
os.makedirs(args.out, exist_ok=True)
CTRL_HZ = 1.0 / (scene.SUBSTEPS * scene.SIM_DT)               # 125 Hz
STRIDE = max(1, round(CTRL_HZ / args.fps))                    # render every Nth control step


def build(Ctor):
    c = Ctor(); c.action_mode = "delta"; c.gain_mode = args.gain_mode
    env = PegEnv(controller=c, episode_length=args.episode_length, weld=True, num_envs=1, seed=0)
    cfg = PPOConfig(policy_hidden_dim=(256, 256), value_hidden_dim=(256, 256),
                    activation="swish", squash=True, state_dependent_std=False)
    cfg.num_envs = 1; cfg.minibatch_size = 1
    cfg.encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=(256, 256), activation="swish")
    cfg.critic_encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=(256, 256), activation="swish")
    cfg.policy_head = PolicyHeadConfig(action_dim=env.act_dim, squash=True, state_dependent_std=False)
    ppo = PPO(cfg, env.obs_dim, env.act_dim, optax.adam(3e-4), optax.adam(3e-4))
    L = pickle.load(open(CKPT, "rb"))
    return env, c, ppo, L["actor_params"], L["norm_state"]


def add_viewer_extras(viewer, env):
    """hide solid hole shapes (draw nothing fancy here; bench is about timing)"""
    hb = [l.split('/')[-1] for l in env.model.body_label].index("hole_base")
    sb = env.model.shape_body.numpy()
    fl = env.model.shape_flags.numpy()
    for i in range(len(sb)):
        if sb[i] == hb:
            fl[i] &= ~int(newton.ShapeFlags.VISIBLE)
    env.model.shape_flags.assign(fl)


def frame(viewer):
    img = np.asarray(viewer.get_frame().numpy())
    if img.dtype != np.uint8: img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img[..., :3] if img.shape[-1] == 4 else img


# ───────────────────────── OLD: jax OSC + render every control step ─────────────
env, c, ppo, ap_, ns = build(OSCController)
viewer = newton.viewer.ViewerGL(headless=True); viewer.set_model(env.model)
viewer.set_camera(pos=wp.vec3(0.78, -0.42, 0.30), pitch=-26.0, yaw=118.0)
o = env.reset()
for _ in range(5):                                           # warmup
    o, _, _, _ = env.step(np.asarray(ppo.select_action_eval(ap_, nrm.normalize(ns, jnp.asarray(o)))))
env.rng = np.random.default_rng(0); o = env.reset(); wp.synchronize_device()
t0 = time.time(); frames = []; tt = 0.0
for i in range(args.episode_length):
    o, r, d, info = env.step(np.asarray(ppo.select_action_eval(ap_, nrm.normalize(ns, jnp.asarray(o)))))
    viewer.begin_frame(tt); viewer.log_state(env.state_0); viewer.end_frame()
    frames.append(frame(viewer)); tt += 1.0 / args.fps
wp.synchronize_device(); t_old = time.time() - t0
iio.mimwrite(f"{args.out}/old_rawdog.mp4", frames, fps=args.fps, macro_block_size=2)
viewer.close(); del env, ppo

# ───────────────────────── NEW: (A) headless graph sim store states ─────────────
env, c, ppo, ap_, ns = build(WarpOSCController)
o = env.reset(); c.set_action(env, np.zeros((1, env.act_dim), np.float32)); env.capture_substep()
o = env.reset()
for _ in range(5):
    o, _, _, _ = env.step(np.asarray(ppo.select_action_eval(ap_, nrm.normalize(ns, jnp.asarray(o)))))
env.rng = np.random.default_rng(0); o = env.reset(); wp.synchronize_device()
t0 = time.time(); states = []
for i in range(args.episode_length):
    o, r, d, info = env.step(np.asarray(ppo.select_action_eval(ap_, nrm.normalize(ns, jnp.asarray(o)))))
    states.append(env.state_0.body_q.numpy().copy())         # store body_q trajectory
wp.synchronize_device(); t_sim = time.time() - t0

# ───────────────────────── NEW: (B) offline render at video fps ─────────────────
t0 = time.time()
viewer = newton.viewer.ViewerGL(headless=True); viewer.set_model(env.model)
viewer.set_camera(pos=wp.vec3(0.78, -0.42, 0.30), pitch=-26.0, yaw=118.0)
st = env.model.state()
frames = []; tt = 0.0
for i in range(0, args.episode_length, STRIDE):              # only video-fps frames
    st.body_q.assign(states[i])
    viewer.begin_frame(tt); viewer.log_state(st); viewer.end_frame()
    frames.append(frame(viewer)); tt += 1.0 / args.fps
iio.mimwrite(f"{args.out}/decoupled.mp4", frames, fps=args.fps, macro_block_size=2)
viewer.close(); t_render = time.time() - t0

print(f"\n=== recording one {args.episode_length}-step trajectory ({args.gain_mode}) ===")
print(f"  OLD  (jax OSC + render every step): {t_old:5.1f}s  ({args.episode_length} frames)")
print(f"  NEW  headless graph sim (store):    {t_sim:5.1f}s")
print(f"  NEW  offline render @ {args.fps}fps:      {t_render:5.1f}s  ({len(frames)} frames, stride {STRIDE})")
print(f"  NEW  total:                         {t_sim + t_render:5.1f}s  -> {t_old/(t_sim+t_render):.2f}x faster")
