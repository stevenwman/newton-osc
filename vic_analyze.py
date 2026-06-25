"""Roll out a FlashSAC peg policy deterministically: record mp4 + log time series
(variable-impedance gains kp_scale/zeta + peg contact force) to npz.

    PYTHONPATH=. DISPLAY=:1 .venv/bin/python vic_analyze.py --gain-mode axis \
        --ckpt runs/vic_axis/best_actor.pkl --out runs/vic_axis
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

import argparse, pickle
import numpy as np, jax, jax.numpy as jnp, optax, warp as wp
import newton
from jax_rl.algos.flash_sac import FlashSAC
from jax_rl.configs.flash_sac_config import FlashSACConfig
from controllers import OSCController
from peg_env import PegEnv, PEG_BODY_LOCAL

ap = argparse.ArgumentParser()
ap.add_argument("--gain-mode", choices=["fixed", "single", "axis"], required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True, help="output dir prefix; writes <out>/replay.mp4 + traj.npz")
ap.add_argument("--episode-length", type=int, default=450)
ap.add_argument("--fps", type=int, default=30)
args = ap.parse_args()

ctrl = OSCController(); ctrl.action_mode = "delta"; ctrl.gain_mode = args.gain_mode
env = PegEnv(controller=ctrl, episode_length=args.episode_length, weld=True, num_envs=1)

cfg = FlashSACConfig(num_blocks=2, actor_hidden_dim=128, critic_hidden_dim=256,
                     expansion=4, num_atoms=101, v_min=-5.0, v_max=5.0)
algo = FlashSAC(cfg, env.obs_dim, env.act_dim, optax.adamw(3e-4), optax.adam(3e-4),
                gamma=0.99, critic_obs_dim=env.obs_dim, num_envs=1)
with open(args.ckpt, "rb") as f:
    loaded = pickle.load(f)
actor_params = loaded["actor_params"]; actor_bs = loaded["actor_batch_stats"]
print(f"[vic] loaded {args.ckpt} gain_mode={args.gain_mode} act_dim={env.act_dim}")

try:
    import imageio.v2 as iio
    viewer = newton.viewer.ViewerGL(headless=True)
    viewer.set_model(env.model)
    viewer.set_camera(pos=wp.vec3(1.05, -0.55, 0.45), pitch=-22.0, yaw=125.0)
    rec = True
except Exception as e:
    print(f"[vic] viewer unavailable ({e}); logging only, no mp4")
    rec = False

key = jax.random.PRNGKey(0)
frames, kp_log, zeta_log, force_log, pegz_log = [], [], [], [], []
obs = env.reset(); t = 0.0
for i in range(args.episode_length):
    key, k = jax.random.split(key)
    a = algo.select_action(actor_params, jnp.asarray(obs), k, deterministic=True,
                           actor_batch_stats=actor_bs)
    obs, rew, done, info = env.step(np.asarray(a))
    # gains used this step (controller stores per-axis arrays, world 0)
    kp_log.append(np.asarray(ctrl.kp_scale[0]))      # (6,) multiplier on KP_TASK
    zeta_log.append(np.asarray(ctrl.zeta[0]))        # (6,) damping ratio
    # Force on the peg = constraint force on its free-joint translational DOFs
    # (9:12). For a free joint these dofs are world-aligned, so qfrc_constraint
    # there IS the world-frame force the peg feels (weld reaction + bore contact;
    # cfrc_ext is not populated by this mujoco_warp). dofs: 0-6 arm, 7-8 fingers,
    # 9-14 peg free joint (9:12 trans, 12:15 rot).
    qfc = wp.to_jax(env.solver.mjw_data.qfrc_constraint).reshape(1, -1)
    force_log.append(np.asarray(qfc[0, 9:12]))                   # (3,) world force on peg
    peg = wp.to_jax(env.state_0.body_q).reshape(1, env.nbody, 7)[0, PEG_BODY_LOCAL]
    pegz_log.append(float(peg[2]))
    if rec:
        viewer.begin_frame(t); viewer.log_state(env.state_0); viewer.end_frame()
        img = np.asarray(viewer.get_frame().numpy())
        if img.dtype != np.uint8: img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[-1] == 4: img = img[..., :3]
        frames.append(img); t += 1.0 / args.fps
    if done: break

os.makedirs(args.out, exist_ok=True)
np.savez(os.path.join(args.out, "traj.npz"),
         kp_scale=np.array(kp_log), zeta=np.array(zeta_log),
         force=np.array(force_log), peg_z=np.array(pegz_log), gain_mode=args.gain_mode)
print(f"[vic] saved {args.out}/traj.npz ({len(kp_log)} steps)  "
      f"|F|max={np.linalg.norm(np.array(force_log),axis=1).max():.2f}N  peg_z_min={min(pegz_log):.3f}")
if rec:
    viewer.close()
    iio.mimwrite(os.path.join(args.out, "replay.mp4"), frames, fps=args.fps, macro_block_size=2)
    print(f"[vic] saved {args.out}/replay.mp4 ({len(frames)} frames)")
