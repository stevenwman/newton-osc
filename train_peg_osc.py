"""FastSAC training on the Newton peg env with the OSC controller.

Mirrors jax_rl's FactoryPegInsert off-policy scheme (warmup -> collect -> replay
buffer -> FastSAC.update with a C51 distributional critic) but on the Newton /
mujoco_warp backend, single env, with the 6-DOF operational-space-control action
(absolute base-frame pose target -> Khatib OSC torques). See train.log for
progress; best_actor.pkl / ckpt.pkl are written to --outdir.

VRAM: tuned to share the GPU (JAX on-demand, capped). Run:
    uv run python train_peg_osc.py --steps 300000 --outdir runs/osc_peg
"""
import os
# Share the GPU: JAX allocates on demand (no 75% grab) and is capped. Set BEFORE
# importing jax. The env/warp sim allocates separately (~1 GB for a single env).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")   # cap JAX; override via env var

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.algos.fast_sac import FastSAC
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.fast_sac_config import FastSACConfig
from controllers import OSCController
from peg_env import PegEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--episode-length", type=int, default=128)
    ap.add_argument("--outdir", default="runs/osc_peg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--min-buffer", type=int, default=2000)
    ap.add_argument("--action-mode", choices=["absolute", "delta"], default="delta",
                    help="OSC action: 'delta' (jax_rl-style, bounded error) or 'absolute' base-frame pose")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== train_peg_osc: steps={args.steps} ep_len={args.episode_length} "
        f"seed={args.seed} action_mode={args.action_mode} ===")
    ctrl = OSCController()
    ctrl.action_mode = args.action_mode
    env = PegEnv(controller=ctrl, episode_length=args.episode_length, seed=args.seed, weld=True)
    log(f"obs_dim={env.obs_dim} act_dim={env.act_dim} (OSC 6-DOF pose, {args.action_mode})")

    # FastSAC config — matches the validated smoke config, scaled up for a real run.
    cfg = FastSACConfig(
        hidden_dim=(256, 256), critic_hidden_dim=(512, 512),
        num_atoms=51, batch_size=256, min_buffer_size=args.min_buffer,
        grad_updates_per_step=4, policy_delay=2,
    )
    optimizer = optax.adamw(3e-4, b2=0.95, weight_decay=0.001)
    alpha_opt = optax.adam(cfg.alpha_lr)
    algo = FastSAC(cfg, env.obs_dim, env.act_dim, optimizer, alpha_opt, gamma=0.97)

    key = jax.random.PRNGKey(args.seed)
    key, k = jax.random.split(key)
    ts = algo.init(k)
    buf = JaxReplayBuffer(env.obs_dim, env.act_dim, max_size=args.buffer)

    obs = env.reset()
    ep_ret, ep_rets, last = 0.0, [], {}
    ep_succ, ep_succs = False, []          # success = is_success reached at any step of the episode
    best = -1e9
    t0 = time.time()

    for t in range(args.steps):
        if buf.size < cfg.min_buffer_size:
            action = np.random.uniform(-1.0, 1.0, env.act_dim).astype(np.float32)
        else:
            key, k = jax.random.split(key)
            a = algo.select_action(ts.actor_params, jnp.asarray(obs)[None], k)
            action = np.asarray(a[0])

        nobs, rew, done, info = env.step(action)
        if not (np.isfinite(rew) and np.all(np.isfinite(nobs))):
            log(f"[warn] non-finite sim at step {t} (contact/solver blowup) — resetting, skipping")
            obs = env.reset()
            ep_ret = 0.0
            continue
        buf.add_batch(
            obs=obs[None], action=action[None], reward=np.array([rew], np.float32),
            next_obs=nobs[None], done=np.array([float(done)], np.float32),
            truncation=np.array([info["truncation"]], np.float32))
        ep_ret += rew
        ep_succ = ep_succ or bool(info.get("success", False))
        obs = nobs
        if done:
            ep_rets.append(ep_ret)
            ep_succs.append(float(ep_succ))
            ep_ret, ep_succ = 0.0, False
            obs = env.reset()

        if buf.size >= cfg.min_buffer_size:
            for _ in range(cfg.grad_updates_per_step):
                key, k = jax.random.split(key)
                batch = buf.sample(cfg.batch_size, k)
                batch["critic_obs"] = batch["obs"]            # symmetric critic
                batch["critic_next_obs"] = batch["next_obs"]
                ts, last = algo.update(ts, batch)

        if (t + 1) % 1000 == 0:
            recent = float(np.mean(ep_rets[-10:])) if ep_rets else float("nan")
            succ_rate = float(np.mean(ep_succs[-50:])) if ep_succs else float("nan")
            sps = (t + 1) / (time.time() - t0)
            log(f"step {t+1:7d}  buf {buf.size:7d}  ep_ret(mean10) {recent:8.3f}  "
                f"succ%(50) {100*succ_rate:5.1f}  "
                f"q_loss {float(last.get('q1_loss', jnp.nan)):8.4f}  "
                f"actor {float(last.get('actor_loss', jnp.nan)):8.4f}  "
                f"alpha {float(last.get('alpha', jnp.nan)):.4f}  eps {len(ep_rets):4d}  {sps:5.1f} sps")
            if ep_rets and recent > best:
                best = recent
                with open(out / "best_actor.pkl", "wb") as f:
                    pickle.dump(jax.device_get(ts.actor_params), f)

        if (t + 1) % 20000 == 0:
            with open(out / "ckpt.pkl", "wb") as f:
                pickle.dump(jax.device_get(ts), f)
            log(f"[ckpt] saved at step {t+1}  (best ep_ret so far {best:.3f})")

    log(f"[done] finished {args.steps} steps, {len(ep_rets)} episodes, best ep_ret {best:.3f}")
    logf.close()


if __name__ == "__main__":
    main()
