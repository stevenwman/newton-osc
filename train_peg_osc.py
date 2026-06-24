"""FastSAC training on the Newton peg env with the OSC controller (batched).

Mirrors jax_rl's FactoryPegInsert off-policy scheme (warmup -> collect -> replay
buffer -> FastSAC.update with a C51 distributional critic) on the Newton /
mujoco_warp backend with the 6-DOF operational-space-control action. num_envs>1
replicates the scene into N parallel worlds (synchronized episodes), so each
iteration collects N transitions. See train.log for progress; best_actor.pkl /
ckpt.pkl are written to --outdir.

    uv run python train_peg_osc.py --num-envs 512 --steps 200000 --outdir runs/osc_peg_b
"""
import os
# Share the GPU: JAX allocates on demand (no 75% grab) and is capped. Set BEFORE
# importing jax. The env/warp sim allocates separately (scales with num_envs).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")   # cap JAX; override via env var

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
    ap.add_argument("--num-envs", type=int, default=1, help="parallel worlds (replicate)")
    ap.add_argument("--steps", type=int, default=300_000,
                    help="control iterations (total transitions = steps * num_envs)")
    ap.add_argument("--episode-length", type=int, default=128)
    ap.add_argument("--outdir", default="runs/osc_peg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--min-buffer", type=int, default=2000)
    ap.add_argument("--grad-updates", type=int, default=None,
                    help="grad updates per iteration (default: config value)")
    ap.add_argument("--action-mode", choices=["absolute", "delta"], default="delta",
                    help="OSC action: 'delta' (jax_rl-style, bounded error) or 'absolute' base-frame pose")
    args = ap.parse_args()

    N = args.num_envs
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== train_peg_osc: num_envs={N} steps={args.steps} ep_len={args.episode_length} "
        f"seed={args.seed} action_mode={args.action_mode} ===")
    ctrl = OSCController()
    ctrl.action_mode = args.action_mode
    env = PegEnv(controller=ctrl, episode_length=args.episode_length, seed=args.seed,
                 weld=True, num_envs=N)
    log(f"obs_dim={env.obs_dim} act_dim={env.act_dim} (OSC 6-DOF pose, {args.action_mode})")

    # FastSAC config — matches the validated smoke config, scaled up for a real run.
    cfg = FastSACConfig(
        hidden_dim=(256, 256), critic_hidden_dim=(512, 512),
        num_atoms=51, batch_size=256, min_buffer_size=args.min_buffer,
        grad_updates_per_step=args.grad_updates or 4, policy_delay=2,
    )
    optimizer = optax.adamw(3e-4, b2=0.95, weight_decay=0.001)
    alpha_opt = optax.adam(cfg.alpha_lr)
    algo = FastSAC(cfg, env.obs_dim, env.act_dim, optimizer, alpha_opt, gamma=0.97)

    key = jax.random.PRNGKey(args.seed)
    key, k = jax.random.split(key)
    ts = algo.init(k)
    buf = JaxReplayBuffer(env.obs_dim, env.act_dim, max_size=args.buffer)

    obs = env.reset()                                  # (N, obs_dim) jnp
    ep_ret = np.zeros(N, np.float32)
    ep_succ = np.zeros(N, bool)
    ep_rets, ep_succs = [], []
    best = -1e9
    t0 = time.time()

    for t in range(args.steps):
        if buf.size < cfg.min_buffer_size:
            action = np.random.uniform(-1.0, 1.0, (N, env.act_dim)).astype(np.float32)
        else:
            key, k = jax.random.split(key)
            action = algo.select_action(ts.actor_params, obs, k)        # (N, act_dim) jnp

        nobs, rew, done, info = env.step(action)
        if not env.last_finite:
            log(f"[warn] non-finite sim at iter {t} (contact/solver blowup) — resetting batch")
            obs = env.reset()
            ep_ret[:] = 0.0; ep_succ[:] = False
            continue
        done_col = np.full(N, float(done), np.float32)
        buf.add_batch(obs=obs, action=action, reward=rew, next_obs=nobs,
                      done=done_col, truncation=done_col)
        ep_ret += np.asarray(rew)
        ep_succ |= info["success"]
        obs = nobs
        if done:
            ep_rets.extend(ep_ret.tolist())
            ep_succs.extend(ep_succ.astype(np.float32).tolist())
            ep_ret[:] = 0.0; ep_succ[:] = False
            obs = env.reset()

        if buf.size >= cfg.min_buffer_size:
            for _ in range(cfg.grad_updates_per_step):
                key, k = jax.random.split(key)
                batch = buf.sample(cfg.batch_size, k)
                batch["critic_obs"] = batch["obs"]            # symmetric critic
                batch["critic_next_obs"] = batch["next_obs"]
                ts, last = algo.update(ts, batch)
        else:
            last = {}

        if (t + 1) % 1000 == 0:
            recent = float(np.mean(ep_rets[-50:])) if ep_rets else float("nan")
            succ_rate = float(np.mean(ep_succs[-200:])) if ep_succs else float("nan")
            env_steps = (t + 1) * N
            sps = env_steps / (time.time() - t0)
            log(f"iter {t+1:7d}  envstep {env_steps:9d}  buf {buf.size:8d}  "
                f"ep_ret(mean50) {recent:8.3f}  succ%(200) {100*succ_rate:5.1f}  "
                f"q_loss {float(last.get('q1_loss', jnp.nan)):8.4f}  "
                f"actor {float(last.get('actor_loss', jnp.nan)):8.4f}  "
                f"alpha {float(last.get('alpha', jnp.nan)):.4f}  eps {len(ep_rets):5d}  {sps:7.0f} env-sps")
            if ep_rets and recent > best:
                best = recent
                with open(out / "best_actor.pkl", "wb") as f:
                    pickle.dump(jax.device_get(ts.actor_params), f)

        if (t + 1) % 20000 == 0:
            with open(out / "ckpt.pkl", "wb") as f:
                pickle.dump(jax.device_get(ts), f)
            log(f"[ckpt] saved at iter {t+1}  (best ep_ret so far {best:.3f})")

    log(f"[done] finished {args.steps} iters ({args.steps*N} env-steps), "
        f"{len(ep_rets)} episodes, best ep_ret {best:.3f}")
    logf.close()


if __name__ == "__main__":
    main()
