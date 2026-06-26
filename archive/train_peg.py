"""Minimal FastSAC training loop on the Newton peg env — smoke check.

Same shape as jax_rl's offpolicy loop (warmup -> collect -> replay buffer ->
FastSAC.update) but stripped of all fluff (no wandb / checkpointing / eval /
obs-norm / distributed / env-backend registry). Single env. Purpose: prove the
Newton-env <-> FastSAC <-> replay-buffer pipeline runs end-to-end with finite
losses. Run:  uv run python train_peg.py
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.algos.fast_sac import FastSAC
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.fast_sac_config import FastSACConfig
from peg_env import PegEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--episode-length", type=int, default=100)
    args = ap.parse_args()

    env = PegEnv(episode_length=args.episode_length, seed=0)

    # Small/fast config for a smoke check (paper defaults are huge).
    cfg = FastSACConfig(
        hidden_dim=(256, 256), critic_hidden_dim=(256, 256),
        num_atoms=51, batch_size=256, min_buffer_size=500,
        grad_updates_per_step=2, policy_delay=2,
    )
    optimizer = optax.adamw(3e-4, b2=0.95, weight_decay=0.001)
    alpha_opt = optax.adam(cfg.alpha_lr)
    algo = FastSAC(cfg, env.obs_dim, env.act_dim, optimizer, alpha_opt, gamma=0.97)

    key = jax.random.PRNGKey(0)
    key, k = jax.random.split(key)
    train_state = algo.init(k)
    buf = JaxReplayBuffer(env.obs_dim, env.act_dim, max_size=50_000)

    obs = env.reset()
    ep_ret, ep_rets = 0.0, []
    last = {}

    for t in range(args.steps):
        if buf.size < cfg.min_buffer_size:
            action = np.random.uniform(-1.0, 1.0, env.act_dim).astype(np.float32)
        else:
            key, k = jax.random.split(key)
            a = algo.select_action(train_state.actor_params, jnp.asarray(obs)[None], k)
            action = np.asarray(a[0])

        nobs, rew, done, info = env.step(action)
        buf.add_batch(
            obs=obs[None], action=action[None], reward=np.array([rew], np.float32),
            next_obs=nobs[None], done=np.array([float(done)], np.float32),
            truncation=np.array([info["truncation"]], np.float32))
        ep_ret += rew
        obs = nobs
        if done:
            ep_rets.append(ep_ret); ep_ret = 0.0
            obs = env.reset()

        if buf.size >= cfg.min_buffer_size:
            for _ in range(cfg.grad_updates_per_step):
                key, k = jax.random.split(key)
                batch = buf.sample(cfg.batch_size, k)
                batch["critic_obs"] = batch["obs"]            # symmetric critic
                batch["critic_next_obs"] = batch["next_obs"]
                train_state, last = algo.update(train_state, batch)

        if (t + 1) % 200 == 0:
            cl = float(last.get("q1_loss", jnp.nan))   # C51 critic loss
            al = float(last.get("actor_loss", jnp.nan))
            alpha = float(last.get("alpha", jnp.nan))
            recent = np.mean(ep_rets[-5:]) if ep_rets else float("nan")
            print(f"step {t+1:5d}  buf {buf.size:6d}  ep_ret(mean5) {recent:8.3f}  "
                  f"q_loss {cl:9.4f}  actor_loss {al:9.4f}  alpha {alpha:.4f}")

    # Smoke verdict: updates ran with finite losses end-to-end.
    cl = float(last.get("q1_loss", jnp.nan))
    al = float(last.get("actor_loss", jnp.nan))
    ok = bool(last) and np.isfinite(cl) and np.isfinite(al) and buf.size > cfg.min_buffer_size
    print(f"\n[smoke] updates ran, final q_loss={cl:.4f} actor_loss={al:.4f} "
          f"episodes={len(ep_rets)} -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
