"""FlashSAC training on the Newton peg env — faithful port of jax_rl's converged run.

This is the P1 isolation experiment: swap our algo (was FastSAC) for FlashSAC and
match jax_rl's FactoryPegInsert regime EXACTLY, while keeping our current Newton
env / OSC / 30-d obs / v18 reward unchanged. If succ% lifts off 0, the algorithm
(FlashSAC's BatchNorm+weight-norm blocks, adaptive reward normalization, Zeta-noise
exploration, asymmetric-capable C51 critic) was the missing piece. If it stays 0,
the bug is downstream in our env / OSC / obs and the next phase swaps those.

Reference config (from the converged run's meta.json):
  algo=flash_sac  num_envs=128  episode_length=450  total=2.5M  gamma=0.99
  num_atoms=101  v_min/v_max=[-5,5]  tau=0.01  batch=2048  grad_updates=16
  alpha_init=0.1  sigma_target=0.3  normalize_reward=True  G_max=5.0
  noise_zeta_mu=2.0  noise_zeta_max=16  weight_norm=True  lr 3e-4->1.5e-4 cosine

FlashSAC machinery copied verbatim from jax-learning (jax_rl/algos/flash_sac.py +
flash_blocks, reward_scaling). Only this train loop is hand-written, adapted from
jax-learning's scripts/train_flashsac.py (dropping its mjx env-bundle / orbax /
wandb infra for our numpy-driving Newton loop).

    uv run python train_peg_flashsac.py --outdir runs/flashsac_p1
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.algos.flash_sac import FlashSAC
from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
from jax_rl.configs.flash_sac_config import FlashSACConfig
from jax_rl.utils.reward_scaling import update_reward_stats, scale_reward
from controllers import OSCController
from peg_env import PegEnv


def _make_zeta_cdf(mu: float, max_n: int) -> jnp.ndarray:
    """CDF for Zeta P(k) ∝ k^(-mu), k=1..max_n (noise-repetition exploration)."""
    ks = jnp.arange(1, max_n + 1, dtype=jnp.float32)
    pmf = ks ** (-mu)
    pmf = pmf / pmf.sum()
    return jnp.cumsum(pmf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=128, help="parallel worlds (ref=128)")
    ap.add_argument("--total-steps", type=int, default=2_500_000,
                    help="total env-steps (ref=2.5M); iters = total // num_envs")
    ap.add_argument("--episode-length", type=int, default=450, help="ref=450")
    ap.add_argument("--batch-size", type=int, default=2048, help="ref=2048")
    ap.add_argument("--grad-updates", type=int, default=16, help="UTD ratio (ref=16)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--min-buffer", type=int, default=10_000, help="ref=10000")
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--action-mode", choices=["absolute", "delta"], default="delta")
    ap.add_argument("--no-reward-norm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs/flashsac_p1")
    args = ap.parse_args()

    N = args.num_envs
    iters = args.total_steps // N
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    log(f"=== train_peg_flashsac: N={N} total_steps={args.total_steps} iters={iters} "
        f"ep_len={args.episode_length} batch={args.batch_size} grad={args.grad_updates} "
        f"seed={args.seed} action_mode={args.action_mode} ===")

    ctrl = OSCController()
    ctrl.action_mode = args.action_mode
    env = PegEnv(controller=ctrl, episode_length=args.episode_length, seed=args.seed,
                 weld=True, num_envs=N)
    log(f"obs_dim={env.obs_dim} act_dim={env.act_dim} (P1: symmetric critic, current OSC/obs)")

    cfg = FlashSACConfig(
        num_blocks=2, actor_hidden_dim=128, critic_hidden_dim=256, expansion=4,
        num_atoms=101, v_min=-5.0, v_max=5.0, tau=0.01, policy_delay=2,
        batch_size=args.batch_size, buffer_size=args.buffer,
        min_buffer_size=args.min_buffer, grad_updates_per_step=args.grad_updates,
        gamma=args.gamma, n_step=1, alpha_init=0.1, sigma_target=0.3, bc_alpha=0.0,
        normalize_reward=not args.no_reward_norm, G_max=5.0,
        noise_zeta_mu=2.0, noise_zeta_max=16,
        lr_init=3e-4, lr_peak=3e-4, lr_end=1.5e-4, lr_warmup_frac=1e-6,
        lr_decay_frac=1.0, weight_norm=True,
    )

    # LR schedule (warmup -> cosine), shared by actor/critic/alpha (matches ref).
    total_grad_steps = max(1, ((args.total_steps - args.min_buffer) // N) * args.grad_updates)
    warmup = max(1, int(cfg.lr_warmup_frac * total_grad_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.lr_init, peak_value=cfg.lr_peak, end_value=cfg.lr_end,
        warmup_steps=warmup, decay_steps=total_grad_steps)
    optimizer = optax.adamw(learning_rate=schedule, b2=0.95, weight_decay=0.001)
    alpha_opt = optax.adam(learning_rate=schedule)

    algo = FlashSAC(cfg, env.obs_dim, env.act_dim, optimizer, alpha_opt,
                    gamma=args.gamma, critic_obs_dim=env.obs_dim, num_envs=N)
    key = jax.random.PRNGKey(args.seed)
    key, k = jax.random.split(key)
    ts = algo.init(k)
    buf = JaxReplayBuffer(env.obs_dim, env.act_dim, max_size=args.buffer)

    zeta_cdf = _make_zeta_cdf(cfg.noise_zeta_mu, cfg.noise_zeta_max)
    actor_net = algo.actor

    @jax.jit
    def select_with_noise(actor_params, actor_bs, obs, noise_state, key):
        mean, log_std = actor_net.apply(
            {"params": actor_params, "batch_stats": actor_bs}, obs, train=False)
        std = jnp.exp(log_std)
        reinit = (noise_state.count == 0) | (noise_state.count >= noise_state.repeat_n)
        k1, k2 = jax.random.split(key)
        new_noise = jax.random.normal(k1, mean.shape)
        u = jax.random.uniform(k2, (mean.shape[0],))
        new_n = (jnp.searchsorted(zeta_cdf, u) + 1).astype(jnp.int32)
        noise = jnp.where(reinit[:, None], new_noise, noise_state.noise)
        repeat_n = jnp.where(reinit, new_n, noise_state.repeat_n)
        count = jnp.where(reinit, jnp.ones_like(noise_state.count), noise_state.count + 1)
        action = jnp.tanh(mean + std * noise)
        return action, noise_state.replace(noise=noise, count=count, repeat_n=repeat_n)

    obs = env.reset()                       # (N, obs_dim) jnp
    ep_ret = np.zeros(N, np.float32)
    ep_succ = np.zeros(N, bool)
    ep_rets, ep_succs = [], []
    best = -1e9
    t0 = time.time()
    last = {}

    for t in range(iters):
        if buf.size < cfg.min_buffer_size:
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (N, env.act_dim), minval=-1.0, maxval=1.0)
        else:
            key, ak = jax.random.split(key)
            action, new_ns = select_with_noise(
                ts.actor_params, ts.actor_batch_stats, obs, ts.noise_state, ak)
            ts = ts.replace(noise_state=new_ns)

        nobs, rew, done, info = env.step(action)
        if not env.last_finite:
            log(f"[warn] non-finite sim at iter {t} — resetting batch")
            obs = env.reset(); ep_ret[:] = 0.0; ep_succ[:] = False
            continue

        # Synchronized TimeLimit episodes: done is pure truncation (no early term),
        # so terminated=0 (always bootstrap), truncated=done.
        done_col = np.full(N, float(done), np.float32)
        rew_j = jnp.asarray(rew)
        if cfg.normalize_reward:
            ts = ts.replace(reward_norm_state=update_reward_stats(
                ts.reward_norm_state, rew_j,
                terminated=jnp.zeros(N, jnp.float32), truncated=jnp.asarray(done_col),
                gamma=args.gamma))

        buf.add_batch(obs=obs, action=action, reward=np.asarray(rew), next_obs=nobs,
                      done=done_col, truncation=done_col)     # raw reward; scaled at sample
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
                key, sk = jax.random.split(key)
                batch = buf.sample(cfg.batch_size, sk)
                if cfg.normalize_reward:
                    batch["reward"] = scale_reward(ts.reward_norm_state, batch["reward"],
                                                   G_max=cfg.G_max)
                batch["critic_obs"] = batch["obs"]            # P1 symmetric critic
                batch["critic_next_obs"] = batch["next_obs"]
                ts, m = algo.update(ts, batch)
                if float(m.get("actor_loss", 0.0)) != 0.0:
                    last = m

        if (t + 1) % 500 == 0:
            recent = float(np.mean(ep_rets[-50:])) if ep_rets else float("nan")
            succ_rate = float(np.mean(ep_succs[-200:])) if ep_succs else float("nan")
            env_steps = (t + 1) * N
            sps = env_steps / (time.time() - t0)
            denom = float("nan")
            if cfg.normalize_reward:
                gv = float(ts.reward_norm_state.G_var); gm = float(ts.reward_norm_state.G_r_max)
                denom = max(gv ** 0.5, gm / cfg.G_max)
            log(f"iter {t+1:7d}  envstep {env_steps:9d}  buf {buf.size:8d}  "
                f"ep_ret(mean50) {recent:8.2f}  succ%(200) {100*succ_rate:5.1f}  "
                f"q_loss {float(last.get('q1_loss', jnp.nan)):7.3f}  "
                f"actor {float(last.get('actor_loss', jnp.nan)):8.3f}  "
                f"alpha {float(last.get('alpha', jnp.nan)):.4f}  "
                f"rscale {denom:7.3f}  eps {len(ep_rets):5d}  {sps:7.0f} env-sps")
            if ep_rets and recent > best:
                best = recent
                with open(out / "best_actor.pkl", "wb") as f:
                    pickle.dump({"actor_params": jax.device_get(ts.actor_params),
                                 "actor_batch_stats": jax.device_get(ts.actor_batch_stats)}, f)

        if (t + 1) % 5000 == 0:
            with open(out / "ckpt.pkl", "wb") as f:
                pickle.dump(jax.device_get(ts), f)
            log(f"[ckpt] saved at iter {t+1}  (best ep_ret {best:.2f})")

    log(f"[done] {iters} iters ({iters*N} env-steps), {len(ep_rets)} episodes, best {best:.2f}")
    logf.close()


if __name__ == "__main__":
    main()
