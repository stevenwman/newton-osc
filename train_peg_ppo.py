"""PPO training on the Newton peg env — on-policy counterpart to train_peg_flashsac.

Vendored PPO from jax-learning (jax_rl/algos/ppo.py + buffers/rollout.py +
utils/normalization.py + configs/ppo_config.py; networks/distributions shared).
Reimplements jax-learning's on-policy loop (scripts/train_ppo.py) against our
batched numpy-driving PegEnv: collect num_steps×N rollout -> GAE bootstrap ->
num_epochs × num_minibatches clipped updates, with running obs normalization.

Purpose: wall-clock + sample-efficiency comparison vs off-policy FlashSAC on the
SAME env/OSC/reward. PPO is env-throughput-bound (few grad updates per env-step);
FlashSAC is grad-bound (UTD). Run on the circular peg first.

    PYTHONPATH=. .venv/bin/python train_peg_ppo.py --env peg --outdir runs/ppo_peg
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

from jax_rl.algos.ppo import PPO
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.buffers.rollout import RolloutBuffer
from jax_rl.utils import normalization as nrm
from controllers import OSCController


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["peg", "square"], default="peg")
    ap.add_argument("--gain-mode", choices=["fixed", "single", "axis"], default="fixed")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--episode-length", type=int, default=450)
    ap.add_argument("--num-steps", type=int, default=64, help="rollout length per update")
    ap.add_argument("--num-minibatches", type=int, default=32)
    ap.add_argument("--num-epochs", type=int, default=4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--action-mode", choices=["absolute", "delta"], default="delta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs/ppo_peg")
    args = ap.parse_args()

    N = args.num_envs
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    iters = args.total_steps // (args.num_steps * N)
    log(f"=== train_peg_ppo: env={args.env} N={N} total={args.total_steps} iters={iters} "
        f"num_steps={args.num_steps} minibatches={args.num_minibatches} epochs={args.num_epochs} "
        f"seed={args.seed} ===")

    if args.env == "square":
        from peg_env_square import PegEnv
    else:
        from peg_env import PegEnv
    ctrl = OSCController(); ctrl.action_mode = args.action_mode; ctrl.gain_mode = args.gain_mode
    env = PegEnv(controller=ctrl, episode_length=args.episode_length, seed=args.seed,
                 weld=True, num_envs=N)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    log(f"obs_dim={obs_dim} act_dim={act_dim} gain_mode={args.gain_mode}")

    # PPO config — network configs + runtime fields populated here (per train.py contract).
    cfg = PPOConfig(
        clip_eps=args.clip_eps, entropy_coef=args.entropy_coef, gae_lambda=args.gae_lambda,
        num_epochs=args.num_epochs, num_minibatches=args.num_minibatches,
        num_steps=args.num_steps, policy_hidden_dim=(256, 256), value_hidden_dim=(256, 256),
        activation="swish", squash=True, state_dependent_std=False,
        max_grad_norm=0.5, anneal_lr=True, normalize_advantage=True, gamma=args.gamma)
    cfg.num_envs = N
    cfg.minibatch_size = (args.num_steps * N) // args.num_minibatches
    cfg.encoder = EncoderConfig(obs_dim=obs_dim, hidden_dim=cfg.policy_hidden_dim, activation=cfg.activation)
    cfg.critic_encoder = EncoderConfig(obs_dim=obs_dim, hidden_dim=cfg.value_hidden_dim, activation=cfg.activation)
    cfg.policy_head = PolicyHeadConfig(action_dim=act_dim, squash=cfg.squash,
                                       state_dependent_std=cfg.state_dependent_std)

    total_grad_steps = max(1, iters * args.num_epochs * args.num_minibatches)
    sched = optax.linear_schedule(args.lr, 0.0, total_grad_steps) if cfg.anneal_lr else args.lr
    mk_opt = lambda: optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(sched))
    ppo = PPO(cfg, obs_dim, act_dim, mk_opt(), mk_opt())

    key = jax.random.PRNGKey(args.seed)
    key, k = jax.random.split(key)
    ts = ppo.init(k)
    norm_state = nrm.init(obs_dim)

    obs = env.reset()                                   # (N, obs_dim)
    ep_ret = np.zeros(N, np.float32); ep_succ = np.zeros(N, bool)
    ep_rets, ep_succs = [], []
    best = -1e9; t0 = time.time()

    for it in range(iters):
        buf = RolloutBuffer(args.num_steps, N, obs_dim, act_dim)
        raw_obs = []
        for step in range(args.num_steps):
            normed = nrm.normalize(norm_state, obs)
            key, k = jax.random.split(key)
            action, logp, value = ppo.select_action(ts, normed, k, critic_obs=normed)
            nobs, rew, done, info = env.step(np.asarray(jnp.clip(action, -1.0, 1.0)))
            if not env.last_finite:
                obs = env.reset(); ep_ret[:] = 0.0; ep_succ[:] = False
                continue
            dc = jnp.full(N, float(done), jnp.float32)
            buf.add(obs=normed, action=action, reward=jnp.asarray(rew) * args.reward_scale,
                    done=dc, truncation=dc, log_prob=logp, value=value)  # timelimit -> trunc=done
            raw_obs.append(np.asarray(obs))
            ep_ret += np.asarray(rew); ep_succ |= info["success"]
            obs = nobs
            if done:
                ep_rets.extend(ep_ret.tolist()); ep_succs.extend(ep_succ.astype(np.float32).tolist())
                ep_ret[:] = 0.0; ep_succ[:] = False
                obs = env.reset()

        # update obs-norm stats from this rollout's raw obs, then bootstrap + GAE + update
        norm_state = nrm.update(norm_state, jnp.asarray(np.concatenate(raw_obs, axis=0)))
        normed_next = nrm.normalize(norm_state, obs)
        key, k = jax.random.split(key)
        _, _, next_value = ppo.select_action(ts, normed_next, k, deterministic=True, critic_obs=normed_next)
        batch = buf.get(next_value, gamma=args.gamma, gae_lambda=args.gae_lambda)
        key, k = jax.random.split(key)
        ts, m = ppo.update(ts, batch, k, next_obs=normed_next)   # symmetric critic

        if (it + 1) % 10 == 0:                          # log + best-save every 10 iters
            recent = float(np.mean(ep_rets[-50:])) if ep_rets else float("nan")
            succ = float(np.mean(ep_succs[-200:])) if ep_succs else float("nan")
            envstep = (it + 1) * args.num_steps * N
            sps = envstep / (time.time() - t0)
            log(f"iter {it+1:6d}  envstep {envstep:9d}  ep_ret(mean50) {recent:8.2f}  "
                f"succ%(200) {100*succ:5.1f}  ploss {float(m.get('policy_loss', jnp.nan)):8.4f}  "
                f"vloss {float(m.get('value_loss', jnp.nan)):8.3f}  "
                f"ent {float(m.get('entropy', jnp.nan)):6.3f}  eps {len(ep_rets):5d}  {sps:7.0f} env-sps")
            snap = {"actor_params": jax.device_get(ts.actor_params),
                    "norm_state": jax.device_get(norm_state)}
            with open(out / "latest_actor.pkl", "wb") as f:   # current policy, every 10 iters
                pickle.dump(snap, f)
            if ep_rets and recent > best:
                best = recent
                with open(out / "best_actor.pkl", "wb") as f:
                    pickle.dump(snap, f)

    log(f"[done] {iters} iters ({iters*args.num_steps*N} env-steps), best ep_ret {best:.2f}")
    logf.close()


if __name__ == "__main__":
    main()
