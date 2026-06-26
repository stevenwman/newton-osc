"""Comprehensive single-env rollout benchmark: roll the SAME sq_vic policy through
full episodes (same seed, no noise, no recording) on 3 OSC backends —
  jax  : controllers.OSCController (jit'd jax)
  warp : controllers_warp.WarpOSCController (warp kernel)
  graph: warp + CUDA-graph-captured substep loop
Compares full-rollout wall-clock AND success/return parity (they should agree
modulo float32). This is the realistic eval workload (policy + set_action + sim +
obs every control step), not a step micro-benchmark.

    PYTHONPATH=. .venv/bin/python vic_rollout_bench.py --gain-mode axis --episodes 5
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse, pickle, time
import numpy as np, jax.numpy as jnp, optax, warp as wp
from jax_rl.algos.ppo import PPO
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.utils import normalization as nrm
from controllers import OSCController
from controllers_warp import WarpOSCController
from peg_env_square import PegEnv

ap = argparse.ArgumentParser()
ap.add_argument("--gain-mode", default="axis", choices=["fixed", "single", "axis"])
ap.add_argument("--episodes", type=int, default=5)
ap.add_argument("--episode-length", type=int, default=450)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
CKPT = f"runs/sq_vic_{args.gain_mode}/best_actor.pkl"


def build_ppo(env):
    cfg = PPOConfig(policy_hidden_dim=(256, 256), value_hidden_dim=(256, 256),
                    activation="swish", squash=True, state_dependent_std=False)
    cfg.num_envs = 1; cfg.minibatch_size = 1
    cfg.encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=(256, 256), activation="swish")
    cfg.critic_encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=(256, 256), activation="swish")
    cfg.policy_head = PolicyHeadConfig(action_dim=env.act_dim, squash=True, state_dependent_std=False)
    ppo = PPO(cfg, env.obs_dim, env.act_dim, optax.adam(3e-4), optax.adam(3e-4))
    L = pickle.load(open(CKPT, "rb"))
    return ppo, L["actor_params"], L["norm_state"]


def rollout(backend):
    Ctor = WarpOSCController if backend in ("warp", "graph") else OSCController
    c = Ctor(); c.action_mode = "delta"; c.gain_mode = args.gain_mode
    env = PegEnv(controller=c, episode_length=args.episode_length, weld=True, num_envs=1, seed=args.seed)
    ppo, actor, ns = build_ppo(env)

    if backend == "graph":
        env.reset(); c.set_action(env, np.zeros((1, env.act_dim), np.float32))
        env.capture_substep()                              # capture (advances rng — reseed below)

    # warm up policy/JIT
    o = env.reset()
    for _ in range(5):
        a = ppo.select_action_eval(actor, nrm.normalize(ns, jnp.asarray(o)))
        o, _, _, _ = env.step(np.asarray(a))
    env.rng = np.random.default_rng(args.seed)             # identical episodes across backends
    wp.synchronize_device()

    succ, rets, t0 = 0, [], time.time()
    for ep in range(args.episodes):
        o = env.reset(); s = False; R = 0.0
        for i in range(args.episode_length):
            a = ppo.select_action_eval(actor, nrm.normalize(ns, jnp.asarray(o)))
            o, r, d, info = env.step(np.asarray(a))
            s = s or bool(np.asarray(info["success"])[0]); R += float(np.asarray(r)[0])
        succ += int(s); rets.append(R)
    wp.synchronize_device()
    dt = time.time() - t0
    del env, ppo
    return dt, succ, float(np.mean(rets))


print(f"=== rollout bench: gain={args.gain_mode}, {args.episodes} eps x {args.episode_length} steps, seed {args.seed} ===")
base = None
for backend in ["jax", "warp", "graph"]:
    dt, succ, mret = rollout(backend)
    sps = args.episodes * args.episode_length / dt
    spd = f"{base/dt:.2f}x" if base else "1.00x (ref)"
    if base is None: base = dt
    print(f"  {backend:5s}: {dt:6.1f}s total | {sps:6.1f} control-steps/s | {spd:>10s} | "
          f"succ {succ}/{args.episodes} | ep_ret {mret:7.1f}")
