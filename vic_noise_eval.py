"""Observation-noise robustness eval for the square VIC PPO policies.

For each policy (fixed/single/axis) and each noise target (perception / goal / both),
sweep Gaussian obs-noise std (in NORMALIZED obs units) and record success rate.
Tests whether learned compliance (single/axis) tolerates perception noise better
than fixed. Dumps noise_eval.npz -> vic_noise_plot.py.

    PYTHONPATH=. .venv/bin/python vic_noise_eval.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import numpy as np, jax, jax.numpy as jnp, optax
from jax_rl.algos.ppo import PPO
from jax_rl.configs.ppo_config import PPOConfig
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.utils import normalization as nrm
from controllers import OSCController
from peg_env_square import PegEnv

N = 512
EP_LEN = 450
SIGMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]   # normalized-obs units; knee ~1-3
GAINS = ["fixed", "single", "axis"]
TARGETS = ["full"]   # perception/goal/both already in noise_eval.npz; merge full in
CKPT = {g: f"runs/sq_vic_{g}/best_actor.pkl" for g in GAINS}
OUT = "runs/sq_vic_noise"


def field_idx(A):
    """Obs index ranges given action_dim A. obs = arm_q(7) arm_qd(7) peg_rel(3)
    peg_quat(4) hole_pos(3) last_act(A) ee_lin(3) ee_ang(3) prev_act(A) goal_yaw(2)."""
    peg = list(range(14, 21))                       # peg_rel + peg_quat
    eevel = list(range(24 + A, 30 + A))             # ee_lin + ee_ang
    hole = list(range(21, 24))                      # hole_pos
    gyaw = list(range(30 + 2 * A, 32 + 2 * A))      # goal_yaw
    perception = peg + eevel
    goal = hole + gyaw
    full = list(range(32 + 2 * A))                  # every obs dim (incl. proprioception + actions)
    return {"perception": np.array(perception), "goal": np.array(goal),
            "both": np.array(sorted(set(perception + goal))), "full": np.array(full)}


def build(gm):
    ctrl = OSCController(); ctrl.action_mode = "delta"; ctrl.gain_mode = gm
    env = PegEnv(controller=ctrl, episode_length=EP_LEN, weld=True, num_envs=N, seed=0)
    cfg = PPOConfig(policy_hidden_dim=(256, 256), value_hidden_dim=(256, 256),
                    activation="swish", squash=True, state_dependent_std=False)
    cfg.num_envs = N; cfg.minibatch_size = 1
    cfg.encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=cfg.policy_hidden_dim, activation=cfg.activation)
    cfg.critic_encoder = EncoderConfig(obs_dim=env.obs_dim, hidden_dim=cfg.value_hidden_dim, activation=cfg.activation)
    cfg.policy_head = PolicyHeadConfig(action_dim=env.act_dim, squash=True, state_dependent_std=False)
    ppo = PPO(cfg, env.obs_dim, env.act_dim, optax.adam(3e-4), optax.adam(3e-4))
    import pickle
    with open(CKPT[gm], "rb") as f:
        loaded = pickle.load(f)
    return env, ppo, loaded["actor_params"], loaded["norm_state"], ctrl


def run_cell(env, ppo, actor_params, norm_state, idx, sigma, key):
    """One episode across N envs with Gaussian noise (sigma) on normalized-obs
    indices `idx`. Returns succ fraction."""
    obs = env.reset()
    succ = np.zeros(N, bool)
    mask = jnp.zeros((1, env.obs_dim)).at[0, idx].set(1.0) if len(idx) else jnp.zeros((1, env.obs_dim))
    for i in range(EP_LEN):
        normed = nrm.normalize(norm_state, jnp.asarray(obs))
        if sigma > 0:
            key, k = jax.random.split(key)
            normed = normed + sigma * mask * jax.random.normal(k, normed.shape)
        a = ppo.select_action_eval(actor_params, normed)
        obs, rew, done, info = env.step(np.asarray(a))
        succ |= np.asarray(info["success"])
        if done:
            break
    return float(succ.mean()), key


def main():
    os.makedirs(OUT, exist_ok=True)
    key = jax.random.PRNGKey(0)
    results = {}                                    # results[gm][target] = [succ per sigma]
    for gm in GAINS:
        env, ppo, ap_, ns_, ctrl = build(gm)
        idxs = field_idx(env.act_dim)
        results[gm] = {}
        for tgt in TARGETS:
            row = []
            for s in SIGMAS:
                sc, key = run_cell(env, ppo, ap_, ns_, idxs[tgt], s, key)
                row.append(sc)
                print(f"[noise] {gm:6s} {tgt:10s} sigma={s:.2f}  succ={100*sc:5.1f}%", flush=True)
            results[gm][tgt] = row
        del env, ppo
    # merge into existing npz (keep perception/goal/both already computed)
    path = os.path.join(OUT, "noise_eval.npz")
    save = {}
    if os.path.exists(path):
        old = np.load(path); save = {k: old[k] for k in old.files}
    save["sigmas"] = np.array(SIGMAS)
    for gm in GAINS:
        for tgt in TARGETS:
            save[f"{gm}__{tgt}"] = np.array(results[gm][tgt])
    np.savez(path, **save)
    print(f"[noise] saved {OUT}/noise_eval.npz")


if __name__ == "__main__":
    main()
