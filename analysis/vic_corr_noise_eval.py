"""Temporally-correlated (sample-and-hold) obs-noise robustness for the square VIC
policies. Unlike the IID eval (vic_noise_eval), the noise vector is resampled every
HOLD_K control steps and held constant between — a persistent (zero-mean over time)
offset the OSC can't average away. Sweep the hold-length K at fixed sigma; compare
fixed/single/axis. K=1 == the IID case.

    PYTHONPATH=. .venv/bin/python analysis/vic_corr_noise_eval.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import numpy as np, jax, jax.numpy as jnp
from jax_rl.utils import normalization as nrm
import vic_noise_eval as V          # sibling: reuse build() + field_idx()

N = 512
EP_LEN = 450
SIGMA = 1.5                          # normalized-obs units; IID-survivable -> isolate correlation
HOLD_KS = [1, 3, 5, 10, 20, 50, 150, 450]   # control steps the noise is held (1 = IID)
GAINS = ["fixed", "single", "axis"]
TARGET = "both"                      # perception + goal
OUT = "runs/sq_vic_corr_noise"


def run_cell(env, ppo, actor_params, norm_state, idx, sigma, hold_k, key):
    """One episode/env with sample-and-hold noise on `idx`: redraw z every hold_k
    control steps, hold between. Returns succ fraction over N envs."""
    obs = env.reset()
    succ = np.zeros(N, bool)
    mask = jnp.zeros((1, env.obs_dim)).at[0, idx].set(1.0)
    z = jnp.zeros((N, env.obs_dim))
    for i in range(EP_LEN):
        if i % hold_k == 0:                         # resample held noise
            key, k = jax.random.split(key)
            z = jax.random.normal(k, (N, env.obs_dim))
        normed = nrm.normalize(norm_state, jnp.asarray(obs)) + sigma * mask * z
        a = ppo.select_action_eval(actor_params, normed)
        obs, rew, done, info = env.step(np.asarray(a))
        succ |= np.asarray(info["success"])
        if done:
            break
    return float(succ.mean()), key


def main():
    os.makedirs(OUT, exist_ok=True)
    key = jax.random.PRNGKey(0)
    save = {"hold_ks": np.array(HOLD_KS), "sigma": SIGMA}
    for gm in GAINS:
        env, ppo, ap_, ns_, ctrl = V.build(gm)
        idx = V.field_idx(env.act_dim)[TARGET]
        row = []
        for hk in HOLD_KS:
            sc, key = run_cell(env, ppo, ap_, ns_, idx, SIGMA, hk, key)
            row.append(sc)
            print(f"[corr] {gm:6s} hold_k={hk:4d}  succ={100*sc:5.1f}%", flush=True)
        save[gm] = np.array(row)
        del env, ppo
    np.savez(os.path.join(OUT, "corr_noise.npz"), **save)
    print(f"[corr] saved {OUT}/corr_noise.npz")


if __name__ == "__main__":
    main()
