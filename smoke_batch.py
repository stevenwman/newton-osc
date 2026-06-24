"""Smoke-test the batched PegEnv with the OSC controller.

Builds the env at a few world counts and checks: obs/reward shapes, finiteness,
synchronized resets, no contact/solver blowup, and rough throughput scaling.
This is the first thing to run on a new machine.

    PYTHONPATH=. .venv/bin/python smoke_batch.py
    PYTHONPATH=. .venv/bin/python smoke_batch.py 1 64 256   # custom world counts
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import sys
import time

import numpy as np
import jax.numpy as jnp

from controllers import OSCController
from peg_env import PegEnv


def run(N, steps=20, ep_len=8):
    ctrl = OSCController()
    ctrl.action_mode = "delta"
    env = PegEnv(controller=ctrl, episode_length=ep_len, seed=0, weld=True, num_envs=N)
    obs = env.reset()
    assert obs.shape == (N, env.obs_dim), obs.shape
    print(f"[N={N:4d}] obs{tuple(obs.shape)} reset_finite={bool(jnp.isfinite(obs).all())} "
          f"ndof/world={env.ndof} nbody/world={env.nbody} hole0={np.asarray(env.hole_pos)[0].round(3)}")

    t0, rews, succ_any, ndone = time.time(), [], np.zeros(N, bool), 0
    for t in range(steps):
        a = np.random.uniform(-1, 1, (N, env.act_dim)).astype(np.float32)
        obs, rew, done, info = env.step(a)
        assert obs.shape == (N, env.obs_dim) and np.asarray(rew).shape == (N,)
        rews.append(np.asarray(rew))
        succ_any |= info["success"]
        if not env.last_finite:
            print(f"[N={N:4d}] NON-FINITE at step {t}")
            return False
        if done:
            ndone += 1
            obs = env.reset()
    dt = time.time() - t0
    rews = np.stack(rews)
    print(f"[N={N:4d}] {steps} steps, {ndone} resets, {dt:.2f}s ({steps * N / dt:7.0f} env-sps)  "
          f"reward[{rews.min():.3f},{rews.max():.3f}] finite={np.isfinite(rews).all()} succ_any={int(succ_any.sum())}")
    return True


if __name__ == "__main__":
    counts = [int(x) for x in sys.argv[1:]] or [1, 8, 64]
    ok = all(run(N) for N in counts)
    print("DONE" if ok else "FAILED")
    sys.exit(0 if ok else 1)
