"""Smoke the batched PegEnv at N=1 (regression) and N=8 with the OSC controller."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

import time
import numpy as np
import jax.numpy as jnp
from controllers import OSCController
from peg_env import PegEnv


def run(N, steps=20, ep_len=8):
    ctrl = OSCController(); ctrl.action_mode = "delta"
    env = PegEnv(controller=ctrl, episode_length=ep_len, seed=0, weld=True, num_envs=N)
    print(f"\n[N={N}] obs_dim={env.obs_dim} act_dim={env.act_dim} ndof/world={env.ndof} "
          f"ncoord/world={env.ncoord} nbody/world={env.nbody}")
    obs = env.reset()
    print(f"[N={N}] reset obs shape={obs.shape} finite={bool(jnp.isfinite(obs).all())} "
          f"hole_pos[0]={np.asarray(env.hole_pos)[0]}")
    t0 = time.time()
    rews, succ_any = [], np.zeros(N, bool)
    ndone = 0
    for t in range(steps):
        a = np.random.uniform(-1, 1, (N, env.act_dim)).astype(np.float32)
        obs, rew, done, info = env.step(a)
        rews.append(np.asarray(rew))
        succ_any |= info["success"]
        assert obs.shape == (N, env.obs_dim), obs.shape
        assert np.asarray(rew).shape == (N,), np.asarray(rew).shape
        if not env.last_finite:
            print(f"[N={N}] NON-FINITE at step {t}"); break
        if done:
            ndone += 1
            obs = env.reset()
    dt = time.time() - t0
    rews = np.stack(rews)
    print(f"[N={N}] {steps} steps, {ndone} resets, {dt:.2f}s "
          f"({steps*N/dt:.0f} env-sps), reward range [{rews.min():.3f}, {rews.max():.3f}] "
          f"mean {rews.mean():.3f}  all-finite={np.isfinite(rews).all()}  succ_any={succ_any.sum()}")
    return rews


r1 = run(1)
r8 = run(8)
print("\nDONE")
