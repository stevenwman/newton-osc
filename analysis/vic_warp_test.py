"""Parity + timing: WarpOSCController vs JAX OSCController on the square env."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time, numpy as np, jax.numpy as jnp, warp as wp
from controllers import OSCController
from controllers_warp import WarpOSCController
from peg_env_square import PegEnv
import peg_scene_square as scene

# --- env with JAX OSC, step to a non-trivial state ---
jc = OSCController(); jc.action_mode = "delta"
env = PegEnv(controller=jc, episode_length=10000, weld=True, num_envs=1)
env.reset()
for _ in range(15):
    env.step(np.random.uniform(-1, 1, (1, env.act_dim)).astype(np.float32))

# warp controller sharing the same env/state; copy the jax controller's targets+gains
wc = WarpOSCController(); wc.action_mode = "delta"; wc.setup(env)
wc.target_pos = jc.target_pos; wc.target_R = jc.target_R
wc.kp_scale = jc.kp_scale; wc.zeta = jc.zeta
wc._push_targets()                                   # populate the wp buffers apply() reads

# --- parity: apply each on the SAME state, compare joint_f ---
jc.apply(env); jf_jax = env.control.joint_f.numpy().copy()
wc.apply(env); jf_warp = env.control.joint_f.numpy().copy()
err = np.max(np.abs(jf_jax[:7] - jf_warp[:7]))
print(f"[parity] tau jax  = {jf_jax[:7].round(3)}")
print(f"[parity] tau warp = {jf_warp[:7].round(3)}")
print(f"[parity] max|diff| = {err:.2e}  -> {'MATCH' if err < 1e-3 else 'MISMATCH'}")

# --- timing at N=1 (apply only) ---
for _ in range(10): jc.apply(env); wc.apply(env)     # warmup/JIT
wp.synchronize_device()
K = 300
t = time.time()
for _ in range(K): jc.apply(env)
wp.synchronize_device(); t_jax = (time.time() - t) / K * 1000
t = time.time()
for _ in range(K): wc.apply(env)
wp.synchronize_device(); t_warp = (time.time() - t) / K * 1000
print(f"[timing] apply  jax {t_jax:.2f} ms  |  warp {t_warp:.2f} ms  -> {t_jax/t_warp:.2f}x faster")
