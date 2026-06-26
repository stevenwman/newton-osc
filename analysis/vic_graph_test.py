"""CUDA-graph capture of the (now jax-free) Warp-OSC substep loop, single env.
Capture one control-step (SUBSTEPS substeps) once, replay via wp.capture_launch,
compare wall vs uncaptured. set_action (jax) stays OUTSIDE the captured region."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time, numpy as np, jax.numpy as jnp, warp as wp
from controllers_warp import WarpOSCController
from peg_env_square import PegEnv

c = WarpOSCController(); c.action_mode = "delta"
env = PegEnv(controller=c, episode_length=100000, weld=True, num_envs=1)
env.reset()
c.set_action(env, np.zeros((1, env.act_dim), np.float32))   # populate wp target buffers
for _ in range(8): env._substep()                            # warmup / JIT all kernels
wp.synchronize_device()

# ---- capture one control-step (the 4-substep loop) ----
try:
    with wp.ScopedCapture() as cap:
        env._substep()
    graph = cap.graph
    captured = True
except Exception as e:
    print(f"[graph] capture FAILED: {type(e).__name__}: {str(e)[:160]}")
    captured = False

K = 300
# uncaptured warp substep loop
wp.synchronize_device(); t = time.time()
for _ in range(K): env._substep()
wp.synchronize_device(); t_unc = (time.time() - t) / K * 1000

if captured:
    wp.synchronize_device(); t = time.time()
    for _ in range(K): wp.capture_launch(graph)
    wp.synchronize_device(); t_cap = (time.time() - t) / K * 1000
    bq = env.state_0.body_q.numpy()
    finite = bool(np.isfinite(bq).all())
    print(f"[graph] control-step (4 substeps): uncaptured {t_unc:.2f} ms | captured {t_cap:.2f} ms "
          f"-> {t_unc/t_cap:.2f}x   (state finite={finite})")
    print(f"[graph] vs earlier: jax-OSC control-step ~38.9 ms, warp-OSC ~31.1 ms (full env.step incl obs)")
else:
    print(f"[graph] uncaptured control-step {t_unc:.2f} ms")
