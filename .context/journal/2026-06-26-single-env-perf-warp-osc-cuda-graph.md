# Journal — 2026-06-26: single-env speed (latency, Warp OSC, CUDA-graph, decoupled recording)

Chased why single-env eval/recording felt slow (~70 physics-substeps/s, far below
CPU MuJoCo) and how training hits 900+ env-sps. Conclusion: it's **launch latency**,
training's only trick is **parallelism**, and the path to fast single-env is **Warp
OSC → CUDA-graph capture**. Recording: **decouple sim from render**.

## Diagnosis — latency-bound, not compute
- Per-control-step wall is ~**flat N=1→256** (55→63 ms — +14% for 256× the work) →
  fixed-overhead/launch-latency dominated, not compute. Per-env cost collapses
  55 ms→0.14 ms (400×) by amortization. Full A/B: **128 full episodes in +12% wall
  vs 1** (17→1957 env-sps).
- Training is fast **purely from parallelism** — verified there is NO graph capture
  / special compilation in the env or trainers (same `env.step` as recording). The
  "deeper than parallelism" instinct = the thing being amortized is *launch latency*,
  not raw compute.
- Profiling N=1: the **OSC's JAX** was 9 ms/substep vs the warp solver's 3.4 ms — the
  op-space math ran as dozens of eager `jnp` launches + ~8 `float(jnp...)` diagnostic
  lines forcing a device→host sync every substep.

## Three fixes (single-env control-step rate)
1. **JIT-fuse OSC** (`_osc_core` @jax.jit) + gate diagnostics behind `debug`:
   OSC 9→5 ms, control-step 50→33 ms (~1.5×). (commit 3676079)
2. **Warp OSC** (`controllers_warp.py`): OSC math as a `@wp.kernel` — one thread/world,
   in-kernel Cholesky-solve for the 7×7 qM and 6×6 Lam_inv (no explicit inverse),
   fingertip + rot_err in-warp, targets/gains pushed to wp arrays once per control-step
   in `set_action`. Per-substep path is **fully jax-free**. Parity vs JAX OSC: torque
   max|diff| **3.2e-3** (float32 cholesky-solve vs jax inverse — behaviorally identical).
   1.26× full step. (commit c51ae39)
3. **CUDA-graph capture** (`peg_env_square.capture_substep` / `use_graph`): now that the
   substep is jax-free, `wp.capture` the 4-substep loop, `wp.capture_launch` per step.
   Sim-only: substep **24.6→2.2 ms (11×)**. (commit ec9be79)

## Full-rollout benchmark (the real workload, not micro-steps)
`vic_rollout_bench.py` — roll the same `sq_vic_axis` policy, full episodes, same seed,
no noise. **Behavior float32-identical across all three (succ 5/5, ep_ret ≈2928.5):**

| backend | ctrl-steps/s | speedup |
|---|---|---|
| jax OSC (jit'd) | 24.0 | 1.00× |
| warp OSC | 30.2 | 1.26× |
| **warp + graph** | **116.9** | **4.88×** |

It's 4.88× (not the sim-only 11×) because the **jax wrapper** — policy inference +
`set_action` + obs read — is now ~half the per-step and stays uncaptured.

## Recording: decouple sim from render (`vic_decouple_record.py`)
Old "rawdog" (jax OSC + render every control step): **20.5 s**, 450 frames @30fps =
4× slow-mo. Decoupled (headless graph-warp sim → store `body_q` → offline render at
video fps, 113 frames): **4.6 s (4.47×)**, real-time video. Render was never the
bottleneck (0.7 s/113 frames); the old path rendered 450 frames interleaved with slow
sim. **Caveat:** two `ViewerGL` in one process → the 2nd renders black; the proper
decoupled form is two processes (sim→npz, then render). States stored ⇒ re-render
(camera/fps/overlay) free; batching the headless sim collects many trajectories at
~one sim cost. (commits d43c7e3 + this)

## Key結論 / gotchas
- **You cannot graph-capture with the JAX OSC** — `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`
  (jax issues on its own stream mid-capture). So the Warp OSC was the **prerequisite**
  for graph capture, not a nice-to-have. There is **no "graph the sim, keep jax OSC"
  middle rung** — it collapses to the jax rate (~24).
- The speedup ladder is **jax 24 → warp 30 → warp+graph 117 ctrl-steps/s**.
- Remaining ceiling = the uncaptured jax wrapper. XLA's own CUDA-graph ("command
  buffers", `--xla_gpu_enable_command_buffer`) could cut it but is **disabled for
  warp-interop safety** (re-enabling risks dlpack-shared-GPU hangs/races). Full
  unification would need porting the policy MLP + `set_action` to Warp too.
- Render is ~7% normally, but **after graph capture `get_frame` (~4 ms) is the bigger
  per-rendered-step cost** → another reason to decouple + render at video fps.
- Recording slowness was N=1 (no parallelism); **batching the sim is the same win**.

## Files added (experimental; JAX OSC stays the default everywhere)
`controllers_warp.py`, `vic_warp_test.py` (parity+timing), `vic_graph_test.py`
(sim-only capture micro), `vic_rollout_bench.py` (full-rollout 3-backend bench),
`vic_decouple_record.py` (decoupled recording), `peg_env_square` `capture_substep()`/
`use_graph`.
