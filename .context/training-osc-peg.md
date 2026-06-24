# Training the OSC peg env (jax_rl-style FastSAC) — outline + status

This sets up RL training on the Newton peg-insertion env with the (now correct)
6-DOF operational-space controller, mirroring jax_rl's FactoryPegInsert scheme.

## What's running (overnight)

```
uv run python train_peg_osc.py --steps 400000 --episode-length 128 \
    --outdir runs/osc_peg --buffer 200000 --min-buffer 2000
```

Launched detached (`nohup`); ~17 sps → ~6 h for 400k steps. Health (early):
finite losses, q_loss decreasing (3.9→1.5), **~0.7 GB VRAM** (a good neighbour —
user3 is using ~6.7 GB on the shared GPU).

- `runs/osc_peg/train.log` — progress (step, buffer, ep_ret mean10, q/actor loss, sps)
- `runs/osc_peg/run.out` — raw stdout (warp/jax noise + the log)
- `runs/osc_peg/best_actor.pkl` — actor params at the best ep_ret-so-far
- `runs/osc_peg/ckpt.pkl` — full train_state every 20k steps (for resume/eval)

Monitor: `tail -f runs/osc_peg/train.log`. Kill: `pkill -f train_peg_osc`.

## How it mirrors jax_rl FactoryPegInsert

Same off-policy skeleton as jax_rl's loop: **warmup (random) → collect → replay
buffer → FastSAC.update** with a **C51 distributional critic** (`num_atoms=51`),
twin critics, delayed actor (`policy_delay=2`), automatic temperature `alpha`.
Reward = the ported phased peg reward (`peg_reward.py`). Action = jax_rl's 6-DOF
OSC pose command (here: absolute base-frame target → Khatib OSC torques). FastSAC
closure is vendored under `jax_rl/` (see [[fastsac-training-pipeline]]).

Config (`train_peg_osc.py`): actor MLP (256,256), critic (512,512), batch 256,
`grad_updates_per_step=4`, gamma 0.97, adamw 3e-4. Symmetric critic (critic_obs =
obs); no privileged critic yet.

## Where it deliberately differs from jax_rl (and what to do for "real" training)

1. **Single env, not batched.** jax_rl runs hundreds–thousands of parallel envs
   (mjx vmap). Single-env at ~17 sps will train slowly and is mainly a pipeline /
   sanity run. **Biggest lever:** batch via `ModelBuilder.replicate(builder, N,
   spacing=0)` + `SolverMuJoCo` (the backend-validation harness already proved
   ~1–2 M sim-steps/s at N=1024–4096 — see [[backend-validation-batched-rl-jax]]).
   The OSC math in `controllers.py` is already written batched (leading `nw` axis);
   the env (`peg_env.py`) needs the per-world reset/obs/action routing from the
   batch harness. This is the main TODO to get jax_rl-class throughput.
2. **Obs is symmetric (21-d): arm q/qd + peg-rel-hole + peg quat.** jax_rl adds a
   privileged critic (fixed_pos, gains, thresholds) + richer actor obs. Add a
   `critic_obs` with privileged state and pass it through `buf` for asymmetric AC.
3. **No EMA / slew on the action.** jax_rl EMA-smooths actions (`ema_factor=0.2`)
   and keeps the target within one delta of the current EE. Our OSC uses a pure
   absolute setpoint (works now that the Jacobian is fixed); consider adding EMA
   back for smoother exploration if SAC thrashes the target.
4. **No hole domain randomization beyond the small ±2cm/±1cm reset DR** and no
   action/obs normalization, eval, or wandb. Add these for a serious run.
5. **C51 v_min/v_max** may need widening if returns exceed the configured range.

## Result of the first 400k-step run (single env)

Ran ~6 h (400k steps, 17.9 sps, single env). best ep_ret 454 (stochastic);
deterministic eval ~170–300. **What the policy learned:** align above the bore
(xy + tilt) and drive the peg down to the bore opening (`peg_min_z` ≈ 0.063–0.080;
bore top ≈ 0.075) — but it does **not** reliably *seat* (would need peg z < ~0.055
with `r_success`). So: alignment + approach solved, full insertion not yet. The
reward is the v18 phased reward (`peg_reward.py`): per step ≈ `r_align`(≤2, aligned
above) + `r_B_desc`(≤1, aligned descent) + `r_floor`(≤2, at seated z) +
`r_success`(≤5, fully seated). Over 128 steps: hover-aligned ≈ 256, full sustained
insertion ≈ ~900. This is consistent with the single-env / 400k-step budget being
small; batching + more steps is the expected path to insertion (jax_rl's 6-DOF runs
plateau ~880 and the tuned 3-DOF hit 6545 with far more env steps).

## Record a video / evaluate a checkpoint

`replay_record.py` rolls out `best_actor.pkl` (or `ckpt.pkl`) and records an mp4 via
the headless GL viewer (`viewer.get_frame()` -> imageio, same as
jax_rl/projects/mud_eval). Needs `imageio[ffmpeg]` (installed in .venv; add to
pyproject for reproducibility). Run with the venv python + a display:

    PYTHONPATH=. DISPLAY=:1 .venv/bin/python replay_record.py --episodes 3 \
        --out runs/osc_peg/replay.mp4            # add --stochastic to sample

(`uv run` sometimes re-syncs/strips the pip-installed imageio; calling
`.venv/bin/python` directly is more reliable. Headless GL still needs an X display
or EGL — `DISPLAY=:1` works here.)

## Evaluate a checkpoint (programmatic)

`best_actor.pkl` holds the actor params. To roll it out deterministically, load
the pickle, `algo = FastSAC(...)` with the same cfg/dims, and call
`algo.select_action(actor_params, obs[None], key)` in an env loop (see the collect
branch of `train_peg_osc.py`). Resume from `ckpt.pkl` (full train_state) by
`pickle.load` → continue the loop.

## VRAM / sharing the GPU

`train_peg_osc.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (JAX on-demand, no
75% grab) and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2` (cap). Measured ~0.7 GB in
practice for single-env. Override the cap via the env var if you batch up
(batched warp sim will dominate VRAM, not JAX). The other big consumer is the warp
contact budget (`nconmax=1024, njmax=4096` in `peg_env.py`) — needed to avoid
contact overflow → NaN; the training loop also has a non-finite-step guard that
resets instead of crashing.

## Open items / next steps
- [ ] Batch the env (`replicate`) for throughput — the single biggest win.
- [ ] Privileged asymmetric critic obs.
- [ ] Eval/checkpoint-rollout script + success-rate metric (peg seated).
- [ ] Revisit OSC action: absolute vs EMA-delta for exploration; orientation gain.
- [ ] Longer run once batched (jax_rl uses millions of env steps).
