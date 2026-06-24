# newton_manip — agent onboarding / run guide

Context for a fresh agent picking this repo up (esp. on a **bigger-GPU machine** to
run batched training at scale). Read `MEMORY.md` and `.context/lessons/` for the
deep history; this file is the "how to run it" quickstart + current state.

---

## What this repo is

RL **peg-insertion** on the **Newton** physics engine (mujoco_warp backend),
mirroring jax-learning's `FactoryPegInsert`. A Franka arm holds a peg (welded to
the gripper) and must insert it into a fixtured bore. Control is an
**operational-space controller (OSC)**; the learner is **FastSAC** (SAC + C51
distributional critic), vendored under `jax_rl/`. Everything (sim + RL) runs on
the GPU; JAX and Warp share the device via dlpack zero-copy.

The hard bug (a scrambled Jacobian) that made the OSC unable to hold position is
**fixed** — see `.context/lessons/osc-implementation-and-the-scrambled-jacobian.md`.
The controller now converges sub-mm.

---

## Environment setup

- **uv** project, Python **3.13**. Newton is installed from git (the PyPI
  `newton-physics` is an empty stub — do NOT use it):

  ```bash
  uv sync          # installs from pyproject/uv.lock (newton[examples] @ git+...)
  ```

  If recreating from scratch:
  `uv add "newton[examples] @ git+https://github.com/newton-physics/newton.git"`.
  The `[examples]` extra pulls mujoco-warp, mujoco, importers, the GL viewer, etc.
  RL deps: flax, optax, distrax (+ tfp-nightly), imageio[ffmpeg] (for replay mp4).

- **GPU**: developed on an RTX 5080 (16 GB, sm_120), Warp 1.14, newton 1.4.0.dev0.
  First run JIT-compiles many Warp/MuJoCo kernels (~30s+); cached in `~/.cache/warp/`.

- **VRAM sharing** (important): always keep
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (set in the scripts) so JAX doesn't grab
  75% of VRAM at init. `XLA_PYTHON_CLIENT_MEM_FRACTION` caps JAX's share. On a
  dedicated big GPU you can raise the fraction.

- **Run Python directly when in doubt**: `uv run` occasionally re-syncs the venv
  and strips pip-only installs (e.g. imageio). For probes and the recorder, call
  `.venv/bin/python` directly. Training via `uv run python ...` is fine.

---

## File map

| File | Role |
|---|---|
| `peg_scene_newton.py` | Builds the MJCF scene → Newton model. `build_model(num_envs=N)` replicates into N worlds. |
| `peg_env.py` | `PegEnv(num_envs=N)` — batched gym-like env. reset/seat/hole-DR/settle/obs/step + vmapped reward. |
| `controllers.py` | `OSCController` (6-DOF Khatib OSC → joint torques) and `JointPositionController`. OSC math is batched on a leading `nw` axis. |
| `peg_reward.py` | Phased insertion reward (byte-identical to jax-learning's factory reward). |
| `train_peg_osc.py` | FastSAC training loop. `--num-envs N` for batched. |
| `replay_record.py` | Roll out a checkpoint → mp4 (headless GL viewer). |
| `osc_track_demo.py`, `env_view.py` | GUI debug/visualization tools. |
| `jax_rl/` | Vendored FastSAC closure (algo, buffers, configs, networks). |
| `old/franka_batch_env.py` | The backend-validation harness that proved batched Newton+JAX throughput. |
| `MEMORY.md`, `.context/` | Durable notes + lessons. **Read these.** |

---

## How to run

### Smoke test the batched env (do this first on a new machine)
```bash
PYTHONPATH=. .venv/bin/python smoke_batch.py          # builds N=1,8,64; checks finite, resets, throughput
```
Expect: obs `(N,30)`, reward ~`[0.6, 1.2]` at the start pose, `all-finite=True`,
no NaN, env-sps scaling ~linearly with N.

### Train — single env (slow, sanity baseline)
```bash
uv run python train_peg_osc.py --steps 300000 --outdir runs/osc_peg
```

### Train — batched (the throughput win; this is the point of the bigger GPU)
```bash
uv run python train_peg_osc.py --num-envs 512 --steps 200000 \
    --episode-length 128 --buffer 1000000 --grad-updates 4 \
    --action-mode delta --outdir runs/osc_peg_batched
# total env-steps = steps * num_envs = 200k * 512 = 102.4M
```
Monitor: `tail -f runs/osc_peg_batched/train.log`. Columns: `iter`, `envstep`,
`buf`, `ep_ret(mean50)`, `succ%(200)`, `q_loss`, `actor`, `alpha`, `eps`, `env-sps`.
Checkpoints: `best_actor.pkl` (best mean return) + `ckpt.pkl` (full state, every 20k iters).

### Replay / record a checkpoint
```bash
PYTHONPATH=. DISPLAY=:1 .venv/bin/python replay_record.py --episodes 3 \
    --out runs/osc_peg_batched/replay.mp4         # add --stochastic to sample
```
Headless GL needs an X display (`DISPLAY=:1`) or EGL.

---

## Choosing num_envs (VRAM budget)

This scene is **collision-heavy** (32 bore-tile meshes per world), so per-world
VRAM is large (~15–20 MB/world for the Warp sim, replicated by `replicate()`).
Rough fit (Warp sim only, add ~1–3 GB for JAX nets+buffer):

| N | Warp VRAM (approx) | Notes |
|---|---|---|
| 64 | ~1.1 GB | validated end-to-end here |
| 128 | ~2.3 GB | |
| 256 | ~4.5 GB | OOM'd on the shared 16 GB dev box (6.7 GB taken by another user) |
| 512 | ~4–5 GB + | needs a mostly-free ≥12 GB GPU |

Contact budgets scale with N in `peg_env.py`: `nconmax=max(1024, N*32)`,
`njmax=max(1024*4, N*128)`. Measured need is small (~2 contacts, ~9 efc per world
with hover actions); a policy inserting against the bore tiles is higher, hence
the headroom. If you see `nefc/broadphase overflow` → NaN, bump these. `njmax`
drives a big `efc_J` allocation — don't oversize it (the original `N*256` OOM'd a
4 GiB array at N=512).

**Recommendation for a big GPU**: start `--num-envs 512`, confirm it builds +
the smoke is finite, then scale to 1024 if VRAM allows.

---

## RL config / the replay-ratio knob

`train_peg_osc.py` currently uses a moderate config (actor `(256,256)`, critic
`(512,512)`, 51 atoms, batch 256, `grad_updates_per_step=4`) — tuned when this was
single-env. For batched training, the meaningful quantity is the **replay ratio**
`grad_updates * batch / num_envs` (samples trained per collected sample). The
FastSAC paper (vendored `FastSACConfig` defaults) targets **8** with batch 8192,
UTD 8, atoms 101, nets `(512,256,128)`/`(768,384,192)`.

- Quick path: keep the current nets, set `--num-envs 512 --grad-updates 4` and
  raise `batch_size` to 1024 in the config → replay ratio `4*1024/512 = 8`.
- Faithful path: move the `FastSACConfig(...)` in `train_peg_osc.py` toward the
  paper defaults (batch 8192, grad_updates 8, atoms 101, wider nets) — heavier per
  iter but the regime FastSAC was designed for. Watch VRAM (big batch + 101 atoms).

`v_min/v_max` is `[-20, 20]`; widen if per-episode returns exceed that once the
policy starts seating (per-step reward max ≈ 7, episode max ≈ 900 for a sustained
insertion over 128 steps).

---

## Current state / open question

- **Batching is implemented and validated** at N=1/8/64: finite losses (q1 ~3,
  actor finite), no NaN, clean synchronized resets, reward matches single-env.
  The only reason it wasn't scaled here is the shared dev GPU's VRAM.
- **Episodes are synchronized**: `done = TimeLimit` only (no early termination,
  matching jax-learning), so all N worlds finish together and the whole batch
  resets at once — no per-world masking in the normal path.
- **Open question the bigger GPU should answer**: does batched training actually
  achieve insertion? The single-env run **plateaued at `succ% = 0.0`** (best
  ep_ret ~454 in v1, ~110 in v2) — it learns to align + approach but does not
  reliably seat. The hypothesis is that this is a data/exploration-budget problem
  that batching (100M+ env-steps) plus a paper-faithful replay ratio should fix.
  If succ% stays 0 at scale, revisit: reward shaping near the seat, the OSC
  orientation gain (rotation tracks slower than translation), `pos_bounds`/action
  scale, and adding a **privileged asymmetric critic** (jax-learning feeds the
  critic fixed_pos/quat, gains, thresholds — not yet ported here).

---

## Sharp edges (don't relearn these)

- **Never `pkill -f <name>` where `<name>` matches your own shell command** — it
  self-kills the agent shell. Kill training by PID.
- **mjw eq layout**: `mjw_model` is single-world; batching lives in
  `mjw_data.nworld`. `eq_solref`/`eq_solimp` are `(nworld, neq, …)`, weld = eq
  index **1**. The weld-stiffen `sr[..., 1, :]` already spans nworld — leave it.
- **`wp.from_jax` for vec3**: pass `dtype=wp.vec3` on an `(N,3)` array; the old
  `.reshape(-1).view(wp.vec3)` only worked at N=1.
- **Free-joint quat order is XYZW** in Newton state (`body_q`, `joint_q`); the
  reward wants **WXYZ** — convert at the boundary (`peg_env.step` does this).
- **Seat the peg before stepping** (`_seat_peg`) or the stiff weld snaps it.
- The GL viewer needs a display; relaunching over a live viewer can crash it.
