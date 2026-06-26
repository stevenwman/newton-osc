# Journal — 2026-06-24/25: FlashSAC, the UTD lever, variable impedance, square task

Two days picking the project up on the RTX 5090 (32 GB). Went from "succ% stuck at
0" to a fully solved peg-insertion (100%), found the real sample-efficiency lever,
ran a variable-impedance study, and built a new square-socket task. Durable facts
are in MEMORY.md / BEST_RECIPE.md / SQUARE_PEG_PLAN.md; this is the narrative.

---

## Day 1 (2026-06-24) — scale on the 5090, then find the algo gap

**Scaled batched training; fixed two contact-budget OOMs.** Fresh checkout, `uv
sync`, RTX 5090 32 GB. Hit an 18 GB `epa_pr` alloc at N=512. Root cause = wrong
contact-budget scaling: in this mujoco_warp `nconmax` is **per-world** (naconmax =
nconmax·nworld) while `njmax` is **total**. Old `nconmax=N*32` grew N² via the
internal ·nworld. Fix: `nconmax=256` constant. Then a second O(N²) trap — the dense
`efc_J ≈ njmax·nv_total`, both ∝N — OOM'd 32 GB at N=2048; fixed with `njmax=N*32`.
N=2048 then trained at ~14k env-sps. (Commits c097b29.)

**Recorded the best policy → diagnosed the plateau.** succ%=0 wasn't a sim/OSC/
reward bug — the policy parked the peg ~4 cm *above* the bore and refused to
descend (hover trap). The reward file's own comments blamed "Q flat in z"; we
confirmed it was an **algorithm** problem.

**FlashSAC was the missing piece.** The jax-learning run that solves FactoryPegInsert
uses **FlashSAC** (BatchNorm + weight-norm residual blocks, adaptive reward
normalization, Zeta-noise exploration, asymmetric-capable C51), not FastSAC.
Vendored it (`jax_rl/algos/flash_sac.py` + flash_blocks + reward_scaling; reused
distributions/distributional/polyak) and wrote `train_peg_flashsac.py` matching the
converged run's config JSON exactly.
- **P1** (algo swap on the unchanged env/OSC/30-d obs/v18 reward): **95.5% success**
  at 2.3M steps, deterministic eval ep_ret ~2860 (≈ jax_rl ~3000). The algorithm
  was the dominant gap. (Commit d36c7c1.)
- **P2** (additive obs: + EE linvel/angvel finite-diff + prev_action → 42-d):
  **~23% better sample efficiency** (90% at 1.47M vs P1's 1.92M), 100% / ep_ret
  2986. (Commit 488d852.)

---

## Day 2 (2026-06-25) — the UTD lever, variable impedance, square task

**UTD is THE sample-efficiency lever.** Disentangled the conflated "replay ratio"
into **UTD = grad_updates/N**, **batch**, and **ratio = UTD·batch**. Sweep at
N=128, batch 2048:

| UTD (grad) | env-steps to 90% | grad-steps to 90% |
|---|---|---|
| 16 | 1.47M | 184k |
| 32 | 704k | 176k |
| 64 | **448k** | 224k |

UTD-64 **beat jax_rl's ~830k** by ~1.85×. Convergence is **grad-step bound**
(~180–220k regardless); higher UTD packs those into fewer env-steps. Wall-time ~flat
(~17–19 min) since it's grad-bound. FlashSAC's normalization is what lets UTD scale
without the replay-ratio barrier.

**Negative results (don't repeat):**
- N=512 batch 8192 grad 16 (P3): undertrained — fixed grad at high N = 4× fewer
  optimizer steps.
- N=256 grad 64 (P6, UTD matched P4): ~neutral; only a small wall gain from GPU util.
- N=256 grad 128 (P7, UTD matched P5): **worse** per grad-step — scaling envs+grad
  together regresses (bigger-N cohorts → more correlated batches).
- Bigger buffer: no help — at 1M it already holds ~17 full episodes, so the replay
  buffer is already phase-saturated (killed the staggered-reset idea too).

**Variable-impedance OSC** (`controllers.py` gain_mode: fixed / single / axis — the
policy modulates task-space Kp and damping ratio ζ, Kd=2ζ√Kp, log-scale [0.5,2]×).
3-way comparison (UTD 32, N=128):

| mode | act_dim | succ% | final ep_ret | 90% crossover |
|---|---|---|---|---|
| fixed | 6 | 100 | 2855 | 704k |
| single | 8 | 98 | 2757 | 960k |
| **axis** | 18 | **100** | **3048** | 832k |

Per-axis **wins on quality** and learned the textbook insertion strategy: **stiff
along z** (insertion axis, underdamped early for fast descent), **compliant in x,y**
(~0.78×) so the bore self-aligns the peg. Plots + recordings in `runs/vic_*`,
analysis tooling `vic_analyze.py` / `vic_plot.py`. (Peg force read from
`qfrc_constraint[9:12]` — `cfrc_ext` is unpopulated in this mujoco_warp.)
(Commits b286f40, 3b6a2e6.)

**New square-socket task** (`SQUARE_PEG_PLAN.md` — planned before building).
- Geometry is *simpler* than the round bore: a square slab with a centered square
  hole = a square annulus, tiled by **4 pinwheel prisms** (each (L+s)×(L−s),
  rotated 90°, exact tiling, no meshes) + blind floor. 88 mm slab (10× the 8.8 mm
  hole), box peg (8×8×42 mm).
- The real new difficulty is **yaw**: a square peg/hole isn't rotationally
  symmetric (4-fold), so the reward/success/DR gained a yaw term (mod 90°); obs +
  controller unchanged (peg quat + ee_angvel expose yaw; OSC rz controls it).
- Spawn raised +10 cm (FK-solved ARM_Q) so the box peg clears the wide slab top.
- Box-on-flat-wall contact makes ~87 efc/world (vs ~9 round) → bumped njmax to
  N*128 (env) / 1024 (joint_play) after a "nefc overflow → increase njmax to 87"
  during a manual slam in the new `joint_play_square.py` slider GUI.
- Files: `peg_scene_square.py`, `peg_env_square.py`, `peg_reward.py` (optional yaw
  term, inert for cylindrical), `train_peg_flashsac.py --env square`. (Commit 7056791.)

**In progress:** training the fixed-stiffness square policy with the best recipe
(FlashSAC UTD 64, N=128) in `runs/square_fixed` — comparing the square/yaw task to
the cylindrical baseline (cylindrical UTD-64 hit 90% at 448k; expect square later /
lower given the extra yaw alignment).

---

## Key takeaways
1. The succ%=0 plateau was the **algorithm** (FastSAC→FlashSAC), not env/OSC/reward.
2. **UTD** is the sample-efficiency knob; parallel envs / buffer size are not
   (grad-step bound, buffer phase-saturated). Wall-time is ~flat.
3. Per-axis **variable impedance** gives the best final policy and a physically
   interpretable stiff-z/soft-xy strategy.
4. **Box-box contact** needs far more solver constraint rows than round-peg contact
   — size njmax from the observed nefc, mind the O(N²) efc_J at high N.

---

## Day 2 cont. — PPO migration, the square solve, variable impedance + noise robustness

**Vendored PPO (on-policy) alongside FlashSAC** to compare regimes (commits a4b4a60,
cf2ce67). Minimal closure: `jax_rl/algos/ppo.py`, `configs/ppo_config.py`,
`buffers/rollout.py` (GAE), `utils/normalization.py`; networks/heads/distributions
shared. `train_peg_ppo.py` = on-policy loop (collect num_steps×N rollout → GAE →
num_epochs×num_minibatches clipped updates, running obs-norm). `--env {peg,square}`.
Our action is OSC-bounded → squash=True (vs jax_rl's joint-space squash=False).

**PPO vs FlashSAC — circular peg:** both solve. FlashSAC ~10× more sample-efficient
(90% @ 448k vs PPO ~4.6M), but PPO's ~2.7× higher env-throughput (on-policy, few
grads/step: ~1050 vs 390 env-sps) nearly cancels it → **wall-clock ~tie (~17–25 min
to 2k reward)**. Sample efficiency favors FlashSAC; throughput favors PPO.

**The square SOLVE — goal-yaw observability was the real unlock.** Both algos
stalled on the square (FlashSAC 8.5%, PPO ~30%). Root cause: the obs gave the peg's
OWN orientation but the GOAL as **position only** — the ±30°-randomized socket yaw
was **invisible**, so a square peg that must match it could only guess. Fix: append
`[cos,sin](hole_yaw)` to the square obs (commit d3da1f3). PPO then **solved it:
30%→99.5%** (crossed 50% at 2.95M vs never-in-8M blind). The square needed, in impact
order: (1) **goal-yaw in obs** (the unlock), (2) bigger margin (16mm peg / 2.4mm/side,
sized by eye via `peg_in_hole_view.py`), (3) box-box njmax fix.

**PPO > FlashSAC on the square.** With goal-yaw, FlashSAC improved (8.5%→71.5% peak)
but **plateaus ~70%** (off-policy hover/yaw local optima); PPO solves 99.5%.
On-policy + entropy explores the contact/yaw landscape better. So: FlashSAC for the
easy circular (sample-efficient), **PPO for the hard contact+yaw square**.

**Square variable-impedance 3-way (PPO, commit 45a81b3):** fixed/single/axis all
SOLVE (100/99/98.5%). Unlike the round peg's steady stiff-z/soft-lateral, the square
axis policy **spikes all gains ~1.9× during the insertion transient then relaxes**
("stiffen to drive in, soften to hold"). VIC pushes HARDER (|F|max 0.28→0.63 N) —
counterintuitive until you note the gain range is symmetric [0.5,2]×, so the policy
exploits the STIFF half (2× fixed's authority) to seat the square against corners.
New viz: `vic_ellipsoid_record.py` renders a transparent **wireframe compliance
ellipsoid** at the EE (radii ∝ 1/Kp; log_lines has no alpha → wireframe = see-through).

**Obs-noise robustness (commit b531418):** `vic_noise_eval.py` sweeps Gaussian noise
(normalized-obs units) on perception / goal / both, succ% over 512 eps/cell. Result:
**per-axis VIC is the MOST noise-robust, global-scalar (single) the LEAST (worse than
fixed).** σ@50%: perception fixed 3.0 / single 3.0 / **axis 4.0**; at σ=3 perception
axis holds 64% vs fixed 25% vs single 10%. Per-axis compliance forgives perception
error (soft directions absorb corrupted targets); one global gain can't adapt
directionally. Goal-noise gentle for all (low-dim). Plot: `runs/sq_vic_noise/`.

### More takeaways
5. **Observability beats algorithm tuning** — the square was unsolvable because the
   policy couldn't see the goal yaw it had to match. Check the obs exposes the task.
6. **PPO (on-policy) > FlashSAC (off-policy) on the contact+yaw square**; reverse on
   the easy circular (sample efficiency). Algorithm choice is task-dependent.
7. **"Variable impedance" ≠ softer** — with a symmetric [0.5,2]× range the policy
   stiffens beyond fixed to seat (higher peak force), AND it's the most obs-noise
   robust. Per-axis is the win; global-scalar (single) hurts robustness.
