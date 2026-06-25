# Square-peg / square-socket insertion — design doc

Plan for a copy of the peg-insert env where the **cylindrical peg → rectangular
prism** and the **annular bore → square socket** (decomposed into 4 prism walls
around a central square opening). Plan first; build after the open decisions (§11)
are confirmed.

---

## 1. What changes vs the cylindrical env (summary)

| Piece | Cylindrical (current) | Square (new) |
|---|---|---|
| Peg geom | `capsule` r=3.99mm, half-len 21mm | `box` half-extents [a, a, 21mm] (Warp supports box) |
| Hole geom | annular bore = **32 mesh tiles** + floor box | **4 box walls** + floor box (primitives, NO meshes) |
| Symmetry | yaw-symmetric (∞-fold) | square = **4-fold** (90°); rect = 2-fold (180°) |
| Reward | xy + tilt + z | xy + tilt + z **+ yaw (mod 90°)** ← new |
| Success | xy + tilt + depth | + yaw aligned (mod 90°) |
| DR | hole pos; yaw disabled (moot) | hole pos **+ yaw** (now matters) |
| Obs | peg quat + ee_angvel already expose yaw | **no change** (yaw already observable) |
| OSC / action / gain_mode | 6-DOF incl. rz | **no change** (rz already controls yaw) |
| Weld | peg↔hand, offset 0.130 | same (recompute offset only if peg length changes) |

**Net:** geometry is actually *simpler* (boxes, no mesh decomposition). The real
work is the **yaw degree of freedom** — the dominant new difficulty for learning.

---

## 2. THE core new difficulty: yaw alignment

The cylindrical peg is rotationally symmetric, so insertion only needs xy + tilt;
the current `peg_reward.py` has **no yaw term** and DR keeps hole yaw fixed (it's a
no-op for a round peg). A square peg in a square hole **must align its rotation
about z** with the socket.

- **Square cross-section → 4-fold symmetry**: 4 valid yaw targets, every 90°. The
  yaw error must be reduced **mod 90°** → `yaw_err ∈ [-45°, +45°]`.
- **Rectangular (a≠b) → 2-fold**: valid every 180°, reduce mod 180°.
- The 45° point (square) is a **saddle** equidistant between two valid orientations
  — a potential exploration trap; the smooth mod-90° reward should pull off it, but
  watch for a yaw local-minimum plateau in training.
- Yaw is **already observable** (peg quat in obs + ee_angvel) and **already
  controllable** (OSC rz axis), so only the *reward/success/DR* need yaw.

This is also where **variable impedance should shine**: lateral + yaw compliance
lets the square self-seat (corners guide), exactly the per-axis stiff-z/soft-xy/
soft-yaw strategy the `axis` policy already discovered for the round peg.

---

## 3. Geometry

### 3.1 Peg → box
- `<geom type="box" size="ax ay az">` (half-extents). Square: `ax = ay = a`.
  Length `az = 0.021` (match current half-len so weld offset/spawn unchanged).
- Cross-section: current capsule r≈4mm. Use `a ≈ 0.004` (8mm side) as the nominal;
  final value tied to clearance (§3.3).
- **Mass/inertia**: keep mass 0.019kg; replace the capsule's diaginertia with the
  **box** formula `I = m/12 · diag(ay²+az², ax²+az², ax²+ay²)` (×4 half-extents²).
  Wrong inertia → weld instability.
- `condim=6`, friction as current.
- Warp supports `box` natively (the capsule was only because Warp lacks `cylinder`).

### 3.2 Square socket → 4 walls + floor
Replaces the 32-tile ring. A square frame is non-convex → 4 convex box walls
around a central square opening (`s` = opening half-size):

```
        +y wall  (box: size [s+t, t/2, H/2], center y=+(s+t/2))
   ┌───────────────────┐
   │                   │
 -x│   square opening  │+x   (-x/+x walls: size [t/2, s, H/2], center x=∓(s+t/2))
   │     2s × 2s        │
   │                   │
   └───────────────────┘
        -y wall
   + bore_floor box below the opening (stops the peg at the bottom)
```
- Wall thickness `t` (~1–2mm), height `H` spanning z ∈ [−5mm, +35mm] like the bore.
- Corner overlap between walls is fine (static colliders). Top/bottom walls span the
  full width (s+t), left/right fit between them (height s) — or all overlap; either
  works.
- **All MJCF box primitives** → no `.obj` generation, no flat-mesh materialize for
  the hole. (Panda meshes still needed.) Simpler than the cylindrical build.

### 3.3 Clearance (key learnability knob)
- Current radial clearance ≈ bore r 4.05mm − peg r 3.99mm ≈ **60µm** (tight).
- Square box-box corners **jam harder** than round (sharp corners, no self-centering
  curvature). Recommend **start looser** (opening half `s = a + 0.3–0.5mm`) to get
  learning signal, then tighten toward 60µm once it solves. Optionally chamfer the
  opening (thin angled boxes) to reduce corner-catch — defer unless jamming blocks.

---

## 4. Reward (`peg_reward.py` additions)

Keep all existing terms (r_xy, r_tilt, altitude-weighted r_align, phase_below
descent/penalty, r_floor, r_success). **Add a yaw term**:

- Extract peg yaw: rotate peg local x-axis by `held_quat`, project to world xy,
  `peg_yaw = atan2(x_world.y, x_world.x)`. Hole yaw from `target_quat` similarly
  (or carried as a scalar through DR).
- `yaw_err = wrap(peg_yaw − hole_yaw, period=π/2)` for square (4-fold); `period=π`
  for rectangular. → `yaw_err ∈ [−period/2, +period/2]`.
- `r_yaw = squashing_fn(yaw_err, a, 0)` — bell peaked at aligned, like r_tilt.
- Fold into `r_align` (so descent is rewarded only when xy **and** tilt **and** yaw
  aligned): `r_align = k·(r_xy + r_tilt + r_yaw)·altitude_bonus`.
- Gate `r_B_desc` (descent) on `aligned_yaw` too (sigmoid on yaw_err), matching the
  existing aligned_xy / aligned_tilt gates — physics already blocks descent if the
  square is mis-yawed (corners hit the walls), so this is reward-shaping the gate
  that physics enforces.
- `is_success`: add `|yaw_err| < yaw_tol` (e.g. 2–5°) to the xy+tilt+depth gate.

Yaw period (90 vs 180) is a function of peg cross-section (§11).

---

## 5. Obs — no change
Peg quat (4) + ee_linvel/ee_angvel (6) + last/prev action already make yaw fully
observable. (Optional: append an explicit `sin/cos(2·yaw_err)` or `(4·yaw_err)`
feature to ease learning the mod-90° wrap — keep as a fallback if training stalls.)

---

## 6. Domain randomization (`peg_env` reset)
- **Enable hole yaw DR** (currently off): randomize hole yaw, e.g. ±15–45°, written
  into the hole pose (mocap quat / Newton `joint_X_p` rotation). Forces the policy to
  match arbitrary socket orientation rather than memorizing one.
- Hole xy/z DR + arm reset noise: unchanged.
- Note: hole yaw must propagate to the reward's `hole_yaw` (add to the carried hole
  pose, not just position).

---

## 7. OSC / action / controller — no change
6-DOF OSC controls rz → yaw authority exists. `gain_mode` (fixed/single/axis)
carries over unchanged. The square task is the natural showcase for the variable-
impedance comparison (yaw + lateral compliance to self-seat).

---

## 8. Code/asset changes (keep cylindrical env intact)

Recommend **copies/variants**, not mutating the working files:
- `assets/factory/square_insert/scene.xml` — peg box geom + inertia; hole_base with
  4 wall boxes + floor; weld; finger-peg excludes. (Copy peg_insert/scene.xml, swap
  geoms.) No mesh assets for the hole.
- `peg_scene_square.py` — like `peg_scene_newton.py` but: peg=box, hole=4 boxes
  inline (drop the 32-tile mesh injection + flat-mesh materialize for the bore).
  **Re-verify body/geom indices** (PEG_BODY_IDX, HAND_IDX, HOLE_JOINT) after
  add_mjcf — geom count changed.
- `peg_reward.py` — add yaw term **behind a flag/param** (so the cylindrical env is
  unaffected) OR a `square_reward.py` copy.
- `peg_env_square.py` (or parametrize `peg_env.py` with a `shape`/`yaw` flag) — hole
  yaw DR, success with yaw, reward wired with hole_yaw. Obs/controller reused as-is.
- `train_peg_flashsac.py` — reuse; just point at the square env (add `--env square`
  or a separate trainer that imports the square env). gain_mode/UTD knobs unchanged.

Decision on **separate files vs parametrize** in §11.

---

## 9. Risks / edge cases
- **Yaw saddle (45°)**: 4-fold symmetry exploration trap; mod-90° reward + yaw DR
  should mitigate, but watch for a succ plateau where peg seats translationally but
  mis-yawed.
- **Box-box jamming**: sharp corners catch worse than round at tight clearance →
  start loose (§3.3), tighten later; chamfer if needed.
- **Contact budget**: box-box face contacts (up to ~4–8 pts/wall × 4 walls) can
  exceed the round-peg contact count → may need to bump `nconmax` (currently 256
  per-world). Watch for `nefc/contact overflow` → NaN; re-tune like the earlier
  contact-budget work.
- **Box inertia**: must set correctly (§3.1) or the stiff weld goes unstable.
- **Index shifts**: PEG_BODY/HAND/HOLE indices after add_mjcf — re-verify (this bit
  us before; MEMORY notes add_mjcf drops welds/excludes — re-add manually).
- **Weld offset**: unchanged if peg half-length stays 21mm; recompute otherwise.

---

## 10. Build & validation plan (after §11 confirmed)
1. `scene_square.xml` + `peg_scene_square.py`; GL viewer: Franka holds box peg over
   the square socket, weld holds, peg seats (mirror `peg_scene_newton.py` checks).
2. Reward unit check: `r_yaw`/`is_success` peak at aligned mod 90°, behave at the
   45° saddle.
3. Env smoke (N=1,8,64): obs finite, reward in range, success fires only when
   seated **and** yaw-aligned; hole-yaw DR varies per reset.
4. Contact-budget check at N=512 (box-box) — bump nconmax if overflow.
5. Train: FlashSAC best recipe (UTD 64, N=128), **fixed** gain first → does it solve
   the yaw task? Then the **fixed/single/axis** variable-impedance comparison (expect
   per-axis to win bigger here — yaw+lateral compliance).
6. Record + plot (reuse `vic_analyze.py`/`vic_plot.py`; add a yaw-error trace).

---

## 11. Decisions (CONFIRMED 2026-06-25)
1. **Square** cross-section (4-fold, yaw mod-90°).
2. **Loose** clearance ~0.3–0.5mm first; tighten later.
3. **Separate `*_square` files** (cylindrical baseline stays pristine). Reward: add
   an *optional* `yaw_period` param to `compute_reward` defaulting to `None` (inert →
   cylindrical behavior bit-identical), square env passes `π/2`.
4. Hole-yaw DR **±30°**.
5. (default) Peg: 8mm square cross-section (a=0.004), half-len 0.021 (weld offset
   unchanged). 6. (default) yaw obs: quat-only; add explicit feature only if it stalls.

### original open-decisions list
1. **Peg cross-section**: square (a=a, 4-fold, mod-90° yaw) — recommended — or
   genuinely rectangular (a≠b, 2-fold, mod-180°)? "Rectangular prism" was said; a
   square prism is the simplest rectangular prism and the cleanest first target.
2. **Clearance**: start loose (~0.3–0.5mm, learnable) then tighten, or go tight
   (~60µm) immediately?
3. **Peg dimensions**: cross-section half `a` (≈4mm?) and length (keep 21mm?).
4. **Hole yaw DR range**: ±15°, ±30°, ±45° (full 4-fold cell)?
5. **Code layout**: separate `*_square` files (keeps cylindrical pristine, more
   duplication) vs parametrize existing env/reward with a `shape`/yaw flag (less dup,
   risk of touching the working baseline)?
6. **Yaw obs feature**: rely on quat alone, or add an explicit yaw-error feature?
