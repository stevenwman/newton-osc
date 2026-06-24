# Implementing OSC on Newton/mujoco_warp — the scrambled-Jacobian trap + the correct recipe

**Date:** 2026-06-24
**Files:** `controllers.py` (`OSCController`), `peg_env.py`, `osc_track_demo.py`

The peg-env operational-space controller (OSC) was unstable for a long time. The
root cause turned out to be a **one-line buffer-shape bug that scrambled the
Jacobian**, and almost every other "explanation" we chased (near-singular pose,
weld stiffness, payload inertia, contacts, the action frame, ridge, nullspace,
gravity comp) was a *downstream symptom* of that garbage Jacobian. This note
records the bug, the diagnostic signature, and the correct OSC recipe so we don't
relive it.

## The root-cause bug

`mujoco_warp.jac(model, data, jacp, jacr, point, body)` writes its output as
**`(nworld, 3, nv)`** — component-major, dof-minor (the kernel does
`jacp_out[worldid, component, dofid]`; see `mujoco_warp/_src/support.py` `_jac`).

The controller had allocated the output buffers as **`(nworld, nv, 3)`** and then
did a `.transpose(0, 2, 1)` to "fix" the layout. With `nv != 3` the kernel's dof
index overran the size-3 axis, so the Jacobian came back **scrambled** (only the
`dof∈{0,1,2}` corner was coincidentally valid), and the transpose compounded it.

```python
# WRONG: buffer (nw, nv, 3) + transpose  -> scrambled J
self._jacp = wp.zeros((nw, nv, 3), ...);  J = jnp.concatenate([jacp.transpose(0,2,1), ...])
# RIGHT: buffer (nw, 3, nv), no transpose
self._jacp = wp.zeros((nw, 3, nv), ...);  J = jnp.concatenate([jacp, jacr], axis=1)
```

**Why it was so destabilizing:** the OSC's task force `Jᵀ Λ (Kp·e − Kd·ẋ)` and the
EE velocity `ẋ = J·q̇` both run through `J`. A scrambled `J` gave a **wrong-sign EE
velocity**, so the damping term `−Kd·ẋ` became *negative damping* → energy pumped
in along the motion direction. It also corrupted `Λ = (J M⁻¹ Jᵀ)⁻¹`, inflating
`cond(J M⁻¹ Jᵀ)` to ~1e5 (which we misread as a "near-singular pose"). With the
shape fixed, `cond` drops to ~1e2 and the EE converges to **sub-mm** and holds.

## The diagnostic signature (memorize this)

> **Holds at ~zero task error, but diverges as soon as it must actively drive.**

At zero error the velocity term and the task force are both ~0, so a wrong `J`
doesn't bite — it "holds." Under motion the wrong-sign velocity damping diverges.
If an impedance/OSC controller has this signature, **suspect the velocity term and
the Jacobian first** (sign, frame, *shape/layout*), not the gains or the plant.

The decisive check that found it: **compare `mjw.jac`'s `J` to a finite-difference
Jacobian** (`J_fd[:,i] = (fk(q+εeᵢ) − fk(q))/ε`), and check `J·q̇` against the
finite-difference EE velocity `(x(t)−x(t−dt))/dt`. They disagreed in sign — done.
A controlled "apply +40 N·m to one joint, does it move as predicted?" probe and an
adversarial code review (a fresh agent told to *find the bug*) were what cracked it
after analysis alone stalled. Lesson: when analysis loops, build one clean
controlled measurement and/or get an adversarial second pair of eyes.

## The correct OSC recipe for this backend

Khatib operational-space control, per-control-substep (`SUBSTEPS=4`, `SIM_DT=0.002`):

1. **Refresh kinematics before reading.** mujoco_warp does *forward-then-integrate*,
   so between `solver.step()` calls `mjw_data.{xpos, xmat, qM, qfrc_bias, jac-inputs}`
   lag `qpos` by one substep. Reading the stale fields makes the gravity-comp
   feedforward do net work on the moving arm (energy injection — a settle that
   should reach rest instead climbs to several rad/s). Call
   `mujoco_warp.forward(mjw_model, mjw_data)` at the top of `apply()` so the fields
   are consistent with the current `qpos/qvel`.
2. **Jacobian:** geometric Jacobian (world frame) at the fingertip *point* on the
   hand body via `mjw.jac` — buffer `(nw, 3, nv)` (see above). Slice arm dofs
   `[:, :, :7]` (verify the arm really is mjw dofs 0..6 for your model; here the
   peg free-joint lands at dofs 9..14, so the slice is valid).
3. **Task wrench:** `F = Kp·e − Kd·ẋ`, with `ẋ = J·q̇` (geometric twist) and the
   pose error in the **world frame**. Position error `e = target − ft_pos`.
4. **Orientation error:** use the **robosuite cross-product form**
   `0.5·Σᵢ (R_cur[:,i] × R_des[:,i])` (world frame). It is singularity-free and
   smooth through 0; the axis-angle-via-matrix-log (`arccos`+`/sin`) is numerically
   fragile near 0/π and biases a near-zero-error hold. Pair the axis-angle error
   with the **geometric** angular velocity `ω = jacr·q̇` and `−Kd·ω` — this is
   canonically correct (robosuite/IsaacLab do exactly this); no analytic/mapping
   Jacobian is needed.
5. **Inertia & torque:** `Λ = (J M⁻¹ Jᵀ + ridge·I)⁻¹`, `τ_task = Jᵀ Λ F`. With the
   Jacobian correct, `cond` is healthy (~1e2) so a small `ridge` (1e-4, matching
   jax_rl) is fine and keeps a tight hold. A large ridge cross-couples
   position↔orientation and biases steady state — only crank it if genuinely
   ill-conditioned.
6. **Redundancy MUST be damped.** A 7-DOF arm under a 6-DOF (or 3-DOF) task has
   undamped null-space modes that wind up under excitation. Add a null-space
   posture term `N·M·(kp_null·(q_def − q) − kd_null·q̇)` and/or explicit joint
   damping `−KD_JOINT·q̇`. Without it even a *correct* minimal OSC diverges under
   driving. Note: a ridge-regularized `Λ` makes the projection `N = I − J̄J`
   slightly leaky into the task space — fine when `cond` is low, watch it if you
   raise the ridge.
7. **Gravity/Coriolis feedforward:** add `qfrc_bias` (= gravity + Coriolis) to the
   joint torque (`τ += bias`). Correct sign — it cancels the `−qfrc_bias` on the
   EoM RHS so `M q̈ = Jᵀ F + null`. (We do *not* compensate the op-space `J̇q̇`
   term; it's a minor second-order effect once `J` is right.)
8. **Apply** via `control.joint_f[:7]` → `qfrc_applied` (verified 1:1 for revolute
   joints; free joints route through `xfrc`, so `joint_f` on them is a no-op).

## Action space — stability vs. parameterization

- **Absolute base-frame target** works cleanly once `J` is correct: action ∈
  [-1,1]⁶ → a box around the bore (pos, base-frame axes) + axis-angle deviation
  from a nominal EE orientation. Converges from a 40 mm error to sub-mm and holds.
- jax_rl/IsaacLab instead build the target as **current_EE + (≤ threshold) delta**.
  That is *not* a bug — it's a slew/error limiter that keeps the OSC's task error
  small, which is the regime where even a marginal controller is stable. We
  originally mistook it for a random-walk bug. With a correct `J` you don't need it,
  but it's a good robustness guard for large commanded jumps.
- Tuning detail observed: position converges faster than orientation purely from
  the gain ratio (Kp_task = 100 pos vs 30 rot).

## Sharp edges checklist
- `mjw.jac` output is `(nw, 3, nv)` — never `(nw, nv, 3)`. Sanity-check `J` vs
  finite differences when porting.
- Read `mjw_data` *after* a `mjw.forward()` or accept one-substep-stale kinematics.
- "Holds at rest, diverges under motion" ⇒ velocity term / Jacobian, not gains.
- A suspicious `cond(J M⁻¹ Jᵀ)` (1e5+) at a non-singular pose ⇒ suspect a bad `J`,
  not the pose.
- Redundant arm ⇒ you *must* damp the null space.
- Big DLS ridge hides ill-conditioning but biases the hold and leaks the
  null-space projection.
- Free joints don't take `joint_f`; mjw mass matrix is dense only for `nv ≤ 32`.

See [[osc-joint-f-cancelled-by-imported-actuators]] for the *other* OSC gotcha
(imported MJCF position servos silently cancelling `joint_f`).
