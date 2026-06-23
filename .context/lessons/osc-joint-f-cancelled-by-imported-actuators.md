# OSC "can't hold position" — imported MJCF actuators silently cancel `control.joint_f`

**Date:** 2026-06-23
**Files:** `bare_arm_view.py` (was broken, fixed), `peg_env.py` (settle fix), `controllers.py` (OSC, unchanged), `peg_scene_newton.py` (already correct)

## Symptom

The operational-space (OSC) controller "could not stably hold position." In the
isolation harness `bare_arm_view.py` the bare Franka arm drifted/flew off (hundreds
of mm) instead of holding the commanded fingertip pose. No amount of OSC tuning
(gains, ridge, fp64, nullspace) fixed it — because none of it was the problem.

## Root cause

`panda.xml` ships a stiff **position-PD actuator on every arm joint**:

```xml
<general gainprm="4500" biasprm="0 -4500 -450"/>   →  force = 4500·(ctrl − q) − 450·q̇
```

`ModelBuilder.add_mjcf()` imports these as **live actuators (kp=4500, kd=450) that
servo the joint to `ctrl` (default 0 = home)**. The OSC drives the arm through
`control.joint_f` (→ `qfrc_applied`, an *external* generalized force). The servo
torque and the OSC torque both land on the same joint, and the servo wins:

- Apply **+40 N·m** on joint0 via `joint_f` → joint moves only **40/4500 = 0.009 rad**
  and stops. The servo absorbs it. (Stripped of actuators: the same +40 N·m spins
  the joint to its limit.)
- OSC "hold" at the IK pose → arm gets **dragged 800 mm toward the servo's home
  pose** and parks there. Looked like divergence; was actually the servo pulling
  to `ctrl=0`.

So the OSC math was correct the whole time — its output never reached the joints.

This is the **same failure class jax_rl documents**: their factory config comment
says `actuator_mode="position_pd" silently disabled the OSC`, and their fix rewrites
the arm actuators in-place to **motor mode** (`gainprm=1`, `biasprm=0` → `ctrl` =
raw torque). The Newton port instead relies on leaving joints in NONE target mode
+ `joint_f`, which only works if the imported `<actuator>` block is removed.

## Why the isolation harness hid it

`bare_arm_view.py` was built to "isolate the OSC math from the peg scene." But it
did `b.add_mjcf(PANDA)` on the **raw** panda.xml — keeping the actuators. So the
isolation harness was itself silently broken, and every OSC experiment run in it
was measuring a controller whose torque was being cancelled. Dead end by
construction. **Lesson: when an isolation harness "can't be made to work no matter
what," suspect the harness/plumbing before the algorithm.**

## Diagnostic journey (including the wrong turns)

The instructive part is how many plausible-but-wrong hypotheses survived until a
*controlled* test killed them:

1. ❌ **Sparse mass matrix** — checked: model is dense (`nv ≤ 32` → `is_sparse=False`).
2. ❌ **DOF ordering / velocity convention** — verified Newton `joint_qd[:7]` ==
   mjw `qvel[:7]` exactly; arm is mjw `[0:7]`, `HAND_MJW_ID=9` correct.
3. ❌ **mjw_data staleness** — measured `stale=0.0`.
4. ❌ **fp32 vs fp64 Λ inversion** — bit-identical results.
5. ❌ **Ill-conditioning at the operating pose** — `cond(Λ⁻¹)≈2.2e5` at *all*
   poses **including the one that held**, so it can't be the discriminator.
6. ❌ **Arm pose / nullspace target** — claimed "both Newton's and jax's poses
   diverge"; this came from a **confounded harness** (a gravity-comp settle that
   itself injected ~11 rad/s before the target was captured). Not a clean test.
7. ✅ **Controlled actuation test** — apply a single constant torque and compare to
   free-fall. `+40 N·m → 0.009 rad`; `position-target → tracks`. That isolated the
   plumbing and exposed the actuators immediately.

**Lesson: stop theorizing on top of a harness you haven't validated end-to-end.
One clean controlled probe (constant input, measured output) beats five analytic
hypotheses.**

## The fix (three parts — all needed in the bare harness)

1. **Strip `<actuator>` (+ `<keyframe>`)** before building, so `joint_f` is the
   only torque input. (Done in-memory: regex-strip, absolutize `meshdir`, pass the
   XML string to `add_mjcf` — no file written.) *This is the root-cause fix.*
2. **Nullspace term + explicit joint-space damping (`KD_JOINT`).** A 6-DOF task on
   a 7-DOF arm leaves the redundant DOF uncontrolled; without joint damping it
   winds up and diverges **even with the actuators gone**. `KD_JOINT` is the
   essential stabilizer (nullspace alone doesn't save it). The real `OSCController`
   already has both; `bare_arm_view`'s inline `osc()` was a buggy partial copy that
   omitted them.
3. **Gravity-comp settle** (feed `qfrc_bias` into the arm dofs each substep) instead
   of a free-fall settle, so the hold target is captured at rest (no startup lurch).

Validated headless: bare arm holds at **0.00 mm**, `hand_z` flat over 60 steps
(was 700 mm runaway).

## Status per file

- **`bare_arm_view.py`** — rewritten with all three fixes. Holds. ✅
- **`peg_scene_newton.py`** — **already** strips `<actuator>`/`<keyframe>`
  (lines ~70-75). The peg env never had the cancellation bug.
- **`controllers.py` `OSCController`** — **already** has nullspace + `KD_JOINT`
  + ridge 1e-2. Unchanged. In the peg env it tracks its target to **< 0.2 mm**.
- **`peg_env.py`** — got the one transferable piece: the **gravity-comp settle** in
  `reset()` (was free-fall). Holds POSITION-mode dofs via servo targets and the
  OSC NONE-mode arm via gravity comp. ⚠️ *Code change not yet re-run in sim (GPU
  busy at time of writing); logic mirrors the headless-validated bare harness.*

## How to avoid / detect next time

- Treat **imported MJCF actuators** as part of the control plumbing. If you intend
  to drive joints with `joint_f` (torque), either strip the `<actuator>` block or
  rewrite it to motor mode (`gainprm=1`, `biasprm=0`). Otherwise the servo cancels
  your torque.
- Quick self-check for any torque-control harness: apply a large constant `joint_f`
  to one joint and confirm it actually accelerates. If it parks at
  `force/servo_kp`, an actuator is fighting you.
- Consider a guard in `peg_scene_newton` that asserts no `<actuator>` survived the
  strip, so this can't silently regress.

## Secondary (not a bug)

In the peg env, `OSCController.set_action` with **zero action** re-anchors
`target_pos` to the *current* fingertip every control step (the EMA/threshold
relative-action design, matching jax_rl). So a zero-action "hold" follows drift and
creeps ~18 mm / 5° over 120 steps. That's the action-space design, not an OSC
tracking failure. For a true static hold, capture the target once and don't
re-anchor.
