"""Bare Franka arm (no peg/bore/weld) + OSC hold, PAUSED start. You drive it.

    uv run python bare_arm_view.py

Builds the panda arm alone, commands OSC to HOLD the initial fingertip pose,
starts paused. Step in the GUI and watch hand_z + pos_err (should stay ~constant).
Isolates the OSC math from the peg/bore/weld scene.

WHY THIS WAS BROKEN (and is now fixed): panda.xml ships stiff position-PD
actuators on every arm joint (`<general gainprm="4500" biasprm="0 -4500 -450">`
= kp=4500/kd=450 servo to ctrl=0). add_mjcf imports them ACTIVE, so they hold
the arm at its home pose and CANCEL the OSC's control.joint_f torque (apply
+40 N.m -> joint moves 40/4500 = 0.009 rad). The OSC looked like it "couldn't
hold" because its torque never reached the joints. Fix mirrors peg_scene_newton
(and jax_rl's "motor mode"): strip <actuator> before building so joint_f is the
only input. Two further requirements for a STABLE 6-DOF-task / 7-DOF-arm hold:
  - the nullspace term + explicit joint-space damping (KD_JOINT) — without them
    the redundant 7th DOF is undamped and winds up (diverges even with the
    actuators gone);
  - gravity-comp settle (not free-fall) before capturing the target, so the
    hold target is grabbed at rest.
"""
import os
import re

import jax.numpy as jnp
import numpy as np
import warp as wp

import mujoco_warp as mjw
import newton
import controllers as C

PANDA = "assets/factory/franka_panda/panda.xml"
ARM_Q = np.asarray(C.NULLSPACE_Q, np.float32)   # seat + nullspace target (well-behaved hold pose)

# Strip the position-PD actuators (they would cancel joint_f) and the keyframe
# (its ctrl size no longer matches), then build from the in-memory string with
# an absolute meshdir so meshes still resolve. Same trick as peg_scene_newton.
_xml = open(PANDA).read()
_xml = re.sub(r"<actuator>.*?</actuator>", "", _xml, flags=re.DOTALL)
_xml = re.sub(r"<keyframe>.*?</keyframe>", "", _xml, flags=re.DOTALL)
_xml = _xml.replace('meshdir="assets"',
                    f'meshdir="{os.path.abspath("assets/factory/franka_panda/assets")}"')

b = newton.ModelBuilder()
b.add_mjcf(_xml)
model = b.finalize()
nb = [l.split("/")[-1] for l in b.body_label]
HAND = nb.index("hand")

# Seat the arm at ARM_Q before stepping.
q = model.joint_q.numpy()
q[:7] = ARM_Q
model.joint_q.assign(q)

solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True,
                                     nconmax=512, njmax=1024, iterations=100, ls_iterations=50)
s0 = model.state()
s1 = model.state()
ctl = model.control()
newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

# Gravity-comp settle (apply qfrc_bias each substep) so the arm holds at ARM_Q
# while mjw_data populates, then we capture a zero-velocity target.
for _ in range(30):
    s0.clear_forces()
    wp.copy(ctl.joint_f, wp.from_jax(jnp.asarray(wp.to_jax(solver.mjw_data.qfrc_bias)).reshape(-1).astype(jnp.float32)))
    solver.step(s0, s1, ctl, None, 0.002)
    s0, s1 = s1, s0

d = solver.mjw_data
mm = solver.mjw_model
nv = d.qvel.shape[1]
xp0 = np.asarray(wp.to_jax(d.xpos).reshape(-1, 3))
HAND_MJW = int(np.argmin(np.linalg.norm(xp0 - s0.body_q.numpy()[HAND][:3], axis=1)))
jp = wp.zeros((1, nv, 3), dtype=wp.float32, device="cuda:0")
jr = wp.zeros((1, nv, 3), dtype=wp.float32, device="cuda:0")
body_arr = wp.array([HAND_MJW], dtype=wp.int32, device="cuda:0")
NULL_Q = jnp.asarray(s0.joint_q.numpy()[:7])


def fingertip():
    xp = wp.to_jax(d.xpos).reshape(1, -1, 3)
    xm = wp.to_jax(d.xmat).reshape(1, -1, 3, 3)
    # Keep orientation as a matrix (avoid the mat<->quat singularity near gripper-down).
    return xp[:, HAND_MJW] + jnp.einsum("nij,j->ni", xm[:, HAND_MJW], C.SITE_OFFSET), xm[:, HAND_MJW]


target_pos, target_R = fingertip()   # HOLD the settled pose


def osc():
    """Full OSCController recipe on the bare arm: task wrench + nullspace posture
    + explicit joint damping + gravity feedforward. (The nullspace + KD_JOINT
    are what keep the redundant DOF from winding up.)"""
    ftp, ftR = fingertip()
    pt = wp.from_jax(ftp.reshape(-1).astype(jnp.float32)).view(wp.vec3)
    mjw.jac(mm, d, jp, jr, pt, body_arr)
    J = jnp.concatenate([wp.to_jax(jp).transpose(0, 2, 1), wp.to_jax(jr).transpose(0, 2, 1)], 1)[:, :, :7]
    qv = wp.to_jax(d.qvel)[:, :7]
    qpos = wp.to_jax(d.qpos)[:, :7]
    qM = wp.to_jax(d.qM)[:, :7, :7]
    bias = wp.to_jax(d.qfrc_bias)[:, :7]
    ftv = jnp.einsum("nij,nj->ni", J, qv)
    pe = target_pos - ftp
    re = C.mat_to_rotvec(target_R @ ftR.transpose(0, 2, 1))
    w = jnp.concatenate([C.KP_TASK[:3] * pe - C.KD_TASK[:3] * ftv[:, :3],
                         C.KP_TASK[3:] * re - C.KD_TASK[3:] * ftv[:, 3:]], -1)
    Mi = jnp.linalg.inv(qM)
    Lam_inv = J @ Mi @ J.transpose(0, 2, 1) + 1e-2 * jnp.eye(6)
    Lam = jnp.linalg.inv(Lam_inv)
    Jbar = Mi @ J.transpose(0, 2, 1) @ Lam
    null_proj = jnp.eye(7) - Jbar @ J
    tau = jnp.einsum("nij,nj->ni", J.transpose(0, 2, 1), jnp.einsum("nij,nj->ni", Lam, w))
    tau = tau + jnp.einsum("nij,nj->ni", null_proj,
                           jnp.einsum("nij,nj->ni", qM, C.KP_NULL * (NULL_Q - qpos) - C.KD_NULL * qv))
    tau = tau - C.KD_JOINT * qv + bias
    tau = jnp.clip(tau, -C.TORQUE_LIMIT, C.TORQUE_LIMIT)
    wp.copy(ctl.joint_f, wp.from_jax(jnp.zeros((1, nv)).at[:, :7].set(tau).reshape(-1).astype(jnp.float32)))
    return float(jnp.linalg.norm(pe))


viewer = newton.viewer.ViewerGL(paused=True)
viewer.set_model(model)
viewer.set_camera(pos=wp.vec3(1.6, -1.2, 1.0), pitch=-20.0, yaw=130.0)
print("[bare] OSC hold, PAUSED. Step in the GUI; watch hand_z (should stay ~constant).")

i = 0
t = 0.0
while viewer.is_running():
    if viewer.should_step():
        for _ in range(4):
            s0.clear_forces()
            err = osc()
            solver.step(s0, s1, ctl, None, 0.002)
            s0, s1 = s1, s0
        print(f"[step {i:4d}] hand_z={float(s0.body_q.numpy()[HAND][2]):.4f}  pos_err={err*1000:.2f}mm")
        i += 1
        t += 1 / 60
    viewer.begin_frame(t)
    viewer.log_state(s0)
    viewer.end_frame()
viewer.close()
