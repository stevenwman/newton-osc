"""Action-space controllers for the Newton peg env (the comparison surface).

Two controllers behind one interface so the same step() drives either action
space on identical physics:

  - JointPositionController: 7-D joint-position deltas -> position servos.
  - OSCController: 6-D operational-space pose deltas -> Khatib OSC -> joint
    torques. Faithful port of the jax_rl OSC, computed on-GPU from Newton's
    mujoco_warp data (mjw_data), batched over worlds, applied via control.joint_f.
    NO CPU sync — J/M/bias come from the same GPU model being stepped, exactly
    like jax_rl's mjx.jac/full_m.

Both are batch-ready: math operates on a leading (nworld,...) axis. Tested at
nworld=1; replicate() scales it.
"""

import jax
import jax.numpy as jnp
import mujoco_warp as mjw
import numpy as np
import warp as wp

import newton

ARM_DOF = 7

# --- OSC config (from jax_rl factory_peg_insert) -----------------------------
EMA_FACTOR = 0.2
POS_THRESHOLD = jnp.array([0.02, 0.02, 0.02])
ROT_THRESHOLD = jnp.array([0.097, 0.097, 0.097])
POS_BOUNDS = jnp.array([0.02, 0.02, 0.10])           # cube around bore top
KP_TASK = jnp.array([100.0, 100.0, 100.0, 30.0, 30.0, 30.0])
KD_TASK = jnp.array([20.0, 20.0, 20.0, 10.954, 10.954, 10.954])
KP_NULL, KD_NULL = 10.0, 6.3246
KD_JOINT = 8.0          # explicit joint-space damping (stabilizes the OSC)
TORQUE_LIMIT = 100.0
# Nullspace posture target = the env's IK start pose (well-conditioned), so the
# nullspace term holds the home config instead of yanking toward a worse one.
NULLSPACE_Q = jnp.array([-0.003, 0.476, 0.003, -2.032, -0.002, 2.508, 0.787])
ASSET_HEIGHT = 0.025
HAND_MJW_ID = 9                                       # hand body id in mjw/mujoco (world=0)
HAND_NEWTON_ID = 8                                    # hand body id in Newton state (world-excl)
SITE_OFFSET = jnp.array([0.0, 0.0, 0.1034])          # fingertip_centered in hand frame


# --- batched quaternion / rotation helpers (wxyz) ----------------------------
def quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = jnp.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    return m.reshape(q.shape[:-1] + (3, 3))


def mat_to_rotvec(R):
    """Axis-angle (rotation vector) of a batched rotation matrix."""
    cos = jnp.clip((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    angle = jnp.arccos(cos)
    axis = jnp.stack([R[..., 2, 1] - R[..., 1, 2],
                      R[..., 0, 2] - R[..., 2, 0],
                      R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    denom = jnp.where(jnp.abs(jnp.sin(angle)) > 1e-6, 2.0 * jnp.sin(angle), 1.0)
    axis = axis / denom[..., None]
    return axis * angle[..., None]


def rotvec_to_mat(v):
    """Batched axis-angle -> rotation matrix (Rodrigues). Identity at |v|->0."""
    a = jnp.linalg.norm(v, axis=-1, keepdims=True)            # (...,1)
    safe = jnp.where(a > 1e-8, a, 1.0)
    k = v / safe
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    c = jnp.cos(a)[..., 0]
    s = jnp.sin(a)[..., 0]
    C = 1.0 - c
    R = jnp.stack([
        c + kx * kx * C, kx * ky * C - kz * s, kx * kz * C + ky * s,
        ky * kx * C + kz * s, c + ky * ky * C, ky * kz * C - kx * s,
        kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C,
    ], axis=-1).reshape(v.shape[:-1] + (3, 3))
    return R


def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def mat_to_quat(R):
    """Batched rotation matrix -> quat (wxyz), trace method with where-branches."""
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = jnp.sqrt(jnp.clip(1.0 + t, 1e-12, None)) * 0.5
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * w)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * w)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * w)
    return jnp.stack([w, x, y, z], axis=-1)


# ============================================================================
class JointPositionController:
    """7-D joint-position-delta action -> position servos on the arm."""
    action_dim = ARM_DOF

    def __init__(self, scale=0.05, kp=400.0, kd=40.0):
        self.scale, self.kp, self.kd = scale, kp, kd

    def configure(self, builder, arm_init, finger):
        for i in range(ARM_DOF + 2):              # arm + 2 fingers
            builder.joint_target_mode[i] = newton.JointTargetMode.POSITION
            builder.joint_target_ke[i] = self.kp
            builder.joint_target_kd[i] = self.kd

    def setup(self, env):
        pass

    def reset(self, env):
        self._target = env._arm_init.copy()
        tq = env.control.joint_target_pos.numpy()
        tq[:ARM_DOF] = env._arm_init
        tq[ARM_DOF:ARM_DOF + 2] = env.finger
        env.control.joint_target_pos.assign(tq)

    def set_action(self, env, action):
        a = np.clip(np.asarray(action, np.float32).reshape(-1)[:ARM_DOF], -1, 1)
        self._target = np.clip(self._target + a * self.scale, env.lo, env.hi)
        tq = env.control.joint_target_pos.numpy()
        tq[:ARM_DOF] = self._target
        env.control.joint_target_pos.assign(tq)

    def apply(self, env):
        pass                                       # servo holds target across substeps


# ============================================================================
class OSCController:
    """6-D operational-space pose-delta action -> Khatib OSC -> joint torques."""
    action_dim = 6
    use_task = True
    use_rot = True          # include orientation in the task wrench
    use_null = True
    use_bias = True
    last_pos_err = 0.0      # diagnostics (norms), updated each apply()
    last_rot_err = 0.0
    last_tau_task = 0.0
    last_tau_null = 0.0
    last_bias = 0.0
    last_tau = 0.0
    last_cond = 0.0         # condition number of Lambda_inv (singularity check)
    last_qvel = 0.0

    def configure(self, builder, arm_init, finger):
        # Arm left in NONE mode (no servo); OSC drives it via control.joint_f.
        # Fingers held by a light servo so the grasp pose is stable.
        for i in (ARM_DOF, ARM_DOF + 1):
            builder.joint_target_mode[i] = newton.JointTargetMode.POSITION
            builder.joint_target_ke[i] = 100.0
            builder.joint_target_kd[i] = 10.0

    def setup(self, env):
        self.nw = env.solver.mjw_data.nworld
        self.nv = env.solver.mjw_data.qvel.shape[1]
        self._jacp = wp.zeros((self.nw, self.nv, 3), dtype=wp.float32, device=env.device)
        self._jacr = wp.zeros((self.nw, self.nv, 3), dtype=wp.float32, device=env.device)
        self._body = wp.array([HAND_MJW_ID] * self.nw, dtype=wp.int32, device=env.device)
        self.arm = jnp.arange(ARM_DOF)

    def _fingertip(self, d):
        """Returns (fingertip_pos, hand_rotation_matrix). Site orientation is
        identity, so the fingertip orientation == the hand's. We keep it as a
        matrix (read directly from mjw) to avoid the mat<->quat singularity at
        near-pi gripper-down orientations."""
        xpos = wp.to_jax(d.xpos).reshape(self.nw, -1, 3)
        xmat = wp.to_jax(d.xmat).reshape(self.nw, -1, 3, 3)
        hp = xpos[:, HAND_MJW_ID]                       # (nw,3)
        hm = xmat[:, HAND_MJW_ID]                       # (nw,3,3)
        ft_pos = hp + jnp.einsum('nij,j->ni', hm, SITE_OFFSET)
        return ft_pos, hm

    def reset(self, env):
        # Called after the env's settle step, so mjw_data is valid. Target =
        # current fingertip pose (zero startup error -> stable hold). Orientation
        # target stored as a matrix.
        ft_pos, ft_R = self._fingertip(env.solver.mjw_data)
        self.target_pos = ft_pos
        self.target_R = ft_R
        self.ema = jnp.zeros((self.nw, 6))

    def set_action(self, env, action):
        a = jnp.asarray(action, jnp.float32).reshape(self.nw, 6)
        self.ema = EMA_FACTOR * a + (1.0 - EMA_FACTOR) * self.ema
        pos_delta = self.ema[:, :3] * POS_THRESHOLD
        rot_delta = self.ema[:, 3:] * ROT_THRESHOLD
        ft_pos, _ = self._fingertip(env.solver.mjw_data)
        anchor = jnp.asarray(env.hole_pos) + jnp.array([0.0, 0.0, ASSET_HEIGHT])
        tgt = jnp.clip(ft_pos + pos_delta, anchor - POS_BOUNDS, anchor + POS_BOUNDS)
        self.target_pos = tgt
        # accumulate orientation target via a rotation-matrix delta (robust)
        self.target_R = self.target_R @ rotvec_to_mat(rot_delta)

    def apply(self, env):
        d = env.solver.mjw_data
        ft_pos, ft_R = self._fingertip(d)
        # Fingertip Jacobian at the world point (on-GPU, batched).
        pt = wp.from_jax(ft_pos.reshape(-1).astype(jnp.float32)).view(wp.vec3)
        mjw.jac(env.solver.mjw_model, d, self._jacp, self._jacr, pt, self._body)
        jacp = wp.to_jax(self._jacp)                    # (nw,nv,3)
        jacr = wp.to_jax(self._jacr)
        J = jnp.concatenate([jacp.transpose(0, 2, 1), jacr.transpose(0, 2, 1)], axis=1)  # (nw,6,nv)
        J_arm = J[:, :, :ARM_DOF]                       # (nw,6,7)

        qvel = wp.to_jax(d.qvel)[:, :ARM_DOF]           # (nw,7)
        qpos = wp.to_jax(d.qpos)[:, :ARM_DOF]           # (nw,7)
        qM = wp.to_jax(d.qM)[:, :ARM_DOF, :ARM_DOF]     # (nw,7,7)
        bias = wp.to_jax(d.qfrc_bias)[:, :ARM_DOF]      # (nw,7) gravity+coriolis

        ft_vel = jnp.einsum('nij,nj->ni', J_arm, qvel)  # (nw,6)
        pos_err = self.target_pos - ft_pos
        R_err = self.target_R @ ft_R.transpose(0, 2, 1)   # matrix-to-matrix (robust)
        rot_err = mat_to_rotvec(R_err)
        if not self.use_rot:
            rot_err = jnp.zeros_like(rot_err)
        self.last_pos_err = float(jnp.linalg.norm(pos_err))
        self.last_rot_err = float(jnp.linalg.norm(rot_err))
        rot_wrench = (KP_TASK[3:] * rot_err - KD_TASK[3:] * ft_vel[:, 3:]) if self.use_rot \
            else jnp.zeros_like(rot_err)
        wrench = jnp.concatenate([
            KP_TASK[:3] * pos_err - KD_TASK[:3] * ft_vel[:, :3],
            rot_wrench,
        ], axis=-1)                                     # (nw,6)

        M_inv = jnp.linalg.inv(qM)                       # (nw,7,7)
        # Damped-least-squares ridge: regularizes the op-space inertia inversion
        # near kinematic singularities (caps Lambda's max eigenvalue ~1/ridge).
        # 1e-2 keeps tau bounded where the arm is poorly conditioned.
        Lam_inv = J_arm @ M_inv @ J_arm.transpose(0, 2, 1) + 1e-2 * jnp.eye(6)
        Lam = jnp.linalg.inv(Lam_inv)                    # (nw,6,6)
        Jbar = M_inv @ J_arm.transpose(0, 2, 1) @ Lam    # (nw,7,6)
        null_proj = jnp.eye(ARM_DOF) - Jbar @ J_arm      # (nw,7,7)
        tau_null = jnp.einsum('nij,nj->ni', null_proj,
                              jnp.einsum('nij,nj->ni', qM,
                                         KP_NULL * (NULLSPACE_Q - qpos) - KD_NULL * qvel))
        tau_task = jnp.einsum('nij,nj->ni', J_arm.transpose(0, 2, 1),
                              jnp.einsum('nij,nj->ni', Lam, wrench))
        tau = tau_task if self.use_task else jnp.zeros_like(tau_task)
        if self.use_null:
            tau = tau + tau_null
        tau = tau - KD_JOINT * qvel                       # explicit joint-space damping
        if self.use_bias:
            tau = tau + bias
        tau = jnp.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)  # (nw,7)

        self.last_tau_task = float(jnp.linalg.norm(tau_task))
        self.last_tau_null = float(jnp.linalg.norm(tau_null))
        self.last_bias = float(jnp.linalg.norm(bias))
        self.last_tau = float(jnp.linalg.norm(tau))
        self.last_cond = float(jnp.linalg.cond(Lam_inv[0]))
        self.last_qvel = float(jnp.linalg.norm(qvel))

        jf = jnp.zeros((self.nw, env.ndof))
        jf = jf.at[:, :ARM_DOF].set(tau).reshape(-1).astype(jnp.float32)
        wp.copy(env.control.joint_f, wp.from_jax(jf))
