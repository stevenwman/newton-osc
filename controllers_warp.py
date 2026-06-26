"""WarpOSCController — the OSC math rewritten as a Warp kernel (vs controllers.py's
JAX _osc_core), so the per-substep hot path is pure Warp (no jax, no dlpack). Goal:
kill the N=1 launch-latency the JAX OSC carries; verify parity + measure speedup.

Approach: one thread per world; per-world dense LA via in-kernel Cholesky-solve (no
explicit inverse). qM (7x7 SPD) and Lam_inv (6x6 SPD) are Cholesky-factored; we
solve instead of invert. set_action stays JAX (1x/control-step) but writes its
outputs (target_pos/R, kp_scale, zeta) to Warp arrays the kernel reads each substep.

Parity target vs controllers.OSCController: tau matches to ~1e-4.
"""
import numpy as np
import jax, jax.numpy as jnp
import warp as wp
import mujoco_warp as mjw
import newton

# reuse the jax-side config + helpers (set_action math, gains) from the jax controller
from controllers import (ARM_DOF, KP_TASK, KD_NULL, KP_NULL, KD_JOINT, TORQUE_LIMIT,
                         NULLSPACE_Q, POS_BOUNDS, POS_THRESHOLD, ROT_THRESHOLD, ASSET_HEIGHT,
                         GAIN_LOG2, GAIN_DIMS, SITE_OFFSET, HAND_MJW_ID, BASE_MJW_ID,
                         rotvec_to_mat, OSCController)

vec6 = wp.types.vector(length=6, dtype=wp.float32)
vec7 = wp.types.vector(length=7, dtype=wp.float32)
mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat77 = wp.types.matrix(shape=(7, 7), dtype=wp.float32)
mat67 = wp.types.matrix(shape=(6, 7), dtype=wp.float32)
mat76 = wp.types.matrix(shape=(7, 6), dtype=wp.float32)

_KP = wp.constant(vec6(*[float(x) for x in KP_TASK]))
_NQ = wp.constant(vec7(*[float(x) for x in NULLSPACE_Q]))
_HAND = wp.constant(int(HAND_MJW_ID))
_SITE = wp.constant(wp.vec3(float(SITE_OFFSET[0]), float(SITE_OFFSET[1]), float(SITE_OFFSET[2])))


@wp.kernel
def fingertip_kernel(xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                     ft_pos: wp.array(dtype=wp.vec3), ft_R: wp.array(dtype=wp.mat33)):
    w = wp.tid()
    hm = xmat[w, _HAND]
    ft_pos[w] = xpos[w, _HAND] + hm * _SITE      # hand pos + R_hand @ site offset
    ft_R[w] = hm                                 # site orientation == hand orientation


@wp.func
def chol7(A: mat77) -> mat77:
    L = mat77()
    for j in range(7):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        L[j, j] = wp.sqrt(s)
        for i in range(j + 1, 7):
            s2 = A[i, j]
            for k in range(j):
                s2 -= L[i, k] * L[j, k]
            L[i, j] = s2 / L[j, j]
    return L


@wp.func
def solve7(L: mat77, b: vec7) -> vec7:
    y = vec7()
    for i in range(7):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
    x = vec7()
    for ii in range(7):
        i = 6 - ii
        s = y[i]
        for k in range(i + 1, 7):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


@wp.func
def chol6(A: mat66) -> mat66:
    L = mat66()
    for j in range(6):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        L[j, j] = wp.sqrt(s)
        for i in range(j + 1, 6):
            s2 = A[i, j]
            for k in range(j):
                s2 -= L[i, k] * L[j, k]
            L[i, j] = s2 / L[j, j]
    return L


@wp.func
def solve6(L: mat66, b: vec6) -> vec6:
    y = vec6()
    for i in range(6):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
    x = vec6()
    for ii in range(6):
        i = 5 - ii
        s = y[i]
        for k in range(i + 1, 6):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


@wp.kernel
def osc_kernel(jacp: wp.array3d(dtype=wp.float32), jacr: wp.array3d(dtype=wp.float32),
               qM: wp.array3d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
               qpos: wp.array2d(dtype=wp.float32), bias: wp.array2d(dtype=wp.float32),
               target_pos: wp.array(dtype=wp.vec3), ft_pos: wp.array(dtype=wp.vec3),
               ft_R: wp.array(dtype=wp.mat33), target_R: wp.array(dtype=wp.mat33),
               kp_scale: wp.array2d(dtype=wp.float32),
               zeta: wp.array2d(dtype=wp.float32), ridge: wp.float32,
               joint_f: wp.array2d(dtype=wp.float32)):
    w = wp.tid()
    # J_arm 6x7 (rows 0-2 linear from jacp, 3-5 angular from jacr); qM7, qvel7, qpos7, bias7
    J = mat67()
    for k in range(7):
        J[0, k] = jacp[w, 0, k]; J[1, k] = jacp[w, 1, k]; J[2, k] = jacp[w, 2, k]
        J[3, k] = jacr[w, 0, k]; J[4, k] = jacr[w, 1, k]; J[5, k] = jacr[w, 2, k]
    M = mat77()
    qv = vec7(); qp = vec7(); bq = vec7()
    for i in range(7):
        qv[i] = qvel[w, i]; qp[i] = qpos[w, i]; bq[i] = bias[w, i]
        for j in range(7):
            M[i, j] = qM[w, i, j]
    # ft_vel = J @ qvel
    fv = vec6()
    for r in range(6):
        s = float(0.0)
        for k in range(7):
            s += J[r, k] * qv[k]
        fv[r] = s
    # errors -> wrench. rot_err = robosuite cross-product form (in-kernel from ft_R,target_R)
    pe = target_pos[w] - ft_pos[w]
    fr = ft_R[w]; tg = target_R[w]
    re = 0.5 * (wp.cross(wp.vec3(fr[0, 0], fr[1, 0], fr[2, 0]), wp.vec3(tg[0, 0], tg[1, 0], tg[2, 0]))
                + wp.cross(wp.vec3(fr[0, 1], fr[1, 1], fr[2, 1]), wp.vec3(tg[0, 1], tg[1, 1], tg[2, 1]))
                + wp.cross(wp.vec3(fr[0, 2], fr[1, 2], fr[2, 2]), wp.vec3(tg[0, 2], tg[1, 2], tg[2, 2])))
    err = vec6(pe[0], pe[1], pe[2], re[0], re[1], re[2])
    wr = vec6()
    for i in range(6):
        kp = _KP[i] * kp_scale[w, i]
        kd = 2.0 * zeta[w, i] * wp.sqrt(kp)
        wr[i] = kp * err[i] - kd * fv[i]
    # Cholesky(qM); MiJt (7x6) = solve(qM, J^T col by col)
    Lm = chol7(M)
    MiJt = mat76()
    for c in range(6):
        rhs = vec7()
        for k in range(7):
            rhs[k] = J[c, k]                       # J^T[:,c] = J[c,:]
        col = solve7(Lm, rhs)
        for k in range(7):
            MiJt[k, c] = col[k]
    # Lam_inv = J @ MiJt + ridge I  (6x6)
    Li = mat66()
    for i in range(6):
        for j in range(6):
            s = float(0.0)
            for k in range(7):
                s += J[i, k] * MiJt[k, j]
            Li[i, j] = s + wp.where(i == j, ridge, 0.0)
    Ll = chol6(Li)
    # tau_task = J^T @ solve(Lam_inv, wrench)
    xw = solve6(Ll, wr)
    tau = vec7()
    for i in range(7):
        s = float(0.0)
        for c in range(6):
            s += J[c, i] * xw[c]
        tau[i] = s
    # nullspace: v = qM @ (kp_null(q_def-q) - kd_null qvel); tau_null = v - MiJt @ solve(Lam_inv, J v)
    qerr = vec7()
    for i in range(7):
        qerr[i] = KP_NULL * (_NQ[i] - qp[i]) - KD_NULL * qv[i]
    v = vec7()
    for i in range(7):
        s = float(0.0)
        for j in range(7):
            s += M[i, j] * qerr[j]
        v[i] = s
    Jv = vec6()
    for r in range(6):
        s = float(0.0)
        for k in range(7):
            s += J[r, k] * v[k]
        Jv[r] = s
    xn = solve6(Ll, Jv)
    for i in range(7):
        s = float(0.0)
        for c in range(6):
            s += MiJt[i, c] * xn[c]
        tau[i] += v[i] - s
    # joint-space damping + bias, clip
    for i in range(7):
        ti = tau[i] - KD_JOINT * qv[i] + bq[i]
        joint_f[w, i] = wp.clamp(ti, -TORQUE_LIMIT, TORQUE_LIMIT)


def _ori_error_np(R_cur, R_des):
    """robosuite cross-product orientation error (numpy, batched (nw,3,3))."""
    return 0.5 * (np.cross(R_cur[:, :, 0], R_des[:, :, 0])
                  + np.cross(R_cur[:, :, 1], R_des[:, :, 1])
                  + np.cross(R_cur[:, :, 2], R_des[:, :, 2]))


class WarpOSCController(OSCController):
    """OSC with the per-substep math in a Warp kernel. Inherits set_action/reset/
    configure/_fingertip from the JAX controller; overrides setup (alloc wp buffers)
    and apply (warp kernel)."""

    def setup(self, env):
        super().setup(env)
        nw = self.nw; dev = env.device
        self._target_pos_wp = wp.zeros(nw, dtype=wp.vec3, device=dev)
        self._target_R_wp = wp.zeros(nw, dtype=wp.mat33, device=dev)
        self._ft_pos_wp = wp.zeros(nw, dtype=wp.vec3, device=dev)
        self._ft_R_wp = wp.zeros(nw, dtype=wp.mat33, device=dev)
        self._kp_wp = wp.zeros((nw, 6), dtype=wp.float32, device=dev)
        self._zeta_wp = wp.zeros((nw, 6), dtype=wp.float32, device=dev)
        self._jf = wp.zeros((nw, env.ndof), dtype=wp.float32, device=dev)

    def _push_targets(self):
        """Convert the jax-side targets/gains to wp arrays ONCE per control step."""
        wp.copy(self._target_pos_wp, wp.from_jax(jnp.asarray(self.target_pos, jnp.float32), dtype=wp.vec3))
        wp.copy(self._target_R_wp, wp.from_jax(jnp.asarray(self.target_R, jnp.float32), dtype=wp.mat33))
        wp.copy(self._kp_wp, wp.from_jax(jnp.asarray(self.kp_scale, jnp.float32)))
        wp.copy(self._zeta_wp, wp.from_jax(jnp.asarray(self.zeta, jnp.float32)))

    def set_action(self, env, action):
        super().set_action(env, action)                 # jax: EMA + target_pos/R + gains
        self._push_targets()                             # 1x/control-step, not per-substep

    def apply(self, env):
        d = env.solver.mjw_data
        mjw.forward(env.solver.mjw_model, d)
        # fingertip pose (warp), Jacobian (warp), OSC (warp) — no jax in the substep path
        wp.launch(fingertip_kernel, dim=self.nw, device=env.device,
                  inputs=[d.xpos, d.xmat], outputs=[self._ft_pos_wp, self._ft_R_wp])
        mjw.jac(env.solver.mjw_model, d, self._jacp, self._jacr, self._ft_pos_wp, self._body)
        wp.launch(osc_kernel, dim=self.nw, device=env.device,
                  inputs=[self._jacp, self._jacr, d.qM, d.qvel, d.qpos, d.qfrc_bias,
                          self._target_pos_wp, self._ft_pos_wp, self._ft_R_wp, self._target_R_wp,
                          self._kp_wp, self._zeta_wp, wp.float32(self.ridge)],
                  outputs=[self._jf])
        wp.copy(env.control.joint_f, self._jf.reshape(-1))
