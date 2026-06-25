"""Newton SQUARE-peg / square-socket RL env (variant of peg_env.py).

Same as PegEnv but: box peg + 4-wall square socket (peg_scene_square), and the
socket now has a YAW that matters (4-fold symmetry) — so reset randomizes hole yaw
(±30°), the reward gets yaw_period=pi/2 + the socket yaw, and success requires yaw
alignment (mod 90°). Obs/controller are unchanged (peg quat + ee_angvel already
expose yaw; OSC rz already controls it). See SQUARE_PEG_PLAN.md.

The action space lives entirely in the controller (controllers.py). Reward = the
phased peg reward with its optional yaw term enabled.

Batched: num_envs=N replicates the scene into N worlds (world-major joint/body
arrays). All episodes share one length and reset_on_done is SYNCHRONIZED (done =
TimeLimit only -> every world finishes together), so the whole batch resets at
once — no per-world masking needed in the normal path. obs/reward stay on-GPU
(wp.to_jax zero-copy + vmap reward); only the infrequent reset touches numpy.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # else JAX grabs ~75% of VRAM at init

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

import newton
import peg_reward
import peg_scene_square as scene
from controllers import JointPositionController

ASSET_HEIGHT = 0.025
HOLE_NOMINAL = np.array([0.6, 0.0, 0.05], dtype=np.float32)
HOLE_XY_NOISE = 0.02
HOLE_Z_NOISE = 0.01
HOLE_YAW_NOISE = 0.5236     # ±30° socket yaw randomization (rad)
YAW_PERIOD = np.pi / 2.0    # square = 4-fold symmetry (yaw aligns mod 90°)
ARM_RESET_NOISE = 0.02     # ±rad/joint perturbation of the arm start pose each reset
                           # (matches jax_rl). Broadens the reset state distribution so
                           # exploration reaches aligned∧low_z states the critic needs to
                           # learn descent — without it the deterministic start + 6-DOF
                           # rotation action-noise stays stuck in the hover trap.
HOLE_JOINT = 11            # local (per-world) joint index of the fixtured hole
PEG_BODY_LOCAL = 12        # local (per-world) body index of the peg
HAND_BODY_LOCAL = scene.HAND_IDX   # 8
ARM_DOF = 7
SETTLE_STEPS = 30          # gravity-comp settle so the controller captures its target at rest


def _quat_rotate_xyzw(q, v):
    """Rotate vec v (...,3) by quaternion q (...,4) in XYZW order. Batched, numpy."""
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


class PegEnv:
    def __init__(self, controller=None, episode_length=100, seed=0, weld=True, num_envs=1):
        self.num_envs = int(num_envs)
        N = self.num_envs
        self.controller = controller or JointPositionController()
        self.model = scene.build_model(
            arm_control_cb=lambda b: self.controller.configure(b, list(scene.ARM_Q), scene.FINGER),
            weld=weld, num_envs=N)
        # NOTE: cone="elliptic" + impratio=50 (from the original env's stiff-
        # insertion intent) destabilize Newton's SolverMuJoCo here — the free arm
        # gains energy and flies up. Default cone/impratio are stable. Revisit if
        # tight-clearance insertion needs the stiffer contact model.
        # Contact budget semantics in this mujoco_warp differ between the two knobs
        # (verified by inspecting mjw_data at N=512):
        #   nconmax is PER-WORLD  -> mjw sets naconmax = naccdmax = nconmax * nworld.
        #   njmax   is TOTAL across the batch (njmax_eff == passed value).
        # So nconmax must be a CONSTANT (do NOT scale by N, or it grows N^2 via the
        # internal *nworld and the convex-narrowphase EPA scratch explodes — N*32
        # blew an 18 GB epa_pr alloc at N=512). 256 contacts/world is ~128x the
        # measured peak (~2 contacts/world). njmax IS total -> scale with N; ~128
        # efc/world headroom over the ~9 measured. BUT the dense efc_J alloc is
        # njmax * nv_total, and nv_total ALSO scales with N -> efc_J is O(N^2):
        # N*128 gave an 8 GiB efc_J at N=512/1024 (fit in 32 GB) but a 32 GiB OOM at
        # N=2048. So keep the per-world njmax multiplier modest. N*32 (~32 efc/world,
        # ~3.5x the resting peak) -> ~8 GiB efc_J at N=2048. Bump if insertion
        # contacts overflow efc -> NaN.
        # Box peg vs flat slab walls makes MANY more contact constraints than the
        # round peg (~87 efc/world seen slamming in, vs ~9 for the cylinder). The
        # cylindrical env's N*32 (32 efc/world) overflows under box-box -> bump to
        # N*128 (128/world). njmax is TOTAL and efc_J ~ njmax*nv_total is O(N^2), but
        # the square task trains at N=128 (-> 16384, efc_J ~0.13 GB), so fine. Revisit
        # the multiplier if scaling N high.
        nconmax = 256
        njmax = max(8192, N * 128)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=True,
            nconmax=nconmax, njmax=njmax, iterations=100, ls_iterations=50)
        if weld:
            # Match jax_rl's deliberately-stiffened grasp weld (solref 0.001 / solimp
            # dimp->1 ~ rigid). Newton's add_equality default (solref 0.02) is too
            # soft -- the peg visibly flops on the gripper. mjw_model is single-world
            # with a LEADING nworld axis on the eq arrays: eq_solref is
            # (nworld, neq, 2), weld = eq index 1 (eq_type=[2,1]: 2=joint-couple,
            # 1=weld). `[..., 1, :]` spans nworld -> stiffens the weld in every world.
            M = self.solver.mjw_model
            sr = M.eq_solref.numpy().copy(); sr[..., 1, :] = (0.001, 1.0); M.eq_solref.assign(sr)
            si = M.eq_solimp.numpy().copy(); si[..., 1, :3] = (0.999, 0.9999, 0.001); M.eq_solimp.assign(si)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.device = wp.get_device()

        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self._arm_init = np.asarray(scene.ARM_Q, dtype=np.float32)
        self.finger = scene.FINGER

        # Per-world counts (world-major arrays divide cleanly by N — see probe).
        self.ndof = self.model.joint_dof_count // N        # per-world dof (OSC apply needs this)
        self.ncoord = self.model.joint_coord_count // N    # per-world coord
        self.nbody = self.model.body_count // N            # per-world body count
        self.njoint = self.model.joint_count // N          # per-world joint count
        self.lo = self.model.joint_limit_lower.numpy()[:ARM_DOF].copy()   # world-0 arm limits (same all worlds)
        self.hi = self.model.joint_limit_upper.numpy()[:ARM_DOF].copy()
        self.hole_pos = jnp.broadcast_to(jnp.asarray(HOLE_NOMINAL), (N, 3))   # (N,3) on GPU
        self.hole_yaw = jnp.zeros(N)                                          # (N,) socket yaw

        self.act_dim = self.controller.action_dim
        # obs (per world) = arm joint state (q,qd) + held-peg pose relative to the
        # goal hole (peg is welded to the EE -> EE-task state) + absolute goal hole
        # position + last action (jax_rl includes the action in the obs).
        # + ee_linvel(3) + ee_angvel(3) (task-space EE velocity, finite-diff — the
        # signal the policy needs to feel it's stalling at the bore lip) + prev_action
        # (action history; jax_rl feeds actions AND prev_actions).
        # + 2 for the GOAL yaw [cos(hole_yaw), sin(hole_yaw)] — the square peg must
        # match the socket's (randomized ±30°) yaw, so the policy MUST observe it.
        # (sin/cos avoids the wrap discontinuity of raw yaw.)
        self.obs_dim = 7 + 7 + 3 + 4 + 3 + self.act_dim + 6 + self.act_dim + 2
        self._last_action = jnp.zeros((N, self.act_dim), jnp.float32)
        self._prev_action = jnp.zeros((N, self.act_dim), jnp.float32)
        self._prev_ft_pos = None        # for finite-diff EE velocity
        self._prev_ft_R = None
        self._ctrl_dt = scene.SUBSTEPS * scene.SIM_DT
        self.steps = 0
        self.last_finite = True

        # vmapped reward / success over the N worlds (the reward math is per-world
        # scalar; vmap adds the leading batch axis). target_quat is unused (del'd in
        # compute_reward) so we bind a constant inside.
        # Square: enable the reward's yaw term (yaw_period=pi/2) + pass the socket
        # yaw per world. Success requires yaw aligned (mod 90°) on top of xy/tilt/depth.
        self._yaw_tol = 0.0873     # ~5° success yaw tolerance
        self._reward_fn = jax.jit(jax.vmap(
            lambda hp, hq, tp, pz, htz, hy: peg_reward.compute_reward(
                held_pos=hp, held_quat=hq, target_pos=tp,
                target_quat=jnp.array([1.0, 0.0, 0.0, 0.0]), peg_z=pz, hole_top_z=htz,
                yaw_period=YAW_PERIOD, hole_yaw=hy, yaw_tol=self._yaw_tol)))
        self._success_fn = jax.jit(jax.vmap(
            lambda pxy, pz, pq, hxy, htz, hy: peg_reward.is_success(
                pxy, pz, pq, hxy, htz, ASSET_HEIGHT, 0.04)
            & (jnp.abs(peg_reward.yaw_error(pq, hy, YAW_PERIOD)) < self._yaw_tol)))

        self.controller.setup(self)

    # -- helpers ------------------------------------------------------------
    def _set_hole(self, pos, yaw):
        """pos:(N,3), yaw:(N,). Write each world's hole-joint parent transform (pos
        + yaw rotation about z), ONE notify. joint_X_p quat is XYZW."""
        N = self.num_envs
        pos = np.asarray(pos, dtype=np.float32).reshape(N, 3)
        yaw = np.asarray(yaw, dtype=np.float32).reshape(N)
        Xp = self.model.joint_X_p.numpy()
        idx = np.arange(N) * self.njoint + HOLE_JOINT
        Xp[idx, :3] = pos
        Xp[idx, 3:7] = np.stack([np.zeros(N), np.zeros(N),
                                 np.sin(yaw / 2), np.cos(yaw / 2)], axis=1)  # XYZW z-rot
        self.model.joint_X_p.assign(Xp)
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        self.hole_pos = jnp.asarray(pos)
        self.hole_yaw = jnp.asarray(yaw)

    def _seat_peg(self):
        """Place each world's peg at hand * weld_offset so the stiff weld doesn't
        snap it across the room on the first step (mirrors the env's reset)."""
        N = self.num_envs
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        bq = self.state_0.body_q.numpy().reshape(N, self.nbody, 7)
        hand = bq[:, HAND_BODY_LOCAL]                       # (N,7) px,py,pz, qx,qy,qz,qw
        hand_pos, hand_q = hand[:, :3], hand[:, 3:7]
        off = np.broadcast_to(np.asarray(scene.WELD_OFFSET, np.float32), (N, 3))
        peg_pos = hand_pos + _quat_rotate_xyzw(hand_q, off)   # weld offset has identity rot
        q = self.model.joint_q.numpy().reshape(N, self.ncoord)
        q[:, 9:12] = peg_pos
        q[:, 12:16] = hand_q                               # warp quat = xyzw
        self.model.joint_q.assign(q.reshape(-1))

    def _ee_vel(self):
        """Task-space EE velocity by finite-diff of the fingertip pose between control
        steps. Angular vel from the skew part of R_curr·R_prevᵀ (no quat, no near-pi
        singularity — matches the controller's matrix-only convention). First call
        after reset returns zeros (no prev). Side-effect: advances the prev pose."""
        d = self.solver.mjw_data
        ft_pos, ft_R = self.controller._fingertip(d)        # (N,3), (N,3,3)
        if self._prev_ft_pos is None:
            lin = jnp.zeros((self.num_envs, 3)); ang = jnp.zeros((self.num_envs, 3))
        else:
            lin = (ft_pos - self._prev_ft_pos) / self._ctrl_dt
            dR = jnp.einsum('nij,nkj->nik', ft_R, self._prev_ft_R)   # R_curr @ R_prevᵀ
            skew = 0.5 * (dR - jnp.transpose(dR, (0, 2, 1)))
            ang = jnp.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], axis=-1) / self._ctrl_dt
        self._prev_ft_pos, self._prev_ft_R = ft_pos, ft_R
        return lin, ang

    def _obs(self):
        N = self.num_envs
        jq = wp.to_jax(self.state_0.joint_q).reshape(N, self.ncoord)
        jqd = wp.to_jax(self.state_0.joint_qd).reshape(N, self.ndof)
        peg = wp.to_jax(self.state_0.body_q).reshape(N, self.nbody, 7)[:, PEG_BODY_LOCAL]
        peg_rel = peg[:, :3] - self.hole_pos
        ee_lin, ee_ang = self._ee_vel()
        goal_yaw = jnp.stack([jnp.cos(self.hole_yaw), jnp.sin(self.hole_yaw)], axis=-1)  # (N,2)
        return jnp.concatenate([
            jq[:, :7], jqd[:, :7], peg_rel, peg[:, 3:7], self.hole_pos,
            self._last_action, ee_lin, ee_ang, self._prev_action, goal_yaw,
        ], axis=1).astype(jnp.float32)                     # (N, obs_dim)

    # -- gym API ------------------------------------------------------------
    def reset(self):
        N = self.num_envs
        # Per-reset arm-pose noise (±ARM_RESET_NOISE rad/joint). Use the SAME noisy
        # config for the start qpos AND the settle servo target below — otherwise the
        # settle (which servos to the target) would pull the arm back to _arm_init and
        # erase the perturbation.
        arm_start = (self._arm_init
                     + self.rng.uniform(-ARM_RESET_NOISE, ARM_RESET_NOISE,
                                        size=(N, ARM_DOF)).astype(np.float32))
        q = self.model.joint_q.numpy().reshape(N, self.ncoord)
        q[:, :7] = arm_start
        q[:, 7:9] = self.finger
        self.model.joint_q.assign(q.reshape(-1))
        self._seat_peg()
        off = self.rng.uniform(
            [-HOLE_XY_NOISE, -HOLE_XY_NOISE, -HOLE_Z_NOISE],
            [HOLE_XY_NOISE, HOLE_XY_NOISE, HOLE_Z_NOISE], size=(N, 3)).astype(np.float32)
        yaw = self.rng.uniform(-HOLE_YAW_NOISE, HOLE_YAW_NOISE, size=N).astype(np.float32)
        self._set_hole(HOLE_NOMINAL + off, yaw)
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        # Settle to populate mjw_data (qM/bias/xpos) while HOLDING the start pose,
        # THEN init the controller so its target matches the current fingertip at
        # REST. A free-fall settle (zero force) lets the arm sag during these steps;
        # an OSC controller then captures a *moving* target -> startup lurch (this
        # was the lesson from bare_arm_view.py). We hold the pose two ways so it's
        # correct for either controller: (a) servo targets for any POSITION-mode
        # dofs (JointPos arm; OSC fingers), and (b) gravity comp fed into the arm
        # dofs for an OSC NONE-mode arm. Free joints take no joint_f, so the welded
        # peg just hangs at its seated pose.
        tq = self.control.joint_target_pos.numpy().reshape(N, self.ndof)
        tq[:, :ARM_DOF] = arm_start
        tq[:, ARM_DOF:ARM_DOF + 2] = self.finger
        self.control.joint_target_pos.assign(tq.reshape(-1))
        self.control.joint_f.zero_()
        for _ in range(SETTLE_STEPS):
            self.state_0.clear_forces()
            bias = self.solver.mjw_data.qfrc_bias.numpy().reshape(N, self.ndof)
            jf = self.control.joint_f.numpy().reshape(N, self.ndof)
            jf[:, :ARM_DOF] = bias[:, :ARM_DOF]
            self.control.joint_f.assign(jf.reshape(-1))
            self.solver.step(self.state_0, self.state_1, self.control, None, scene.SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.control.joint_f.zero_()          # hand a clean force buffer to the controller
        self.controller.reset(self)
        self.steps = 0
        self._last_action = jnp.zeros((N, self.act_dim), jnp.float32)
        self._prev_action = jnp.zeros((N, self.act_dim), jnp.float32)
        self._prev_ft_pos = None              # first _obs velocity = 0
        self._prev_ft_R = None
        return self._obs()

    def _substep(self):
        for _ in range(scene.SUBSTEPS):
            self.state_0.clear_forces()
            self.controller.apply(self)
            self.solver.step(self.state_0, self.state_1, self.control, None, scene.SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self, action):
        N = self.num_envs
        self._prev_action = self._last_action          # shift: prev_action = last step's action
        self._last_action = jnp.asarray(action, jnp.float32).reshape(N, self.act_dim)
        self.controller.set_action(self, action)
        self._substep()
        self.steps += 1
        peg = wp.to_jax(self.state_0.body_q).reshape(N, self.nbody, 7)[:, PEG_BODY_LOCAL]
        peg_pos, peg_xyzw = peg[:, :3], peg[:, 3:7]
        peg_wxyz = jnp.stack([peg_xyzw[:, 3], peg_xyzw[:, 0], peg_xyzw[:, 1], peg_xyzw[:, 2]], axis=-1)
        hole = self.hole_pos                                # (N,3)
        hole_top_z = hole[:, 2] + ASSET_HEIGHT              # (N,)
        reward = self._reward_fn(peg_pos, peg_wxyz, hole, peg_pos[:, 2], hole_top_z, self.hole_yaw)
        success = self._success_fn(peg_pos[:, :2], peg_pos[:, 2], peg_wxyz, hole[:, :2],
                                   hole_top_z, self.hole_yaw)
        obs = self._obs()
        self.last_finite = bool(jnp.isfinite(reward).all() & jnp.isfinite(obs).all())
        self.steps_done = self.steps >= self.episode_length
        done = self.steps_done
        info = {"truncation": float(done), "success": np.asarray(success)}
        return obs, reward, done, info
