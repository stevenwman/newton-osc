"""Newton peg-insertion RL env (batched, gym-like) with pluggable controller.

The action space lives entirely in the controller (controllers.py), so the same
env compares JointPosition vs OSC on identical physics. Reward = ported phased
peg reward. reset seats the peg in the gripper + randomizes the hole pose.

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
import peg_scene_newton as scene
from controllers import JointPositionController

ASSET_HEIGHT = 0.025
HOLE_NOMINAL = np.array([0.6, 0.0, 0.05], dtype=np.float32)
HOLE_XY_NOISE = 0.02
HOLE_Z_NOISE = 0.01
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
        # Contact budgets are TOTAL across the batch in mujoco_warp -> scale with N
        # (floored at the single-env values so N=1 is unchanged). Measured: at rest
        # ~7 efc/world (weld+finger-coupling), under contact peak ~2 contacts &
        # ~9 efc/world with hover actions; a policy inserting against the 32 bore
        # tiles is higher, so budget ~16x the hover peak. (njmax drives a big
        # efc_J alloc — N*256 OOM'd a 4GiB array at N=512; N*128 is ample headroom.)
        nconmax = max(1024, N * 32)
        njmax = max(4096, N * 128)
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

        self.act_dim = self.controller.action_dim
        # obs (per world) = arm joint state (q,qd) + held-peg pose relative to the
        # goal hole (peg is welded to the EE -> EE-task state) + absolute goal hole
        # position + last action (jax_rl includes the action in the obs).
        self.obs_dim = 7 + 7 + 3 + 4 + 3 + self.act_dim
        self._last_action = jnp.zeros((N, self.act_dim), jnp.float32)
        self.steps = 0
        self.last_finite = True

        # vmapped reward / success over the N worlds (the reward math is per-world
        # scalar; vmap adds the leading batch axis). target_quat is unused (del'd in
        # compute_reward) so we bind a constant inside.
        self._reward_fn = jax.jit(jax.vmap(
            lambda hp, hq, tp, pz, htz: peg_reward.compute_reward(
                held_pos=hp, held_quat=hq, target_pos=tp,
                target_quat=jnp.array([1.0, 0.0, 0.0, 0.0]), peg_z=pz, hole_top_z=htz)))
        self._success_fn = jax.jit(jax.vmap(
            lambda pxy, pz, pq, hxy, htz: peg_reward.is_success(
                pxy, pz, pq, hxy, htz, ASSET_HEIGHT, 0.04)))

        self.controller.setup(self)

    # -- helpers ------------------------------------------------------------
    def _set_hole(self, pos):
        """pos: (N,3). Write each world's hole-joint parent transform, ONE notify."""
        pos = np.asarray(pos, dtype=np.float32).reshape(self.num_envs, 3)
        Xp = self.model.joint_X_p.numpy()
        idx = np.arange(self.num_envs) * self.njoint + HOLE_JOINT
        Xp[idx, :3] = pos
        self.model.joint_X_p.assign(Xp)
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        self.hole_pos = jnp.asarray(pos)

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

    def _obs(self):
        N = self.num_envs
        jq = wp.to_jax(self.state_0.joint_q).reshape(N, self.ncoord)
        jqd = wp.to_jax(self.state_0.joint_qd).reshape(N, self.ndof)
        peg = wp.to_jax(self.state_0.body_q).reshape(N, self.nbody, 7)[:, PEG_BODY_LOCAL]
        peg_rel = peg[:, :3] - self.hole_pos
        return jnp.concatenate([
            jq[:, :7], jqd[:, :7], peg_rel, peg[:, 3:7], self.hole_pos, self._last_action,
        ], axis=1).astype(jnp.float32)                     # (N, obs_dim)

    # -- gym API ------------------------------------------------------------
    def reset(self):
        N = self.num_envs
        q = self.model.joint_q.numpy().reshape(N, self.ncoord)
        q[:, :7] = self._arm_init
        q[:, 7:9] = self.finger
        self.model.joint_q.assign(q.reshape(-1))
        self._seat_peg()
        off = self.rng.uniform(
            [-HOLE_XY_NOISE, -HOLE_XY_NOISE, -HOLE_Z_NOISE],
            [HOLE_XY_NOISE, HOLE_XY_NOISE, HOLE_Z_NOISE], size=(N, 3)).astype(np.float32)
        self._set_hole(HOLE_NOMINAL + off)
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
        tq[:, :ARM_DOF] = self._arm_init
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
        return self._obs()

    def _substep(self):
        for _ in range(scene.SUBSTEPS):
            self.state_0.clear_forces()
            self.controller.apply(self)
            self.solver.step(self.state_0, self.state_1, self.control, None, scene.SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self, action):
        N = self.num_envs
        self._last_action = jnp.asarray(action, jnp.float32).reshape(N, self.act_dim)
        self.controller.set_action(self, action)
        self._substep()
        self.steps += 1
        peg = wp.to_jax(self.state_0.body_q).reshape(N, self.nbody, 7)[:, PEG_BODY_LOCAL]
        peg_pos, peg_xyzw = peg[:, :3], peg[:, 3:7]
        peg_wxyz = jnp.stack([peg_xyzw[:, 3], peg_xyzw[:, 0], peg_xyzw[:, 1], peg_xyzw[:, 2]], axis=-1)
        hole = self.hole_pos                                # (N,3)
        hole_top_z = hole[:, 2] + ASSET_HEIGHT              # (N,)
        reward = self._reward_fn(peg_pos, peg_wxyz, hole, peg_pos[:, 2], hole_top_z)   # (N,)
        success = self._success_fn(peg_pos[:, :2], peg_pos[:, 2], peg_wxyz, hole[:, :2], hole_top_z)
        obs = self._obs()
        self.last_finite = bool(jnp.isfinite(reward).all() & jnp.isfinite(obs).all())
        self.steps_done = self.steps >= self.episode_length
        done = self.steps_done
        info = {"truncation": float(done), "success": np.asarray(success)}
        return obs, reward, done, info
