"""Newton peg-insertion RL env (single-env, gym-like) with pluggable controller.

The action space lives entirely in the controller (controllers.py), so the same
env compares JointPosition vs OSC on identical physics. Reward = ported phased
peg reward. reset seats the peg in the gripper + randomizes the hole pose.
Not batched yet (num_envs=1); replicate() later.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # else JAX grabs ~75% of VRAM at init

import jax.numpy as jnp
import numpy as np
import warp as wp

import newton
import peg_reward
import peg_scene_newton as scene
from controllers import JointPositionController

ASSET_HEIGHT = 0.025
HOLE_NOMINAL = np.array([0.6, 0.0, 0.05])
HOLE_XY_NOISE = 0.02
HOLE_Z_NOISE = 0.01
HOLE_JOINT = 11
ARM_DOF = 7
SETTLE_STEPS = 30          # gravity-comp settle so the controller captures its target at rest


def _wxyz(q_xyzw):
    return jnp.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


class PegEnv:
    def __init__(self, controller=None, episode_length=100, seed=0, weld=True):
        self.controller = controller or JointPositionController()
        self.model = scene.build_model(
            arm_control_cb=lambda b: self.controller.configure(b, list(scene.ARM_Q), scene.FINGER),
            weld=weld)
        # NOTE: cone="elliptic" + impratio=50 (from the original env's stiff-
        # insertion intent) destabilize Newton's SolverMuJoCo here — the free arm
        # gains energy and flies up. Default cone/impratio are stable. Revisit if
        # tight-clearance insertion needs the stiffer contact model.
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=True,
            nconmax=1024, njmax=4096, iterations=100, ls_iterations=50)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.device = wp.get_device()

        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self._arm_init = np.asarray(scene.ARM_Q, dtype=np.float32)
        self.finger = scene.FINGER
        self.lo = self.model.joint_limit_lower.numpy()[:ARM_DOF].copy()
        self.hi = self.model.joint_limit_upper.numpy()[:ARM_DOF].copy()
        self.ndof = self.model.joint_dof_count
        self.hole_pos = HOLE_NOMINAL.copy()

        self.act_dim = self.controller.action_dim
        self.obs_dim = 7 + 7 + 3 + 4
        self.steps = 0
        self.controller.setup(self)

    # -- helpers ------------------------------------------------------------
    def _set_hole(self, pos):
        Xp = self.model.joint_X_p.numpy()
        Xp[HOLE_JOINT][:3] = pos
        self.model.joint_X_p.assign(Xp)
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        self.hole_pos = np.asarray(pos, dtype=np.float32)

    def _seat_peg(self):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        h = self.state_0.body_q.numpy()[scene.HAND_IDX]
        hand = wp.transform(wp.vec3(*h[:3]), wp.quat(*h[3:7]))
        peg = wp.transform_multiply(hand, wp.transform(scene.WELD_OFFSET, wp.quat_identity()))
        q = self.model.joint_q.numpy()
        q[9:12] = [peg.p[0], peg.p[1], peg.p[2]]
        q[12:16] = [peg.q[0], peg.q[1], peg.q[2], peg.q[3]]
        self.model.joint_q.assign(q)

    def _obs(self):
        jq = self.state_0.joint_q.numpy()
        jqd = self.state_0.joint_qd.numpy()
        peg = self.state_0.body_q.numpy()[scene.PEG_BODY_IDX]
        peg_rel = peg[:3] - self.hole_pos
        return np.concatenate([jq[:7], jqd[:7], peg_rel, peg[3:7]]).astype(np.float32)

    # -- gym API ------------------------------------------------------------
    def reset(self):
        q = self.model.joint_q.numpy()
        q[:7] = self._arm_init
        q[7:9] = self.finger
        self.model.joint_q.assign(q)
        self._seat_peg()
        off = self.rng.uniform(
            [-HOLE_XY_NOISE, -HOLE_XY_NOISE, -HOLE_Z_NOISE],
            [HOLE_XY_NOISE, HOLE_XY_NOISE, HOLE_Z_NOISE]).astype(np.float32)
        self._set_hole(HOLE_NOMINAL + off)
        self.solver.reset(self.state_0)
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
        tq = self.control.joint_target_pos.numpy()
        tq[:ARM_DOF] = self._arm_init
        tq[ARM_DOF:ARM_DOF + 2] = self.finger
        self.control.joint_target_pos.assign(tq)
        self.control.joint_f.zero_()
        for _ in range(SETTLE_STEPS):
            self.state_0.clear_forces()
            bias = self.solver.mjw_data.qfrc_bias.numpy().reshape(-1)
            jf = self.control.joint_f.numpy()
            jf[:ARM_DOF] = bias[:ARM_DOF]
            self.control.joint_f.assign(jf)
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
        self.controller.set_action(self, action)
        self._substep()
        self.steps += 1
        peg = self.state_0.body_q.numpy()[scene.PEG_BODY_IDX]
        reward = float(peg_reward.compute_reward(
            held_pos=jnp.asarray(peg[:3]), held_quat=_wxyz(peg[3:7]),
            target_pos=jnp.asarray(self.hole_pos), target_quat=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
            peg_z=jnp.asarray(peg[2]),
            hole_top_z=jnp.asarray(self.hole_pos[2] + ASSET_HEIGHT)))
        done = self.steps >= self.episode_length
        return self._obs(), reward, done, {"truncation": float(done)}
