"""Backend-validation harness: batched Franka arms in Newton, JAX interop.

Goal is NOT a polished RL env. It is a probe to answer one question: can the
Newton / MuJoCo-Warp backend drive many Franka arms in parallel and exchange
state with JAX cleanly enough to train on. It checks four things:

  1. throughput  - env-steps/sec across an env-count sweep (backend-only and
                   with the full JAX round-trip, so you see interop overhead)
  2. reset       - per-world masked reset does not corrupt sibling worlds
  3. obs/action  - obs is [N, obs_dim], actions are routed per-env
  4. determinism - same seed+actions -> same trajectory (within tolerance;
                   GPU contact solvers are not guaranteed bit-exact)

Run:  uv run python franka_batch_env.py
      uv run python franka_batch_env.py --num-envs 1024 --steps 200
"""

import os

# JAX preallocates ~75% of VRAM by default, which starves Warp/Newton on the
# same GPU. Disable it BEFORE importing jax so both share the pool on demand.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

import newton

# Franka fr3 home pose: 7 arm joints (from the brick-stacking example) + 2
# fingers held open (joint limit ~0.04 m).
FRANKA_INIT_Q = [
    -3.6802115e-03,
    2.3901723e-02,
    3.6804110e-03,
    -2.3683236e00,
    -1.2918962e-04,
    2.3922248e00,
    7.8549200e-01,
    0.04,
    0.04,
]
# PD gains / limits per joint (arm then fingers), same as the example.
JOINT_TARGET_KE = [400, 400, 400, 400, 400, 400, 400, 100, 100]
JOINT_TARGET_KD = [40, 40, 40, 40, 40, 40, 40, 10, 10]
JOINT_EFFORT = [87, 87, 87, 87, 12, 12, 12, 100, 100]
JOINT_ARMATURE = [0.3] * 4 + [0.11] * 3 + [0.15] * 2


class FrankaBatchEnv:
    """N independent fixed-base Franka arms stepped together on the GPU.

    obs    = [joint_pos (ncoord), joint_vel (ndof)] per world -> [N, obs_dim]
    action = joint position targets (PD control)              -> [N, act_dim]
    """

    def __init__(self, num_envs, fps=50, substeps=4, device=None):
        self.num_envs = num_envs
        self.substeps = substeps
        self.sim_dt = 1.0 / (fps * substeps)
        self.device = device or wp.get_device()

        # Build one arm, then replicate into `num_envs` worlds. Worlds are kept
        # at the origin (no spacing) for numerical stability, as Newton advises.
        arm = self._build_arm()
        scene = newton.ModelBuilder()
        scene.replicate(arm, num_envs, spacing=(0.0, 0.0, 0.0))
        self.model = scene.finalize()

        self.ncoord = self.model.joint_coord_count // num_envs
        self.ndof = self.model.joint_dof_count // num_envs
        self.obs_dim = self.ncoord + self.ndof
        self.act_dim = self.ndof

        # MuJoCo-Warp solver, MuJoCo-internal contacts (no separate collision
        # pipeline needed for a fixed-base arm probe).
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=True)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_q, self.model.joint_q)

        # Capture the substep loop once as a CUDA graph; replay each step.
        self._capture()

    def _build_arm(self):
        b = newton.ModelBuilder()
        # SolverMuJoCo needs its custom attribute schema registered on the
        # builder before geometry is added.
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        b.joint_q[:9] = FRANKA_INIT_Q
        b.joint_target_q[:9] = FRANKA_INIT_Q
        b.joint_target_ke[:9] = JOINT_TARGET_KE
        b.joint_target_kd[:9] = JOINT_TARGET_KD
        b.joint_effort_limit[:9] = JOINT_EFFORT
        b.joint_armature[:9] = JOINT_ARMATURE
        return b

    def _simulate(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self):
        with wp.ScopedCapture() as capture:
            self._simulate()
        self.graph = capture.graph

    def _obs(self):
        """Read current joint state as a JAX array [N, obs_dim] (zero-copy view)."""
        q = wp.to_jax(self.state_0.joint_q).reshape(self.num_envs, self.ncoord)
        qd = wp.to_jax(self.state_0.joint_qd).reshape(self.num_envs, self.ndof)
        return jnp.concatenate([q, qd], axis=1)

    def step(self, actions, sync=True):
        """Advance one control frame. `actions` is JAX [N, act_dim] of position
        targets. Returns obs as JAX [N, obs_dim].

        sync=True makes the JAX<->Warp handoff stream-safe (correct but slower);
        set False only when you have already ordered the streams yourself.
        """
        if sync:
            actions = jax.block_until_ready(actions)
        # Zero-copy view of the JAX buffer, copied into the graph-captured
        # control buffer so the CUDA graph always reads a stable pointer.
        act_wp = wp.from_jax(jnp.reshape(actions, (-1,)))
        wp.copy(self.control.joint_target_q, act_wp)

        wp.capture_launch(self.graph)
        if sync:
            wp.synchronize_device(self.device)
        return self._obs()

    def step_backend_only(self):
        """Replay the physics graph with the current control targets, no JAX."""
        wp.capture_launch(self.graph)

    def reset(self, world_mask=None):
        """Reset all worlds (mask=None) or a boolean-masked subset to the model
        default pose, clearing MuJoCo's internal warm-start buffers per world."""
        wm = None if world_mask is None else wp.array(world_mask, dtype=wp.bool, device=self.device)
        self.solver.reset(self.state_0, world_mask=wm)
        self.solver.reset(self.state_1, world_mask=wm)
        return self._obs()


# --------------------------------------------------------------------------- #
# Validation checks
# --------------------------------------------------------------------------- #


def check_throughput(num_envs, steps, warmup=10):
    print(f"\n[1] THROUGHPUT  (num_envs={num_envs}, steps={steps})")
    env = FrankaBatchEnv(num_envs)
    print(f"    obs_dim={env.obs_dim}  act_dim={env.act_dim}  ndof/env={env.ndof}")

    # --- backend-only: pure graph replay, control targets held fixed ---------
    for _ in range(warmup):
        env.step_backend_only()
    wp.synchronize_device(env.device)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step_backend_only()
    wp.synchronize_device(env.device)
    dt_backend = time.perf_counter() - t0
    sps_backend = steps * num_envs / dt_backend

    # --- full JAX round-trip: random actions in, obs out, every step ---------
    key = jax.random.PRNGKey(0)
    init = jnp.asarray(FRANKA_INIT_Q, dtype=jnp.float32)
    act = jnp.broadcast_to(init, (num_envs, env.act_dim))
    for _ in range(warmup):
        key, k = jax.random.split(key)
        act = init + 0.05 * jax.random.normal(k, (num_envs, env.act_dim))
        env.step(act)
    t0 = time.perf_counter()
    for _ in range(steps):
        key, k = jax.random.split(key)
        act = init + 0.05 * jax.random.normal(k, (num_envs, env.act_dim))
        obs = env.step(act)
    jax.block_until_ready(obs)
    dt_jax = time.perf_counter() - t0
    sps_jax = steps * num_envs / dt_jax

    print(f"    backend-only : {sps_backend:>12,.0f} control-steps/s  ({dt_backend*1e3/steps:.2f} ms/step)")
    print(f"    with JAX I/O : {sps_jax:>12,.0f} control-steps/s  ({dt_jax*1e3/steps:.2f} ms/step)")
    print(f"    (each control step = {env.substeps} sim substeps -> "
          f"{sps_backend*env.substeps:,.0f} sim-steps/s backend)")
    return sps_backend, sps_jax


def check_reset():
    print("\n[2] RESET  (per-world masked)")
    n = 8
    env = FrankaBatchEnv(n)
    init_obs = np.asarray(env.reset())  # all worlds at default pose

    # Drive every world away from the default with a fixed offset.
    off = jnp.asarray(FRANKA_INIT_Q, dtype=jnp.float32) + 0.3
    act = jnp.broadcast_to(off, (n, env.act_dim))
    for _ in range(40):
        obs = env.step(act)
    moved = np.asarray(obs)
    drift_before = np.abs(moved - init_obs).max()

    # Reset only even worlds.
    mask = np.zeros(n, dtype=bool)
    mask[::2] = True
    obs_after = np.asarray(env.reset(world_mask=mask))

    even_err = np.abs(obs_after[0::2] - init_obs[0::2]).max()   # should be ~0
    odd_err = np.abs(obs_after[1::2] - init_obs[1::2]).max()    # should be large

    print(f"    drift from default after 40 steps : {drift_before:.4f}")
    print(f"    reset even worlds -> err vs default: {even_err:.6f}  (expect ~0)")
    print(f"    odd  worlds       -> err vs default: {odd_err:.6f}  (expect >> 0)")
    ok = even_err < 1e-4 and odd_err > 1e-2
    print(f"    => {'PASS' if ok else 'FAIL'}: masked reset is isolated per-world")
    return ok


def check_obs_action():
    print("\n[3] OBS / ACTION ROUTING")
    n = 16
    env = FrankaBatchEnv(n)
    env.reset()

    # Per-world action: world i gets a target offset scaled by i. If actions are
    # routed per-env, the worlds must fan out (no two identical rows).
    init = jnp.asarray(FRANKA_INIT_Q, dtype=jnp.float32)
    scale = jnp.linspace(-0.4, 0.4, n).reshape(n, 1)
    act = init + scale * jnp.ones((n, env.act_dim))
    for _ in range(30):
        obs = env.step(act)
    obs = np.asarray(obs)

    shape_ok = obs.shape == (n, env.obs_dim)
    # Pairwise: how many world-rows are distinct?
    distinct = len({obs[i].tobytes() for i in range(n)})
    spread = obs[:, : env.ncoord].std(axis=0).max()  # variation across worlds
    routed_ok = distinct == n and spread > 1e-3

    print(f"    obs shape = {obs.shape}  (expect ({n}, {env.obs_dim}))  -> {'OK' if shape_ok else 'BAD'}")
    print(f"    distinct world states = {distinct}/{n}   cross-world std = {spread:.4f}")
    ok = shape_ok and routed_ok
    print(f"    => {'PASS' if ok else 'FAIL'}: obs batched and actions hit the right env")
    return ok


def check_determinism():
    print("\n[4] DETERMINISM  (same seed+actions twice)")
    n = 32
    steps = 50

    def rollout():
        env = FrankaBatchEnv(n)
        env.reset()
        key = jax.random.PRNGKey(123)
        init = jnp.asarray(FRANKA_INIT_Q, dtype=jnp.float32)
        for _ in range(steps):
            key, k = jax.random.split(key)
            act = init + 0.05 * jax.random.normal(k, (n, env.act_dim))
            obs = env.step(act)
        return np.array(obs)  # copy out of the shared dlpack buffer

    a = rollout()
    b = rollout()
    max_diff = np.abs(a - b).max()
    # Bit-exactness is not guaranteed on GPU; treat tiny drift as deterministic.
    ok = max_diff < 1e-3
    print(f"    max |obs_run1 - obs_run2| over {steps} steps = {max_diff:.3e}")
    print(f"    => {'PASS' if ok else 'FAIL'} (tol 1e-3; not bit-exact by design)")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=None,
                   help="single throughput run at this env count (skips sweep)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--sweep", type=int, nargs="+", default=[256, 1024, 4096])
    args = p.parse_args()

    print("=" * 70)
    print("Newton + MuJoCo-Warp backend validation : batched Franka + JAX")
    print(f"warp device: {wp.get_device()}   jax devices: {jax.devices()}")
    print("=" * 70)

    if args.num_envs is not None:
        check_throughput(args.num_envs, args.steps)
    else:
        for n in args.sweep:
            check_throughput(n, args.steps)

    results = {
        "reset": check_reset(),
        "obs/action": check_obs_action(),
        "determinism": check_determinism(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    for k, v in results.items():
        print(f"  {k:14s}: {'PASS' if v else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
