# Project memory — newton_manip

Durable notes for this project. (Auto-loaded project memory at
`~/.claude/projects/.../memory/MEMORY.md` just redirects here.)

---

## Newton model composition vs MuJoCo MjSpec

Newton `ModelBuilder` is an **append-only accumulator**, NOT a live editable spec like
`mujoco.MjSpec`. It can GRAFT subtrees (`add_urdf/add_mjcf(..., parent_body=idx,
base_joint={...})`, or `add_builder` + `add_joint(type, parent, child)`; bodies looked
up via `builder.body_label.index(name)`). It CANNOT remove / replace / detach an
existing subtree in memory; `finalize()` freezes the model. So "replace G1's arm with
Franka joints" is not a live Newton op — needs source surgery (edit URDF/MJCF, or do
attach/detach in `mujoco.MjSpec` then `spec.to_xml()` → `add_mjcf()`). Since Newton's
SolverMuJoCo IS mujoco_warp, the mjspec authoring workflow stays usable upstream.

VERIFIED working recipe (`g1_franka_compose.py`, repo root): replaced G1's right arm with
the Franka arm via mjspec → Newton. Steps that mattered: (1) `MjSpec.from_file` both;
(2) absolutize every `mesh.file` to an abs path + set `spec.meshdir=""` (two source models =
two meshdirs, one compiler meshdir can't cover both); (3) strip `spec.actuators`, `spec.sensors`
AND `spec.keys` — keyframes encode fixed qpos/ctrl sizes and break on `spec.delete`; (4) record
shoulder frame (`parent.add_frame()`, set pos/quat) BEFORE deleting arm; (5) `spec.delete(arm_body)`;
(6) `frame.attach_body(donor.body("link0"), "fr_", "")` — graft body choice = how much of the
donor: `link0`=whole arm, `link5`=last 3 joints; zero the graft body's `.pos` to mount flush;
(7) `spec.compile()` to validate, `spec.to_xml()`, `builder.add_mjcf(xml)`. Result: 37 dof
humanoid+franka arm, imports + views in Newton. Menagerie MJCFs from github google-deepmind/mujoco_menagerie
(sparse-clone unitree_g1 + franka_emika_panda); currently in scratchpad (not persisted in repo).

Interactive joint playground `g1_franka_play.py`: pins pelvis (compose(fix_base=True) deletes
the pelvis free joint), zero gravity, position servos, imgui sliders per joint. GOTCHA driving
imported joints: a stripped/actuator-less MJCF imports joints in target mode NONE, so
`control.joint_target_q` does nothing even with PD gains set. Must set per-dof
`builder.joint_target_mode[:] = newton.JointTargetMode.POSITION` (=1) AND `joint_target_ke/kd`;
then SolverMuJoCo synthesizes a position actuator per dof and joints track (verified 0.005 rad).
Newton viewer UI: `viewer.register_ui_callback(lambda ui: gui(ui), position="side")`; in gui,
`changed, val = ui.slider_float(label, val, lo, hi)`, `ui.button(label)`, `ui.text(...)`.
Update a target buffer each frame via `control.joint_target_q.assign(np_array)` then replay graph.

Simplest Newton sim (no example harness): `hello_newton.py` (repo root) — verified box
drops 3.0m → rests at 0.5m. MuJoCo→Newton map: from_xml_path→ModelBuilder+add_*+finalize;
MjData→model.state()x2+control()+contacts(); mj_step→solver.step(s0,s1,control,contacts,dt)+swap;
launch_passive→newton.viewer.ViewerGL; sync→begin_frame/log_state/end_frame. Pick a solver
(SolverXPBD light / SolverMuJoCo = full mujoco_warp). Run substeps + model.collide() yourself.

---

## Newton install & examples

- PyPI `newton-physics` is an **empty 1.6kB stub** — do NOT use it. Real package
  import name is `newton`.
- Install from git with the examples extra (bundles all deps):
  `uv add "newton[examples] @ git+https://github.com/newton-physics/newton.git"`.
  The `[examples]` extra pulls `sim` (mujoco-warp, mujoco — the solver many
  examples use), `importers` (trimesh, pycollada, scipy, usd-core), pyglet (GL
  viewer), GitPython (asset download), imgui_bundle. Do NOT hand-add these
  one-by-one.
- Env: uv venv, python 3.13, GPU = RTX 5080 (cuda:0, sm_120). Warp 1.14,
  newton 1.4.0.dev0.
- Run an example by file path under
  `.venv/lib/python3.13/site-packages/newton/examples/<cat>/example_*.py`.
  Brick stacking (Franka arm stacks 3 bricks, MuJoCo-Warp solver):
  `contacts/example_brick_stacking.py` — confirmed working in GL viewer.
- Common flags from `newton.examples.create_parser()`:
  `--viewer {gl,usd,rtx,rerun,null,viser}` (default gl), `--num-frames`,
  `--paused`, `--headless`, `--test`. GL viewer needs a display (DISPLAY was :1).
- First run JIT-compiles many Warp/MuJoCo kernels (~30s+); cached after in
  `~/.cache/warp/`.

---

## Peg-insertion scene ported to Newton (validated in viewer)

`peg_scene_newton.py` (repo root) recreates jax-learning's FactoryPegInsert physics scene in
Newton and shows it in the GL viewer (Franka holding welded peg over the bore ring). Source
assets copied into `assets/factory/` (peg_insert/scene.xml, franka_panda/, 32 bore tiles).
Build = port of env's `_build_scene_xml` (inject 32 bore-tile mesh+geom into placeholders,
regex-splice panda.xml inline stripping its <compiler>/<option>), then materialize scene.xml +
all 99 meshes FLAT into one dir (bare filenames resolve) and `builder.add_mjcf(path)`.

add_mjcf fidelity GOTCHAS (key learnings):
- WELD equality is SILENTLY DROPPED by add_mjcf (only the panda finger-coupling joint-equality
  survives). Re-add manually: `builder.add_equality_constraint_weld(body1=hand, body2=peg,
  relpose=..., torquescale=1.0)`. Its relpose z sign is OPPOSITE the intuitive/seat offset:
  +0.13 puts peg BELOW the hand. (deprecated API but works; warns about half-filled eq custom
  attrs — harmless, headless ran fine.)
- mocap body NOT supported -> imported as a fixed (jointed-to-world) body. Fine for a fixtured
  hole; means runtime hole DR via mocap_pos is lost (revisit for domain randomization).
- freejoint OK. Free-joint joint_q quat order is XYZW (matches warp body_q), NOT wxyz.
- Layout for this scene: dofs 0-6 arm, 7-8 fingers, 9-14 peg freejoint; hand body idx 8, peg
  body idx 12, hole_base idx 11. joint_q peg coords = q[9:16] (pos3 + quat xyzw).
Seat the peg at hand*translate(0,0,-0.13) via eval_fk before stepping or the stiff weld snaps.
NOT yet ported: OSC controller, obs_spec, reward, training (next).

GL viewer launch gotcha: relaunching while a prior ViewerGL window is alive can crash the new
one (exit 144); pkill -9 stale viewers + sleep before relaunch.

## FastSAC training pipeline on Newton peg env (smoke check PASSES)

End-to-end RL pipeline works: FastSAC trains on the Newton peg env, finite losses (C51 q_loss
decreasing 3.6->2.5, actor_loss finite), 20 episodes, no NaN. Files:
- `jax_rl/` — vendored FastSAC closure from jax-learning, PRUNED to the true 22-file closure
  (initially bulk-copied 44 via cp -r, then deleted 20 unused + trimmed buffers/__init__ &
  networks/__init__ which side-effect-imported rollout.py/contraction_metric.py). Closure =
  algos/fast_sac.py, configs/{fast_sac_config,networks_config}.py, networks/{builders,activations,
  distributions}.py + encoders/mlp.py + heads/{gaussian,value,deterministic,q_distributional}.py
  (builders imports value+deterministic at load), buffers/jax_replay_buffer.py,
  utils/{distributional,polyak}.py + package __init__s. configs/__init__.py REPLACED with minimal
  export (orig pulls env_presets -> heavy chain). algos/__init__.py emptied. Deps: flax, optax,
  distrax (+tfp-nightly). To find a runtime closure: import+exercise, then diff sys.modules
  (jax_rl.*) vs files on disk.
- `peg_reward.py` — phased reward copied as-is from factory/reward.py (pure jax, works).
- `peg_env.py` — single-env gym-like Newton peg env. Action = 7D joint-position delta (one of
  the two target action spaces; OSC next). obs = [arm_q(7),arm_qd(7),peg_rel_hole(3),peg_quat(4)]=21.
  reset seats peg + DR's hole via joint_X_p+notify.
- `train_peg.py` — stripped off-policy loop (warmup->collect->buffer->FastSAC.update), no
  wandb/ckpt/eval. Run: `uv run python train_peg.py`.

PORT GOTCHAS (all fixed):
- FastSAC.update REQUIRES batch["critic_obs"] & ["critic_next_obs"]; for symmetric obs inject
  =obs/=next_obs after buffer.sample.
- FastSAC update metrics keys = q1_loss/q2_loss/actor_loss/alpha/alpha_loss/entropy/q1_mean/q2_mean
  — there is NO "critic_loss" key (logging the wrong key returns the nan default and looks broken).
- SolverMuJoCo DEFAULT contact budgets overflow once peg touches bore ("nefc/broadphase overflow"
  -> NaN state). Must pass nconmax/njmax (used 1024/4096) + cone="elliptic", impratio=50, iterations=100.
- control.joint_target_q is per-DOF (15 here), NOT per-coord (16); free joint=6dof/7coord. Don't
  wp.copy joint_q into it; set arm dofs [0:7], fingers [7:9].
- reward expects quat WXYZ; Newton body_q is XYZW -> convert before compute_reward.
- Arm start pose hand-tuned via FK search (no IK) to place peg ~7cm above bore so reward is live.

SIMPLIFICATIONS to revisit for real training: single env (not batched via replicate); OSC
controller not yet ported (joint-pos only); symmetric obs (no privileged critic); C51 v_min/v_max
[-20,20] may clip large returns; hole DR magnitudes small (±2cm/±1cm). See [[peg-insertion-scene-ported]].

## Backend validation: batched RL with JAX

User's training framework is **JAX** (not torch). Harness: `franka_batch_env.py`
(repo root) — `FrankaBatchEnv` class + 4 checks.

**Verdict: Newton/MuJoCo-Warp backend works for parallel RL.** On RTX 5080,
fixed-base Franka arm (9 dof), arm-only NO contacts (`use_mujoco_contacts=True`,
no objects):
- 256 envs: 207k control-steps/s backend, 121k with JAX I/O
- 1024 envs: 721k / 429k
- 4096 envs: 1.98M / 1.25M  (= 7.9M sim-steps/s; 4 substeps/control-step). Near-linear.
- reset (per-world `solver.reset(state, world_mask)`), obs/action routing
  ([N,18]), determinism (~2e-6 drift, not bit-exact) all PASS.

Key API facts (Newton 1.4.0.dev0):
- Batch envs: `ModelBuilder.replicate(builder, world_count, spacing)`; keep
  spacing=0 for stability. joint_q/qd are world-major → reshape (N, ndof).
- Solver: `newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)`;
  step = `solver.step(s0, s1, control, None, dt)`. Call
  `SolverMuJoCo.register_custom_attributes(builder)` before adding geometry.
- Per-world reset: `solver.reset(state, world_mask=wp.array(bool[N]), flags=None)`
  resets masked worlds to model defaults + clears MuJoCo warm-start. Reset both
  state_0 and state_1.
- JAX interop: `wp.from_jax` / `wp.to_jax` (zero-copy dlpack). Must set
  `os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"` BEFORE importing jax or
  JAX eats VRAM. With CUDA graph, `wp.copy` actions into the captured
  `control.joint_target_q` buffer (don't reassign). dlpack deleter throws a
  benign TypeError at interpreter exit.
- obs = [state_0.joint_q, state_0.joint_qd]; PD position control via
  control.joint_target_q + joint_target_ke/kd.

**Caveats:** JAX per-step sync cost ~35-40% (probe uses naive
block_until_ready+synchronize; real training amortizes). Arm-only numbers —
contact-heavy manip (bricks/grasp) will be slower; re-bench with real scene.

## OSC controller fixed (was scrambled Jacobian) + RL training running

The peg-env OSC was unstable for a long time; root cause = a buffer-shape bug.
`mujoco_warp.jac` writes `(nworld, 3, nv)` but `OSCController` allocated
`(nw, nv, 3)` + a transpose, scrambling J. Garbage J -> wrong-sign EE velocity ->
negative damping -> "holds at ~0 error, diverges under any motion". Fix: allocate
`(nw, 3, nv)`, drop the transpose (`controllers.py` setup/apply). Now converges to
sub-mm and holds; cond(J M^-1 J^T) drops ~1e5(artifact)->1e2. Full recipe + the
diagnostic signature + sharp edges in `.context/lessons/osc-implementation-and-the-
scrambled-jacobian.md`. Secondary real fixes folded in: `mjw.forward()` before
reading mjw_data (forward-then-integrate staleness -> gravity-comp energy
injection); robosuite cross-product orientation error (vs fragile matrix-log);
absolute base-frame pose action; ridge 1e-4. Verified: `osc_track_demo.py` (GUI,
target+EE RGB axes) tracks random workspace poses; translation converges faster
than rotation (Kp 100 vs 30). The earlier "near-singular ARM_Q / weld / payload /
contacts" conclusions were all artifacts of the bad Jacobian.

RL: `train_peg_osc.py` = FastSAC on the OSC peg env (jax_rl-style: warmup->collect
->replay->C51 update), 6-DOF OSC action. Single-env smoke trains with finite,
decreasing losses; ~0.7GB VRAM (prealloc=false, mem_fraction=0.2 — good neighbor).
Overnight run in `runs/osc_peg/` (best_actor.pkl, ckpt.pkl, train.log). Biggest
TODO for real perf = batch the env via `replicate()` (OSC math already batched;
env routing from the backend-validation harness). Outline: `.context/training-osc-
peg.md`.
