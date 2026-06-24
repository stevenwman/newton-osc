"""Recreate the FactoryPegInsert scene in Newton and validate it in a viewer.

Pipeline:
  1. assemble the factory MJCF (splice panda.xml + inject 32 bore tiles) — the
     exact scene the jax-learning env builds at runtime
  2. materialize it + all meshes into one flat dir (bare filenames resolve)
  3. newton.ModelBuilder().add_mjcf(...) — imports arm + freejoint peg + the
     grasp WELD (mocap hole -> fixed body, which is fine: the bore is fixtured)
  4. pose the arm, seat the peg in the gripper via FK, position-hold, step
  5. view: Franka holding the welded peg above the bore ring

This is scene/physics validation only — OSC control, obs, reward, training come
later. Run:
    uv run python peg_scene_newton.py            # GL viewer
    uv run python peg_scene_newton.py --headless # checks import + weld holds
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import warp as wp

import newton

ROOT = Path(__file__).parent / "assets" / "factory"
PEG = ROOT / "peg_insert"
PANDA = ROOT / "franka_panda"
BORE = PEG / "extracted" / "bore_tiles"
BUILD = Path("/tmp/claude-1001/-home-stevenman-Desktop-Work-Research-newton-manip"
             "/6ab1e160-f7b3-488c-94fa-9f85a993736c/scratchpad/peg_build")

# Gripper-down start pose placing the PEG TIP ~2.7cm ABOVE the bore opening
# (fingertip z~0.150). The old pose placed the *fingertip* 3cm above the bore but
# the peg hangs ~2.7cm below the fingertip, so its TIP spawned ~1.8cm BELOW the
# bore top (already inside -> no approach phase, broken task). jax_rl spawns the
# tip ~2cm above. OSC-calibrated, within joint limits, well-conditioned.
ARM_Q = [-0.0166, 0.3849, 0.0169, -2.0429, -0.0086, 2.4447, 0.7924]
FINGER = 0.04
KP, KD = 400.0, 40.0
# Peg seats at hand + R_hand·[0,0,+0.130] — SAME side as the fingertip site
# (+0.1034 along hand-z), matching the original env. Must agree with the weld
# relpose sign or the weld yanks the peg across on the first step.
WELD_OFFSET = wp.vec3(0.0, 0.0, 0.130)
HAND_IDX = 8
PEG_BODY_IDX = 12
N_ARM_DOF = 9                              # 7 arm + 2 fingers (peg freejoint after)
PEG_Q = slice(9, 16)                       # peg freejoint coords: pos(3) + quat(4)
SUBSTEPS = 4
SIM_DT = 0.002              # match the original env timestep (decimation 4 -> 125Hz control)


def build_scene_xml() -> str:
    """Inject bore tiles + splice panda inline (port of the env's _build_scene_xml)."""
    s = (PEG / "scene.xml").read_text()
    pieces = sorted(BORE.glob("bore_*.obj"))
    assets_block = "\n".join(
        f'    <mesh name="bore_{i:02d}" file="{p.name}"/>' for i, p in enumerate(pieces))
    geoms_block = "\n".join(
        f'      <geom name="bore_{i:02d}" type="mesh" mesh="bore_{i:02d}" '
        f'condim="6" friction="1.0 0.01 0.0001" solref="0.004 1" '
        f'solimp="0.98 0.995 0.001"/>' for i, _ in enumerate(pieces))
    s = s.replace("<!-- BORE_TILE_ASSETS -->", assets_block)
    s = s.replace("<!-- BORE_TILE_GEOMS injected here -->", geoms_block)

    panda = (PANDA / "panda.xml").read_text()
    inner = re.search(r"<mujoco[^>]*>(.*)</mujoco>", panda, re.DOTALL).group(1)
    inner = re.sub(r"<compiler[^/]*/>", "", inner)   # scene's compiler/option win
    inner = re.sub(r"<option[^/]*/>", "", inner)
    # Strip panda's <actuator>s: add_mjcf imports them as position servos (ctrl=0)
    # that drag the arm toward its home pose, fighting our controller. We drive
    # the arm ourselves (OSC torque via joint_f, or Newton's own position servo).
    inner = re.sub(r"<actuator>.*?</actuator>", "", inner, flags=re.DOTALL)
    # ...and the <keyframe> (its ctrl size no longer matches once actuators go).
    inner = re.sub(r"<keyframe>.*?</keyframe>", "", inner, flags=re.DOTALL)
    return re.sub(r'<include\s+file="\.\./franka_panda/panda\.xml"\s*/>', inner, s)


def materialize():
    """Write scene.xml + every mesh (panda + bore) flat into BUILD so the bare
    filenames in the assembled XML resolve for Newton's importer."""
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "scene.xml").write_text(build_scene_xml())
    for p in (PANDA / "assets").iterdir():
        if p.is_file():
            shutil.copy(p, BUILD / p.name)
    for p in BORE.glob("bore_*.obj"):
        shutil.copy(p, BUILD / p.name)
    return BUILD / "scene.xml"


def build_model(arm_control_cb=None, weld=True):
    """Build the peg scene. arm_control_cb(builder) configures arm joint control
    (modes/gains); if None, defaults to a position servo (used by the viewer).
    weld=False skips the grasp weld (peg falls away) — for isolating arm control."""
    scene = materialize()
    b = newton.ModelBuilder()
    b.add_mjcf(str(scene))

    # add_mjcf also drops the scene's <contact><exclude> finger<->peg pairs, so
    # the welded peg (sitting between the fingers) jams them with huge contact
    # forces -> the arm gets blasted. Re-add the collision filters.
    _bl = [l.split('/')[-1] for l in b.body_label]
    _sb = b.shape_body
    _peg_b = _bl.index("peg")
    _fingers = {_bl.index("left_finger"), _bl.index("right_finger")}
    _peg_sh = [i for i in range(len(_sb)) if _sb[i] == _peg_b]
    _fing_sh = [i for i in range(len(_sb)) if _sb[i] in _fingers]
    for _p in _peg_sh:
        for _f in _fing_sh:
            b.shape_collision_filter_pairs.append((_p, _f))

    # Newton's add_mjcf drops the scene's <weld> (only the panda finger-coupling
    # equality survives), so re-add the grasp weld: peg held 13 cm below the hand.
    # Newton's relpose z is opposite the seat offset in practice (+0.13 here puts
    # the peg below the hand, hanging toward the bore).
    if weld:
      b.add_equality_constraint_weld(
        body1=HAND_IDX, body2=PEG_BODY_IDX,
        relpose=wp.transform(wp.vec3(0.0, 0.0, 0.130), wp.quat_identity()),
        torquescale=1.0,
    )

    # Seed the arm pose (peg freejoint stays free; the weld controls it).
    cfg = ARM_Q + [FINGER, FINGER]
    for i, v in enumerate(cfg):
        b.joint_q[i] = v
        b.joint_target_q[i] = v
    # Arm control setup: caller-provided, else default position servo (viewer).
    if arm_control_cb is None:
        for i in range(N_ARM_DOF):
            b.joint_target_mode[i] = newton.JointTargetMode.POSITION
            b.joint_target_ke[i] = KP
            b.joint_target_kd[i] = KD
    else:
        arm_control_cb(b)
    return b.finalize()


def seat_peg(model, state):
    """Place the peg at hand * weld_offset so the stiff weld doesn't snap it
    across the room on the first step (mirrors the env's reset)."""
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    h = state.body_q.numpy()[HAND_IDX]                 # px,py,pz, qx,qy,qz,qw
    hand_tf = wp.transform(wp.vec3(*h[:3]), wp.quat(*h[3:7]))
    peg_tf = wp.transform_multiply(hand_tf, wp.transform(WELD_OFFSET, wp.quat_identity()))
    q = model.joint_q.numpy()
    q[9:12] = [peg_tf.p[0], peg_tf.p[1], peg_tf.p[2]]
    q[12:16] = [peg_tf.q[0], peg_tf.q[1], peg_tf.q[2], peg_tf.q[3]]  # warp quat = xyzw
    model.joint_q.assign(q)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    model = build_model()
    print(f"[peg] imported: {model.body_count} bodies, {model.joint_dof_count} dof, "
          f"{model.equality_constraint_count} equality constraint(s)")

    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    seat_peg(model, state_0)

    states = [state_0, state_1]

    def simulate():
        for _ in range(SUBSTEPS):
            states[0].clear_forces()
            solver.step(states[0], states[1], control, None, SIM_DT)
            states[0], states[1] = states[1], states[0]

    if args.headless:
        h0 = states[0].body_q.numpy()[HAND_IDX][:3]
        for _ in range(120):
            simulate()
        wp.synchronize_device()
        bq = states[0].body_q.numpy()
        hand = bq[HAND_IDX][:3]
        peg = bq[PEG_BODY_IDX][:3]
        # weld check: peg should remain ~13cm below the hand
        gap = float(np.linalg.norm(np.array(peg) - np.array(hand)))
        print(f"[peg] hand moved {np.linalg.norm(hand - h0):.4f} m over 120 steps")
        print(f"[peg] peg-hand distance = {gap:.4f} m (weld target ~0.13) -> "
              f"{'WELD HOLDS' if abs(gap - 0.13) < 0.05 else 'WELD SLIPPED?'}")
        return

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(1.4, -1.0, 0.8), pitch=-20.0, yaw=130.0)
    t = 0.0
    while viewer.is_running():
        simulate()
        viewer.begin_frame(t)
        viewer.log_state(states[0])
        viewer.end_frame()
        t += 1 / 60
    viewer.close()


if __name__ == "__main__":
    main()
