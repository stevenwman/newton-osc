"""Square-peg / square-socket scene (variant of peg_scene_newton.py).

Same build pipeline (splice panda.xml inline, add_mjcf, re-add weld + finger-peg
excludes, seed arm pose) but the hole is 4 box walls + floor (inline primitives in
square_insert/scene.xml) and the peg is a box — so NO bore-mesh injection/copy.
Body indices are unchanged (only geom geometry differs), so HAND_IDX/PEG_BODY_IDX
match the cylindrical scene. See SQUARE_PEG_PLAN.md.

    PYTHONPATH=. .venv/bin/python peg_scene_square.py            # GL viewer
    PYTHONPATH=. .venv/bin/python peg_scene_square.py --headless # weld-holds check
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import warp as wp
import newton

ROOT = Path(__file__).parent / "assets" / "factory"
SQUARE = ROOT / "square_insert"
PANDA = ROOT / "franka_panda"
BUILD = Path("/tmp/square_peg_build")

# Reused unchanged from the cylindrical scene (peg half-length 21mm == cylinder, so
# the weld offset / spawn pose hold).
# Raised +10cm vs the cylindrical pose (FK-solved) so the peg tip spawns well
# clear of the 88mm slab top (world z=0.085) — the wide slab means any small xy
# offset would otherwise clip the plate at spawn. fingertip z 0.150 -> 0.250.
ARM_Q = [-0.0168, 0.1839, 0.0178, -2.0761, -0.0083, 2.3806, 0.7924]
FINGER = 0.04
KP, KD = 400.0, 40.0
WELD_OFFSET = wp.vec3(0.0, 0.0, 0.130)
HAND_IDX = 8
PEG_BODY_IDX = 12
N_ARM_DOF = 9
PEG_Q = slice(9, 16)
SUBSTEPS = 4
SIM_DT = 0.002


def build_scene_xml() -> str:
    """Splice panda.xml inline (no bore tiles — the square walls are inline boxes)."""
    s = (SQUARE / "scene.xml").read_text()
    panda = (PANDA / "panda.xml").read_text()
    inner = re.search(r"<mujoco[^>]*>(.*)</mujoco>", panda, re.DOTALL).group(1)
    inner = re.sub(r"<compiler[^/]*/>", "", inner)
    inner = re.sub(r"<option[^/]*/>", "", inner)
    inner = re.sub(r"<actuator>.*?</actuator>", "", inner, flags=re.DOTALL)
    inner = re.sub(r"<keyframe>.*?</keyframe>", "", inner, flags=re.DOTALL)
    return re.sub(r'<include\s+file="\.\./franka_panda/panda\.xml"\s*/>', inner, s)


def materialize():
    """Write the assembled scene + panda meshes flat into BUILD (no bore meshes)."""
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "scene.xml").write_text(build_scene_xml())
    for p in (PANDA / "assets").iterdir():
        if p.is_file():
            shutil.copy(p, BUILD / p.name)
    return BUILD / "scene.xml"


def _build_builder(arm_control_cb=None, weld=True):
    scene = materialize()
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(str(scene))

    # Re-add finger<->peg collision filters (add_mjcf drops the <exclude>s).
    _bl = [l.split('/')[-1] for l in b.body_label]
    _sb = b.shape_body
    _peg_b = _bl.index("peg")
    _fingers = {_bl.index("left_finger"), _bl.index("right_finger")}
    _peg_sh = [i for i in range(len(_sb)) if _sb[i] == _peg_b]
    _fing_sh = [i for i in range(len(_sb)) if _sb[i] in _fingers]
    for _p in _peg_sh:
        for _f in _fing_sh:
            b.shape_collision_filter_pairs.append((_p, _f))

    # Re-add the grasp weld (add_mjcf drops the scene <weld>).
    if weld:
        b.add_equality_constraint_weld(
            body1=HAND_IDX, body2=PEG_BODY_IDX,
            relpose=wp.transform(wp.vec3(0.0, 0.0, 0.130), wp.quat_identity()),
            torquescale=1.0,
        )

    cfg = ARM_Q + [FINGER, FINGER]
    for i, v in enumerate(cfg):
        b.joint_q[i] = v
        b.joint_target_q[i] = v
    if arm_control_cb is None:
        for i in range(N_ARM_DOF):
            b.joint_target_mode[i] = newton.JointTargetMode.POSITION
            b.joint_target_ke[i] = KP
            b.joint_target_kd[i] = KD
    else:
        arm_control_cb(b)
    return b


def build_model(arm_control_cb=None, weld=True, num_envs=1):
    b = _build_builder(arm_control_cb, weld)
    if num_envs == 1:
        return b.finalize()
    top = newton.ModelBuilder()
    top.replicate(b, num_envs, spacing=(0.0, 0.0, 0.0))
    return top.finalize()


def seat_peg(model, state):
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    h = state.body_q.numpy()[HAND_IDX]
    hand_tf = wp.transform(wp.vec3(*h[:3]), wp.quat(*h[3:7]))
    peg_tf = wp.transform_multiply(hand_tf, wp.transform(WELD_OFFSET, wp.quat_identity()))
    q = model.joint_q.numpy()
    q[9:12] = [peg_tf.p[0], peg_tf.p[1], peg_tf.p[2]]
    q[12:16] = [peg_tf.q[0], peg_tf.q[1], peg_tf.q[2], peg_tf.q[3]]
    model.joint_q.assign(q)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    model = build_model()
    print(f"[sq] imported: {model.body_count} bodies, {model.joint_dof_count} dof, "
          f"{model.equality_constraint_count} equality constraint(s)")

    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    state_0 = model.state(); state_1 = model.state(); control = model.control()
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
        hand = bq[HAND_IDX][:3]; peg = bq[PEG_BODY_IDX][:3]
        gap = float(np.linalg.norm(np.array(peg) - np.array(hand)))
        print(f"[sq] hand moved {np.linalg.norm(hand - h0):.4f} m over 120 steps")
        print(f"[sq] peg-hand dist = {gap:.4f} m (weld ~0.13) -> "
              f"{'WELD HOLDS' if abs(gap - 0.13) < 0.05 else 'WELD SLIPPED?'}")
        return

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(1.4, -1.0, 0.8), pitch=-20.0, yaw=130.0)
    t = 0.0
    while viewer.is_running():
        simulate()
        viewer.begin_frame(t); viewer.log_state(states[0]); viewer.end_frame()
        t += 1 / 60
    viewer.close()


if __name__ == "__main__":
    main()
