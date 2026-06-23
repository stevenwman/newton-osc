"""mjspec surgery -> Newton: replace G1's right arm with Franka's last 3 joints.

Demonstrates the composability workflow we landed on:
  1. load both MJCFs into mujoco.MjSpec (live, editable model graph)
  2. lop off G1's right arm subtree (spec.delete)
  3. graft Franka's distal chain (link5 -> joints 5,6,7 + hand) onto the shoulder
     via frame.attach_body
  4. spec.to_xml() -> the composed MJCF string
  5. newton.ModelBuilder().add_mjcf(...) "takes care of the rest"

mjspec does the surgery Newton's append-only ModelBuilder can't. Run:
    uv run python g1_franka_compose.py            # headless, prints stats
    uv run python g1_franka_compose.py --view     # GL viewer (static rest pose)
"""

import argparse
import os

import mujoco
import warp as wp

import newton

MEN = "/tmp/claude-1001/-home-stevenman-Desktop-Work-Research-newton-manip/6ab1e160-f7b3-488c-94fa-9f85a993736c/scratchpad/mujoco_menagerie"
G1_XML = f"{MEN}/unitree_g1/g1.xml"
FR_XML = f"{MEN}/franka_emika_panda/panda.xml"
OUT_XML = "/tmp/claude-1001/-home-stevenman-Desktop-Work-Research-newton-manip/6ab1e160-f7b3-488c-94fa-9f85a993736c/scratchpad/g1_franka.xml"

ARM_ROOT = "right_shoulder_pitch_link"   # G1 right-arm subtree root -> delete
FRANKA_GRAFT = "link0"                    # Franka base link -> whole 7-DOF arm + hand


def absolutize_meshes(spec: mujoco.MjSpec, model_dir: str) -> None:
    """Rewrite every mesh file to an absolute path so the emitted XML resolves
    no matter where it (or its meshes) live. Two source models => two meshdirs,
    so a single compiler meshdir can't work; absolute per-mesh paths can."""
    base = os.path.join(model_dir, spec.meshdir) if spec.meshdir else model_dir
    for m in spec.meshes:
        if not os.path.isabs(m.file):
            m.file = os.path.join(base, m.file)
    spec.meshdir = ""


def strip_controls(spec: mujoco.MjSpec) -> None:
    """Drop actuators/sensors/keyframes — irrelevant for a kinematics/visual
    demo, and a guaranteed snag once we delete joints: keyframes encode a fixed
    qpos/ctrl size that no longer matches after surgery."""
    for coll in (spec.actuators, spec.sensors, spec.keys):
        for el in list(coll):
            spec.delete(el)


def find_parent(spec: mujoco.MjSpec, child_name: str):
    """Return the body whose direct child is `child_name` (BFS from worldbody)."""
    stack = [spec.worldbody]
    while stack:
        b = stack.pop()
        c = b.first_body()
        while c:
            if c.name == child_name:
                return b
            stack.append(c)
            c = b.next_body(c)
    return None


def compose(fix_base: bool = False) -> str:
    g1 = mujoco.MjSpec.from_file(G1_XML)
    fr = mujoco.MjSpec.from_file(FR_XML)

    absolutize_meshes(g1, os.path.dirname(G1_XML))
    absolutize_meshes(fr, os.path.dirname(FR_XML))
    strip_controls(g1)
    strip_controls(fr)

    # Optionally weld the pelvis to the world (drop the free joint) so the robot
    # hangs in place — handy for a joint-control playground where we don't want
    # to also solve balance.
    if fix_base:
        pelvis_joint = g1.body("pelvis").first_joint()
        if pelvis_joint is not None:
            g1.delete(pelvis_joint)

    # Record the shoulder mounting frame BEFORE deleting the arm.
    arm = g1.body(ARM_ROOT)
    parent = find_parent(g1, ARM_ROOT)
    assert arm is not None and parent is not None, "arm/parent not found"
    shoulder = parent.add_frame()
    shoulder.pos = arm.pos
    shoulder.quat = arm.quat

    # Surgery: remove the G1 right arm, graft the whole Franka arm at its base.
    # Zero the graft body's native offset so it mounts flush at the shoulder.
    g1.delete(arm)
    fr.body(FRANKA_GRAFT).pos = [0.0, 0.0, 0.0]
    shoulder.attach_body(fr.body(FRANKA_GRAFT), "fr_", "")

    model = g1.compile()  # validate in mujoco first
    print(f"[mujoco] composed OK: {model.nbody} bodies, {model.nq} qpos, {model.nv} dof")

    xml = g1.to_xml()
    with open(OUT_XML, "w") as f:
        f.write(xml)
    print(f"[mujoco] wrote composed MJCF -> {OUT_XML}")
    return xml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view", action="store_true", help="open GL viewer")
    args = p.parse_args()

    xml = compose()

    # Newton "takes care of the rest".
    builder = newton.ModelBuilder()
    builder.add_mjcf(xml)
    model = builder.finalize()
    print(f"[newton] imported: {model.body_count} bodies, "
          f"{model.joint_dof_count} dof, {model.joint_count} joints")

    # Static rest-pose view (no stepping -> humanoid won't collapse uncontrolled).
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    viewer = newton.viewer.ViewerGL() if args.view else newton.viewer.ViewerNull()
    viewer.set_model(model)
    frames = 10**9 if args.view else 3
    t, i = 0.0, 0
    while viewer.is_running() and i < frames:
        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.end_frame()
        t += 1 / 60
        i += 1
    viewer.close()
    print("[newton] done")


if __name__ == "__main__":
    main()
