"""Interactive joint playground for the G1+Franka frankenrobot.

Builds the composed model (G1 with its right arm replaced by a Franka arm),
pins the pelvis, zeroes gravity, and drives every joint with a position servo
so you can pose the robot from imgui sliders in the viewer.

Zero gravity + fixed base = a clean kinematic playground: each slider sets a
joint target and the joint goes there (no balance, no sag, no tuning).

Run:  uv run python g1_franka_play.py            # GL viewer with sliders
      uv run python g1_franka_play.py --gravity  # enable gravity (will sag)
"""

import argparse

import numpy as np
import warp as wp

import newton
from g1_franka_compose import compose

KP, KD = 200.0, 20.0          # position-servo gains
SUBSTEPS = 4
SIM_DT = 1.0 / (60 * SUBSTEPS)
ONE_DOF = {int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)}


def build(gravity: bool):
    xml = compose(fix_base=True)
    builder = newton.ModelBuilder()
    builder.add_mjcf(xml)
    builder.gravity = -9.81 if gravity else 0.0

    # Position servos on every dof. The stripped MJCF leaves joints in NONE mode,
    # so we must flip each dof to POSITION target mode AND set PD gains — Newton
    # then synthesizes a position actuator per dof.
    n = builder.joint_dof_count
    builder.joint_target_mode[:] = [newton.JointTargetMode.POSITION] * n
    builder.joint_target_ke[:] = [KP] * n
    builder.joint_target_kd[:] = [KD] * n

    # Per-dof labels (base fixed => every joint is 1-dof, so joints line up with
    # dofs; skip any fixed joints just in case).
    labels = [name.split("/")[-1]
              for name, jt in zip(builder.joint_label, builder.joint_type)
              if int(jt) in ONE_DOF]
    assert len(labels) == n, f"label/dof mismatch: {len(labels)} vs {n}"

    model = builder.finalize()
    return model, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gravity", action="store_true", help="enable gravity (sags)")
    ap.add_argument("--headless", action="store_true", help="no window (sanity check)")
    args = ap.parse_args()

    model, labels = build(args.gravity)
    n = model.joint_dof_count
    print(f"[play] {n} controllable joints, gravity={'on' if args.gravity else 'off'}")

    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Slider state: targets start at the model's default pose; ranges from joint
    # limits, with a sane fallback for unlimited joints.
    default = model.joint_q.numpy().copy()
    targets = default.copy()
    lo = model.joint_limit_lower.numpy().copy()
    hi = model.joint_limit_upper.numpy().copy()
    for i in range(n):
        if hi[i] - lo[i] < 1e-3:        # no real limit -> generic range
            lo[i], hi[i] = -3.1416, 3.1416

    wp.copy(control.joint_target_q, model.joint_q)

    # Capture the substep loop as a CUDA graph; we overwrite joint_target_q each
    # frame from the slider targets and replay. State handles kept in a list so
    # the swap happens inside the captured graph.
    states = [state_0, state_1]

    def simulate_swap():
        for _ in range(SUBSTEPS):
            states[0].clear_forces()
            solver.step(states[0], states[1], control, None, SIM_DT)
            states[0], states[1] = states[1], states[0]

    with wp.ScopedCapture() as cap:
        simulate_swap()
    graph = cap.graph

    if args.headless:
        # Sanity check: command a +0.3 offset CLIPPED to joint limits and confirm
        # joints track it (clipping avoids penalizing the servo for unreachable
        # past-limit targets).
        targets[:] = np.clip(default + 0.3, lo, hi)
        for _ in range(300):
            control.joint_target_q.assign(targets)
            wp.capture_launch(graph)
        wp.synchronize_device()
        err = float(np.abs(states[0].joint_q.numpy() - targets).max())
        print(f"[play] headless tracking test: max |q-target| = {err:.4f} (lower=better)")
        return

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)

    def gui(ui):
        ui.text("Joint targets (rad / m)")
        if ui.button("reset to default"):
            targets[:] = default
        for i, lbl in enumerate(labels):
            changed, val = ui.slider_float(f"{i}:{lbl}", float(targets[i]),
                                           float(lo[i]), float(hi[i]))
            if changed:
                targets[i] = val

    viewer.register_ui_callback(lambda ui: gui(ui), position="side")

    t = 0.0
    while viewer.is_running():
        control.joint_target_q.assign(targets)
        wp.capture_launch(graph)
        viewer.begin_frame(t)
        viewer.log_state(states[0])
        viewer.end_frame()
        t += 1 / 60
    viewer.close()


if __name__ == "__main__":
    main()
