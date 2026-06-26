"""Interactive joint playground for the square-socket scene: imgui sliders per arm
joint (+ fingers) drive the Franka via Newton's position servo; the welded box peg
follows. Use it to hand-drive the peg into the square slab and eyeball clearances.

    PYTHONPATH=. DISPLAY=:1 .venv/bin/python joint_play_square.py
"""
import numpy as np
import warp as wp
import newton
import peg_scene_square as scene

ARM_DOF = 7
NF = 9  # 7 arm + 2 fingers (position-servo'd; peg freejoint dofs 9-14 follow the weld)

model = scene.build_model()                      # arm/fingers in POSITION servo mode
# Box peg vs flat slab walls makes many more contact constraints than the round
# peg (~87 efc seen when slamming the peg in, vs ~9 for the cylinder). Give njmax
# generous headroom so a hard manual slam doesn't overflow (default is < 87).
solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True,
                                     nconmax=256, njmax=1024)
state_0, state_1 = model.state(), model.state()
control = model.control()
scene.seat_peg(model, state_0)

lo = model.joint_limit_lower.numpy().copy()
hi = model.joint_limit_upper.numpy().copy()
targets = np.array(list(scene.ARM_Q) + [scene.FINGER, scene.FINGER], dtype=np.float32)
labels = [f"arm_{i}" for i in range(ARM_DOF)] + ["finger_L", "finger_R"]


def gui(ui):
    global targets
    ui.text("Franka joint control — square socket")
    for i in range(NF):
        loi, hii = (float(lo[i]), float(hi[i])) if hi[i] > lo[i] else (-3.14, 3.14)
        if i >= ARM_DOF:
            loi, hii = 0.0, 0.04
        changed, v = ui.slider_float(labels[i], float(targets[i]), loi, hii)
        if changed:
            targets[i] = v
    if ui.button("reset to ARM_Q"):
        targets[:] = list(scene.ARM_Q) + [scene.FINGER, scene.FINGER]


viewer = newton.viewer.ViewerGL()
viewer.set_model(model)
viewer.set_camera(pos=wp.vec3(1.4, -1.0, 0.8), pitch=-20.0, yaw=130.0)
viewer.register_ui_callback(lambda ui: gui(ui), position="side")

states = [state_0, state_1]
t = 0.0
while viewer.is_running():
    tq = control.joint_target_pos.numpy()
    tq[:NF] = targets
    control.joint_target_pos.assign(tq)
    for _ in range(scene.SUBSTEPS):
        states[0].clear_forces()
        solver.step(states[0], states[1], control, None, scene.SIM_DT)
        states[0], states[1] = states[1], states[0]
    viewer.begin_frame(t)
    viewer.log_state(states[0])
    viewer.end_frame()
    t += 1 / 60
viewer.close()
