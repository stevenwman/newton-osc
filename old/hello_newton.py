"""Simplest possible Newton sim — a box dropping on the ground, in a viewer.

Newton's analog of the minimal MuJoCo loop:

    mujoco.MjModel.from_xml_path(...)   ->  ModelBuilder() + add_* + finalize()
    mujoco.MjData(m)                    ->  model.state()  (x2, double-buffered)
    mujoco.mj_step(m, d)               ->  solver.step(s0, s1, control, contacts, dt)
    mujoco.viewer.launch_passive(...)  ->  newton.viewer.ViewerGL()
    viewer.sync()                       ->  begin_frame / log_state / end_frame

The extra pieces vs MuJoCo: you pick a solver object, you double-buffer state
(swap s0/s1 each substep), and you run substeps yourself.

Run:  uv run python hello_newton.py            # opens a GL window
      uv run python hello_newton.py --headless # no window (just steps)
"""

import argparse

import warp as wp

import newton

p = argparse.ArgumentParser()
p.add_argument("--headless", action="store_true", help="no window, just step")
args = p.parse_args()

# 1. Build the model (analog of from_xml_path) -----------------------------
builder = newton.ModelBuilder()
builder.add_ground_plane()
body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity()))
builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
model = builder.finalize()

# 2. Solver + state buffers (analog of MjData) -----------------------------
solver = newton.solvers.SolverXPBD(model, iterations=10)
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()

# 3. Viewer ----------------------------------------------------------------
viewer = newton.viewer.ViewerNull() if args.headless else newton.viewer.ViewerGL()
viewer.set_model(model)

# 4. Step loop (analog of mj_step + sync) ----------------------------------
fps, substeps = 60, 10
frame_dt = 1.0 / fps
sim_dt = frame_dt / substeps
sim_time = 0.0
max_frames = 120 if args.headless else 10**9  # headless: stop after 2 s

frame = 0
while viewer.is_running() and frame < max_frames:
    viewer.begin_frame(sim_time)
    for _ in range(substeps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0  # swap
    viewer.log_state(state_0)
    viewer.end_frame()
    sim_time += frame_dt
    frame += 1

box_z = state_0.body_q.numpy()[0][2]
print(f"done: {frame} frames, box height = {box_z:.3f} m (started at 3.0, box half = 0.5)")
viewer.close()
