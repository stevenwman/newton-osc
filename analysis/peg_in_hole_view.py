"""Visual margin check: no weld, peg frozen centered + aligned IN the square hole,
rendered static (no physics step) so you can orbit and eyeball the clearance.

    PYTHONPATH=. DISPLAY=:1 .venv/bin/python peg_in_hole_view.py
"""
import numpy as np
import warp as wp
import newton
import peg_scene_square as scene

HOLE = np.array([0.6, 0.0, 0.05])     # hole_base world pos (scene.xml)

model = scene.build_model(weld=False)  # NO weld — peg is a free body we place by hand
state = model.state()

q = model.joint_q.numpy()
q[:7] = scene.ARM_Q                     # arm parked at its (raised) start pose, out of the way
q[7:9] = scene.FINGER
# Peg free-joint coords 9:16 = pos(3) + quat(xyzw). Center it in the hole, aligned
# vertical (yaw=0). z mid-channel so it sits inside the 40mm socket (world 0.045..0.085).
q[9:12] = [HOLE[0], HOLE[1], 0.065]
q[12:16] = [0.0, 0.0, 0.0, 1.0]        # identity (xyzw) -> peg long-axis = world z
model.joint_q.assign(q)
newton.eval_fk(model, model.joint_q, model.joint_qd, state)

viewer = newton.viewer.ViewerGL()
viewer.set_model(model)
# Near top-down over the hole so the even gap around the square reads clearly.
viewer.set_camera(pos=wp.vec3(0.6, 0.0, 0.32), pitch=-80.0, yaw=90.0)
t = 0.0
while viewer.is_running():
    viewer.begin_frame(t)      # no solver.step -> peg stays frozen centered in the hole
    viewer.log_state(state)
    viewer.end_frame()
    t += 1 / 60
viewer.close()
