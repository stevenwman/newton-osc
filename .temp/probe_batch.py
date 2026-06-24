"""Probe the replicated peg-scene layout before rewriting the env for batching.

Answers: do counts divide by N? where are the weld eq-constraints after replicate?
what is the joint_X_p stride for the hole joint? is mjw_data.nworld == N?
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import numpy as np
import warp as wp
import newton
import peg_scene_newton as scene

N = 2
model = scene.build_model(weld=True, num_envs=N)
print(f"[counts] bodies={model.body_count} joints={model.joint_count} "
      f"dof={model.joint_dof_count} coord={model.joint_coord_count} "
      f"eq={model.equality_constraint_count}")
for nm, c in [("body", model.body_count), ("joint", model.joint_count),
              ("dof", model.joint_dof_count), ("coord", model.joint_coord_count),
              ("eq", model.equality_constraint_count)]:
    print(f"   {nm:6s} // N = {c}/{N} = {c/N}  ({'OK int' if c % N == 0 else 'NOT DIVISIBLE'})")

njpw = model.joint_count // N
print(f"[joint_X_p] njoint_per_world={njpw}; hole local idx={scene.HOLE_JOINT if hasattr(scene,'HOLE_JOINT') else 11}")
HOLE_LOCAL = 11
Xp = model.joint_X_p.numpy()
for w in range(N):
    idx = w * njpw + HOLE_LOCAL
    print(f"   world {w}: joint_X_p[{idx}] pos = {Xp[idx][:3]}")

solver = newton.solvers.SolverMuJoCo(
    model, use_mujoco_contacts=True, nconmax=2048, njmax=8192,
    iterations=100, ls_iterations=50)
M = solver.mjw_model
d = solver.mjw_data
print(f"[mjw] nworld={d.nworld}  qvel.shape={tuple(d.qvel.shape)}  "
      f"xpos.shape={tuple(d.xpos.shape)}")
print(f"[mjw] neq={M.neq if hasattr(M,'neq') else '?'}  "
      f"eq_type={M.eq_type.numpy() if hasattr(M,'eq_type') else '?'}")
for attr in ("eq_solref", "eq_solimp", "eq_type", "eq_obj1id", "eq_obj2id"):
    if hasattr(M, attr):
        a = getattr(M, attr).numpy()
        print(f"   {attr}: shape={a.shape}\n{a}")

# body_q peg index per world
state = model.state()
newton.eval_fk(model, model.joint_q, model.joint_qd, state)
bq = state.body_q.numpy()
nbpw = model.body_count // N
print(f"[body_q] nbody_per_world={nbpw}; peg local idx=12")
for w in range(N):
    idx = w * nbpw + 12
    print(f"   world {w}: body_q[{idx}] pos = {bq[idx][:3]}")
print("DONE")
