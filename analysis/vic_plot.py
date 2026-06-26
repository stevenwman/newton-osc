"""Plot variable-impedance gains + peg forces from vic_analyze traj.npz files.

Outputs to runs/vic_plots/:
  gains_single.png  — 2 panels (Kp scale, zeta), single line each
  gains_axis.png    — 2 panels (Kp scale, zeta), 6 lines each (per task axis)
  forces.png        — peg force |F| and Fz over time, all 3 policies
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "runs/vic"   # e.g. "runs/sq_vic"
RUNS = {"fixed": f"{PREFIX}_fixed", "single": f"{PREFIX}_single", "axis": f"{PREFIX}_axis"}
AXES = ["x", "y", "z", "rx", "ry", "rz"]
OUT = f"{PREFIX}_plots"
os.makedirs(OUT, exist_ok=True)

data = {k: np.load(os.path.join(v, "traj.npz")) for k, v in RUNS.items()}

# ── Gains: single (1 line/panel) ────────────────────────────────────────────
d = data["single"]
t = np.arange(len(d["kp_scale"]))
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t, d["kp_scale"][:, 0], color="C0")        # global -> all axes identical
ax[0].set_ylabel("Kp scale (×base)"); ax[0].set_title("single: global stiffness")
ax[0].axhline(1.0, ls=":", c="gray", lw=0.8)
ax[1].plot(t, d["zeta"][:, 0], color="C1")
ax[1].set_ylabel("damping ratio ζ"); ax[1].set_xlabel("control step")
ax[1].axhline(1.0, ls=":", c="gray", lw=0.8); ax[1].set_title("single: global damping")
fig.tight_layout(); fig.savefig(f"{OUT}/gains_single.png", dpi=120); plt.close(fig)

# ── Gains: axis (6 lines/panel) ─────────────────────────────────────────────
d = data["axis"]
t = np.arange(len(d["kp_scale"]))
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for i, name in enumerate(AXES):
    ax[0].plot(t, d["kp_scale"][:, i], label=name)
    ax[1].plot(t, d["zeta"][:, i], label=name)
ax[0].set_ylabel("Kp scale (×base)"); ax[0].set_title("axis: per-axis stiffness")
ax[0].axhline(1.0, ls=":", c="gray", lw=0.8); ax[0].legend(ncol=6, fontsize=8)
ax[1].set_ylabel("damping ratio ζ"); ax[1].set_xlabel("control step")
ax[1].axhline(1.0, ls=":", c="gray", lw=0.8); ax[1].set_title("axis: per-axis damping")
fig.tight_layout(); fig.savefig(f"{OUT}/gains_axis.png", dpi=120); plt.close(fig)

# ── Forces: all 3 policies ──────────────────────────────────────────────────
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for k, c in zip(["fixed", "single", "axis"], ["C0", "C1", "C2"]):
    f = data[k]["force"]
    t = np.arange(len(f))
    ax[0].plot(t, np.linalg.norm(f, axis=1), color=c, label=k)
    ax[1].plot(t, f[:, 2], color=c, label=k)
ax[0].set_ylabel("|peg force|  (N)"); ax[0].set_title("peg constraint force (weld+contact)")
ax[0].legend()
ax[1].set_ylabel("peg Fz  (N)"); ax[1].set_xlabel("control step")
ax[1].axhline(0.0, ls=":", c="gray", lw=0.8); ax[1].set_title("vertical (insertion-axis) force")
ax[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/forces.png", dpi=120); plt.close(fig)

print(f"saved {OUT}/gains_single.png, gains_axis.png, forces.png")
for k in RUNS:
    f = data[k]["force"]
    print(f"  {k:7s} |F|mean={np.linalg.norm(f,axis=1).mean():.3f} |F|max={np.linalg.norm(f,axis=1).max():.3f}  "
          f"peg_z_min={data[k]['peg_z'].min():.3f}")
