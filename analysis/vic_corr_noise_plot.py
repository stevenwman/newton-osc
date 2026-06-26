"""Plot correlated (sample-and-hold) noise robustness: succ% vs hold-length K."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "runs/sq_vic_corr_noise"
d = np.load(os.path.join(OUT, "corr_noise.npz"))
ks = d["hold_ks"]; sigma = float(d["sigma"])
COL = {"fixed": "C0", "single": "C1", "axis": "C2"}

fig, ax = plt.subplots(figsize=(8, 5))
for gm in ["fixed", "single", "axis"]:
    ax.plot(ks, 100 * d[gm], "-o", color=COL[gm], label=gm)
ax.set_xscale("log")
ax.set_xlabel("noise hold-length K (control steps; 1 = IID, 450 = constant/episode)")
ax.set_ylabel("success %")
ax.set_title(f"Square VIC: temporally-correlated obs-noise (sample-and-hold, σ={sigma}, target=both)")
ax.grid(alpha=0.3, which="both"); ax.set_ylim(-3, 103); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "corr_noise.png"), dpi=120)
print(f"saved {OUT}/corr_noise.png")
print("\nsucc% vs hold-K:")
print("  K     " + "  ".join(f"{int(k):>4d}" for k in ks))
for gm in ["fixed", "single", "axis"]:
    print(f"  {gm:6s}" + "  ".join(f"{100*v:4.0f}" for v in d[gm]))
