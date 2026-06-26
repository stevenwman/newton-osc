"""Plot obs-noise robustness curves from vic_noise_eval.npz: succ% vs sigma, one
panel per noise target (perception / goal / both), 3 policy curves each."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "runs/sq_vic_noise"
d = np.load(os.path.join(OUT, "noise_eval.npz"))
sig = d["sigmas"]
GAINS = ["fixed", "single", "axis"]
TARGETS = ["perception", "goal", "both", "full"]
COL = {"fixed": "C0", "single": "C1", "axis": "C2"}

fig, ax = plt.subplots(1, len(TARGETS), figsize=(5 * len(TARGETS), 4.5), sharey=True)
for j, tgt in enumerate(TARGETS):
    for gm in GAINS:
        y = 100 * d[f"{gm}__{tgt}"]
        ax[j].plot(sig, y, "-o", color=COL[gm], label=gm)
    ax[j].set_title(f"noise target: {tgt}")
    ax[j].set_xlabel("obs noise σ (normalized units)")
    ax[j].grid(alpha=0.3); ax[j].set_ylim(-3, 103)
    ax[j].legend()
ax[0].set_ylabel("success %")
fig.suptitle("Square VIC: obs-noise robustness (succ% vs σ)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "noise_robustness.png"), dpi=120)
print(f"saved {OUT}/noise_robustness.png")

# σ at which succ drops below 50% (robustness threshold), per policy/target
print("\nσ@50%-succ (robustness threshold; higher = more robust):")
for tgt in TARGETS:
    line = f"  {tgt:10s}"
    for gm in GAINS:
        y = 100 * d[f"{gm}__{tgt}"]
        below = np.where(y < 50)[0]
        thr = f"{sig[below[0]]:.2f}" if len(below) else f">{sig[-1]:.1f}"
        line += f"  {gm}={thr}"
    print(line)
