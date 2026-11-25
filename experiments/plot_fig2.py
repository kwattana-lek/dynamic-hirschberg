"""Figure 2: representative sequences from synthetic_dataset_robust (the training data).
Plots both channels (L = affected eye, R = fellow eye); blink frames (value 0.0) are shaded."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "synthetic_dataset_robust")
pick = {"Healthy": "Healthy_0003.csv", "Intermittent": "Intermittent_0005.csv", "Constant": "Constant_0002.csv"}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for i, (c, f) in enumerate(pick.items()):
    df = pd.read_csv(os.path.join(D, c, f)); t = np.arange(len(df)) / 30
    for col, lab, ls in [("L_ratio", "Left (affected) eye", "-"), ("R_ratio", "Right (fellow) eye", "--")]:
        v = df[col].values.astype(float); blink = v == 0.0
        ax[i].plot(t, np.where(blink, np.nan, v), ls, lw=1.6, label=lab)
        for k in np.flatnonzero(blink): ax[i].axvspan(t[k], t[k] + 1/30, color="grey", alpha=.25, lw=0)
    ax[i].set_title(f"({'ABC'[i]}) {c}"); ax[i].set_xlabel("Time (s)"); ax[i].grid(alpha=.3, ls="--")
ax[0].set_ylabel("Normalized displacement (iris diameters)"); ax[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "figure2_trajectories.png"), dpi=300)
print("saved")
