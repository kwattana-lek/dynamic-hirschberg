"""Figure 3 (final): recommended CNN-only network in full, with the compared alternatives."""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures")
fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
def box(x, y, w, h, t, c, lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=c, ec="k", lw=lw, ls=ls))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=8.5)
def arr(x1, y1, x2, y2, ls="-"): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2, ls=ls))
ax.text(0.1, 4.35, "Recommended: CNN-only temporal convolutional network (24,067 parameters)", fontsize=10, fontweight="bold")
y0 = 2.9
blocks = [("Input 300 × 4\nu1, u2, mask1, mask2", "#f2f2f2", 1.8), ("Conv1D 64, k = 5\nReLU", "#cfe2f3", 1.5),
          ("MaxPool 2\nDropout 0.3", "#cfe2f3", 1.4), ("Conv1D 64, k = 5\nReLU", "#b6d7a8", 1.5),
          ("Global average\npooling", "#b6d7a8", 1.5), ("Dense 32\nReLU", "#f4cccc", 1.3), ("Dense 3, softmax\nH / I / C", "#f4cccc", 1.6)]
x = 0.1; xs = []
for t, c, w in blocks:
    box(x, y0, w, 1.0, t, c, lw=2.0); xs.append((x, w)); x += w + 0.25
for (x1, w1), (x2, _) in zip(xs[:-1], xs[1:]): arr(x1 + w1, y0 + 0.5, x2, y0 + 0.5)
ax.text(0.1, 2.25, "Alternatives compared (same input and dense head; front end = first Conv1D, MaxPool and Dropout):", fontsize=9.5, style="italic")
alts = [("CNN–BiLSTM (71,619)\nfront end → BiLSTM, 64 units/direction", "#fff2cc"),
        ("CNN–Transformer (46,467)\nfront end → positional embedding + 4-head encoder + GAP", "#ead1dc"),
        ("BiLSTM-only (39,555)\nBiLSTM, 64 units/direction, applied to the input", "#fce5cd")]
for i, (t, c) in enumerate(alts):
    box(0.1 + i * 4.3, 0.55, 4.05, 1.15, t, c, ls="--")
fig.tight_layout(); fig.savefig(os.path.join(F, "figure3_architectures.png"), dpi=300); print("ok")
