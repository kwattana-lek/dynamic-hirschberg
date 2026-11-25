"""Figure 1: DH workflow (measurement, synthetic generator, quality gate, CNN-only classifier). Output: results/figures/figure1_workflow.png"""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures")

fig, ax = plt.subplots(figsize=(14, 4.6)); ax.set_xlim(0, 14); ax.set_ylim(0, 4.6); ax.axis("off")
def box(x, y, w, h, t, c, fs=9, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=c, ec="k", lw=1.1, **kw))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2, ls="-", lab=None, lx=0, ly=0.12):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2, ls=ls))
    if lab: ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, lab, ha="center", fontsize=8, style="italic")
def frame(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc="none", ec=c, lw=1.3, ls="--"))
    ax.text(x + 0.1, y + h - 0.22, t, ha="left", fontsize=10.5, color=c, weight="bold")

# (i) measurement
frame(0.05, 2.35, 7.6, 2.15, "(i) Per-frame measurement", "#1f4e9c")
box(0.25, 2.7, 1.45, 1.1, "Video\nresampled to\n30 frames/s", "#e1f0fb")
box(2.0, 2.7, 1.6, 1.1, "MediaPipe\nFace Mesh:\niris landmarks", "#e1f0fb")
box(3.9, 2.7, 1.65, 1.1, "Pupil center,\ncorneal reflex,\niris diameter", "#e1f0fb")
box(5.85, 2.7, 1.65, 1.1, "Signed normalized\ndisplacement $u^{(e)}_t$\n+ validity $v^{(e)}_t$", "#b3dcf5")
for x1, x2 in [(1.7, 2.0), (3.6, 3.9), (5.55, 5.85)]: arr(x1, 3.25, x2, 3.25)

# (ii) generator
frame(0.05, 0.1, 7.6, 1.95, "(ii) Synthetic temporal sequence generator", "#b03030")
box(0.25, 0.35, 2.3, 1.05, "Class patterns, gaze shifts,\nnoise, blinks, dropout", "#fdecec")
box(2.95, 0.35, 2.0, 1.05, "Labeled 10-s\nsequences\n(Healthy / Intermittent\n/ Constant)", "#f9cfcf", fs=8.5)
box(5.35, 0.35, 2.15, 1.05, "Training of the\nclassifier", "#f9cfcf")
arr(2.55, 0.87, 2.95, 0.87); arr(4.95, 0.87, 5.35, 0.87)

# (iii) window, gate, classifier
frame(7.95, 0.1, 6.0, 4.4, "(iii) Window analysis", "#2e7d32")
box(8.15, 2.7, 1.6, 1.1, "Window of\n300 frames (10 s)\n$u^{(1)}, u^{(2)}, v^{(1)}, v^{(2)}$", "#e8f5e9", fs=8.5)
box(10.05, 2.7, 1.55, 1.1, "Quality gate\njoint validity $\\geq q$,\ngap $\\leq g$", "#fff2cc", fs=8.5)
box(11.95, 2.7, 1.85, 1.1, "CNN-only\ntemporal classifier", "#c8e6c9")
box(11.95, 0.35, 1.85, 1.05, "Alignment class:\nHealthy / Intermittent\n/ Constant", "#81c784", fs=8.5)
box(9.95, 0.35, 1.75, 1.05, "“Cannot assess”\n(with reason)", "#eeeeee", fs=8.5)
arr(7.5, 3.25, 8.15, 3.25); arr(9.75, 3.25, 10.05, 3.25)
arr(11.6, 3.25, 11.95, 3.25, lab="pass", ly=0.12)
arr(10.82, 2.7, 10.82, 1.4, lab="fail", lx=0.28, ly=0.0)
arr(12.87, 2.7, 12.87, 1.4)
ax.plot([6.42, 6.42, 13.88, 13.88], [0.35, 0.22, 0.22, 3.0], ":", color="k", lw=1.3)
arr(13.88, 3.0, 13.8, 3.0, ls=":"); ax.text(9.0, 0.27, "trained weights", fontsize=8, style="italic")
fig.tight_layout(); fig.savefig(os.path.join(F, "figure1_workflow.png"), dpi=300); plt.close(fig)
print("saved")
