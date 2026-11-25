"""Figures and pooled per-class metrics for the revised manuscript (v2)."""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "results"); F = os.path.join(R, "figures")
C = ["Healthy", "Intermittent", "Constant"]
d = np.load(os.path.join(H, "..", "..", "data", "synthetic_v2", "synthetic_v2.npz")); y = d["y"]
te = np.load(os.path.join(R, "predictions", "test_idx.npy")); yt = y[te]
pooled = {}
for n in ["CNN-BiLSTM", "CNN-only", "BiLSTM-only", "CNN-Transformer"]:
    P = np.concatenate([np.load(os.path.join(R, "predictions", f"{n}_seed{s}.npy")).argmax(1) for s in range(5)]); Y = np.tile(yt, 5)
    pr, rc, f1, sup = precision_recall_fscore_support(Y, P, labels=[0, 1, 2])
    pooled[n] = {"confusion": confusion_matrix(Y, P, labels=[0, 1, 2]).tolist(),
                 **{c: {"precision": pr[i], "recall": rc[i], "f1": f1[i], "support": int(sup[i])} for i, c in enumerate(C)}}
json.dump(pooled, open(os.path.join(R, "pooled_per_class.json"), "w"), indent=2, default=float)
cm = np.array(pooled["CNN-only"]["confusion"]); fig, ax = plt.subplots(figsize=(4.4, 3.9)); ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
ax.set_xticks(range(3), C); ax.set_yticks(range(3), C); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
fig.tight_layout(); fig.savefig(os.path.join(F, "confusion_cnn_only_pooled.png"), dpi=300); plt.close(fig)

# Figure 3: shared front end and compared temporal modules
fig, ax = plt.subplots(figsize=(13, 4.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")
def box(x, y, w, h, t, c): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=c, ec="k", lw=1.1)); ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=8.5)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1))
box(0.1, 1.6, 1.7, 1.0, "Input 300 × 4\nu1, u2,\nmask1, mask2", "#f2f2f2")
box(2.2, 1.6, 2.2, 1.0, "Conv1D 64, k = 5, ReLU\nMaxPool 2, Dropout 0.3\n(shared front end)", "#cfe2f3")
heads = [("Bidirectional LSTM\n64 units/direction\n→ 128  [CNN–BiLSTM, 71,619]", "#fff2cc", 3.2),
         ("Conv1D 64, k = 5 + global\naverage pooling\n[CNN-only, 24,067]", "#d9ead3", 1.95),
         ("Positional embedding +\nTransformer block (4 heads)\n+ GAP  [CNN–Transformer, 41,667]", "#ead1dc", 0.7)]
for t, c, yy in heads:
    box(5.0, yy, 3.6, 0.95, t, c); arr(4.4, 2.1, 5.0, yy + 0.47); arr(8.6, yy + 0.47, 9.2, 2.1)
box(9.2, 1.6, 1.6, 1.0, "Dense 32\nReLU", "#f4cccc"); arr(10.8, 2.1, 11.2, 2.1)
box(11.2, 1.6, 1.7, 1.0, "Dense 3, softmax\nHealthy / Intermittent\n/ Constant", "#f4cccc")
arr(1.8, 2.1, 2.2, 2.1)
ax.text(6.8, 0.25, "BiLSTM-only [39,555]: bidirectional LSTM applied directly to the input, same dense head", ha="center", fontsize=8.5, style="italic")
fig.tight_layout(); fig.savefig(os.path.join(F, "figure3_architectures.png"), dpi=300); plt.close(fig)

# Figure: real signals of the accepted windows
rv = os.path.join(R, "real_video"); fig, ax = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
W = 300                                                       # 10-s windows at 30 frames/s
PANELS = [("Strabismus", "252652595", 1, "Strabismus 6, window 2 (accepted)"),
          ("Strabismus", "124197414", 0, "Strabismus 1, window 1 (cannot assess: 54% jointly valid)")]
for a, (g, clip, w, title) in zip(ax, PANELS):
    df = pd.read_csv(os.path.join(rv, f"{g}_{clip}_signal.csv")).iloc[w*W:(w+1)*W]; t = np.arange(len(df)) / 30
    v1, v2 = df.eye1_valid.values > 0, df.eye2_valid.values > 0
    a.plot(t, np.where(v1, df.eye1_u, np.nan), "-", lw=1.3, label="Eye 1, $u_1$")
    a.plot(t, np.where(v2, df.eye2_u, np.nan), "--", lw=1.3, label="Eye 2, $u_2$")
    a.plot(t, np.where(v1 & v2, df.eye1_u + df.eye2_u, np.nan), "k-", lw=1.8, label="Binocular index $u_1+u_2$")
    for tt in t[~(v1 & v2)]: a.axvspan(tt, tt + 1/30, color="grey", alpha=.2, lw=0)
    a.axhline(0, color="grey", lw=.8); a.set_title(title, fontsize=10); a.set_xlabel("Time (s)"); a.grid(alpha=.3, ls="--")
ax[0].set_ylabel("Signed normalized displacement"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "real_signals.png"), dpi=300); plt.close(fig)
print(json.dumps({n: {c: round(v[c]["f1"], 3) for c in C} for n, v in pooled.items()}, indent=1)); print(pooled["CNN-only"]["confusion"])
