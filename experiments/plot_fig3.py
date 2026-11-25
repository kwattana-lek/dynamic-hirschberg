"""Figure 3: CNN-BiLSTM architecture exactly as in train_model.py / rerun_experiments.py."""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
blocks = [("Input\n(150 × 2)\nL, R ratio", "#f2f2f2", "—"),
          ("Conv1D\n64 filters, k = 5\nReLU → (146 × 64)", "#cfe2f3", "704"),
          ("MaxPooling1D\npool 2\n→ (73 × 64)", "#d9ead3", "0"),
          ("Dropout\n0.3", "#d9ead3", "0"),
          ("Bidirectional LSTM\n64 units/direction\nlast state → 128", "#fff2cc", "66,048"),
          ("Dense 32, ReLU\nlatent z", "#ead1dc", "4,128"),
          ("Dense 3\nsoftmax", "#f4cccc", "99")]
fig, ax = plt.subplots(figsize=(13, 2.9)); ax.set_xlim(0, 13); ax.set_ylim(0, 2.9); ax.axis("off")
w, gap = 1.6, 0.25
for i, (txt, col, p) in enumerate(blocks):
    x = 0.15 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.6, boxstyle="round,pad=0.02", fc=col, ec="k", lw=1.2))
    ax.text(x + w/2, 1.55, txt, ha="center", va="center", fontsize=9)
    ax.text(x + w/2, 0.45, f"params: {p}", ha="center", va="center", fontsize=8, color="#444")
    if i < len(blocks) - 1:
        ax.annotate("", (x + w + gap, 1.55), (x + w, 1.55), arrowprops=dict(arrowstyle="->", lw=1.3))
ax.text(6.5, 0.08, "Total trainable parameters: 70,979", ha="center", fontsize=9, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "figure3_architecture.png"), dpi=300)
print("ok")
