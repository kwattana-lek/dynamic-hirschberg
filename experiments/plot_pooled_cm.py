"""Pooled confusion matrix and per-class metrics of the CNN-BiLSTM over the 5 seeds (90 test sequences x 5)."""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
H = os.path.dirname(os.path.abspath(__file__)); C = ["Healthy", "Intermittent", "Constant"]
y = pd.read_csv(os.path.join(H, "predictions", "test_files.csv"))["label"].values
Y, P = [], []
for s in range(5):
    P.append(np.load(os.path.join(H, "predictions", f"CNN-BiLSTM_seed{s}_probs.npy")).argmax(1)); Y.append(y)
Y, P = np.concatenate(Y), np.concatenate(P)
cm = confusion_matrix(Y, P, labels=[0, 1, 2]); pr, rc, f1, sup = precision_recall_fscore_support(Y, P)
out = {"confusion_pooled": cm.tolist(), **{c: dict(precision=pr[i], recall=rc[i], f1=f1[i], support=int(sup[i])) for i, c in enumerate(C)}}
json.dump(out, open(os.path.join(H, "pooled_cnn_bilstm.json"), "w"), indent=2, default=float); print(out)
fig, ax = plt.subplots(figsize=(4.4, 3.9)); ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12)
ax.set_xticks(range(3), C); ax.set_yticks(range(3), C); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
fig.tight_layout(); fig.savefig(os.path.join(H, "figures", "confusion_matrix_pooled.png"), dpi=300)
