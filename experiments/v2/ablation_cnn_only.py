"""
Input ablation of the recommended CNN-only network (10-s windows), same data, split, seeds and training settings
as run_synthetic_v2.py (whose E3 ablates the CNN-BiLSTM). The default configuration reuses the E1 predictions.
Output: results/ablation_cnn_only.json
"""
import os, sys, json, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import run_synthetic_v2 as S

d = np.load(os.path.join(HERE, "..", "..", "data", "synthetic_v2", "synthetic_v2.npz")); U, M, y = d["u"], d["mask"], d["y"]
idx = np.arange(len(y))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
out = {}
for mode, lab in [("binocular", "Binocular index u1+u2 + joint mask"), ("no_mask", "u1, u2 without masks"),
                  ("default", "u1, u2 + masks (default)"), ("diffs", "u1, u2 + masks + 1st/2nd differences")]:
    X = S.net_input(U, M, mode); f1, acc = [], []
    for s in S.SEEDS:
        if mode == "default":
            p = np.load(os.path.join(HERE, "results", "predictions", f"CNN-only_seed{s}.npy")).argmax(1)
        else:
            m, _ = S.train(S.cnn_only, X[tr], y[tr], X[va], y[va], s); p = m.predict(X[te], verbose=0).argmax(1)
        f1.append(f1_score(y[te], p, average="weighted")); acc.append(accuracy_score(y[te], p))
    out[lab] = {"channels": int(X.shape[-1]), "accuracy": S.ms(acc), "weighted_f1": S.ms(f1)}
    print(lab, out[lab], flush=True)
json.dump(out, open(os.path.join(HERE, "results", "ablation_cnn_only.json"), "w"), indent=2)
print("DONE")
