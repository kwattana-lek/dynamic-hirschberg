"""
Selective classification with the quality gate (revised protocol, replaces E5 of run_synthetic_v2.py).

Models: CNN-only (recommended) and CNN-BiLSTM, all five seeds, retrained with the identical settings and
split of run_synthetic_v2.py and saved to results/models/. Predictions on the original test set are compared
with the saved predictions to confirm that retraining reproduces the reported models.

Test set ("variable quality"): 600 fresh sequences (200 per class; generator seed 7100) whose per-eye dropout
rate is drawn from U(0, 0.5), with dropout bursts (p = 0.5) and 0-4 blinks, so recording quality ranges from
good to poor. For each gate setting we report: accepted and rejected counts, coverage, accepted class counts,
and weighted F1 of accepted and of rejected windows as mean +- SD over the five seeds (the gate itself does
not depend on the model, so counts are identical across seeds). Metrics are reported as undefined when no
window is accepted, and flagged when any class has fewer than 10 accepted windows.
Output: results/gate_eval_v2.json
"""
import os, sys, json, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import run_synthetic_v2 as S
import generate_synthetic_v2 as G
from dh_measure import quality_gate

R = os.path.join(HERE, "results")
d = np.load(os.path.join(HERE, "..", "..", "data", "synthetic_v2", "synthetic_v2.npz")); U, M, y = d["u"], d["mask"], d["y"]
idx = np.arange(len(y))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
X = S.net_input(U, M)

models, repro = {}, {}
for name in ["CNN-only", "CNN-BiLSTM"]:
    models[name] = []; agree = []
    for s in S.SEEDS:
        m, _ = S.train(S.ARCH[name], X[tr], y[tr], X[va], y[va], s)
        m.save(os.path.join(R, "models", f"{name}_seed{s}.keras")); models[name].append(m)
        old = np.load(os.path.join(R, "predictions", f"{name}_seed{s}.npy")).argmax(1)
        agree.append(float((m.predict(X[te], verbose=0).argmax(1) == old).mean()))
        print(name, s, "agreement with reported predictions", agree[-1], flush=True)
    repro[name] = agree

Uq, Mq, yq, _ = G.generate_set(7100, 200, drop=(0.0, 0.5), burst_p=0.5, blinks=(0, 4))
Xq = S.net_input(Uq, Mq)
preds = {n: [m.predict(Xq, verbose=0).argmax(1) for m in ms] for n, ms in models.items()}
joint = (Mq[..., 0] * Mq[..., 1]).mean(1)
settings = [(None, None), (0.5, 15), (0.6, 15), (0.7, 15), (0.7, 30), (0.8, 15), (0.9, 15)]
rows = []
for q, g in settings:
    acc = np.ones(len(yq), bool) if q is None else np.array([quality_gate(Mq[i, :, 0], Mq[i, :, 1], q, g)[0] for i in range(len(yq))])
    row = {"min_joint": q, "max_gap": g, "n_total": int(len(yq)), "n_accepted": int(acc.sum()), "n_rejected": int((~acc).sum()),
           "coverage_pct": float(100 * acc.mean()),
           "accepted_per_class": {c: int(((yq == k) & acc).sum()) for k, c in enumerate(G.CLASSES)}}
    row["small_class_flag"] = bool(acc.any() and min(row["accepted_per_class"].values()) < 10)
    for n in preds:
        fa = [f1_score(yq[acc], p[acc], average="weighted") for p in preds[n]] if acc.any() else None
        fr = [f1_score(yq[~acc], p[~acc], average="weighted") for p in preds[n]] if (~acc).any() else None
        row[n] = {"f1_accepted": S.ms(fa) if fa else None, "f1_rejected": S.ms(fr) if fr else None}
    rows.append(row); print(row, flush=True)
out = {"test_set": {"n": int(len(yq)), "seed": 7100, "drop": [0.0, 0.5], "burst_p": 0.5, "blinks": [0, 4],
                    "joint_valid_pct_median": float(100 * np.median(joint)),
                    "joint_valid_pct_quartiles": [float(100 * np.percentile(joint, 25)), float(100 * np.percentile(joint, 75))]},
       "reproduction_agreement": repro, "rows": rows}
json.dump(out, open(os.path.join(R, "gate_eval_v2.json"), "w"), indent=2, default=float)
print("DONE")
