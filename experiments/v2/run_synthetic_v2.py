"""
All synthetic experiments for the revised manuscript (generator v2, signed input with validity masks).

Data: data/synthetic_v2/synthetic_v2.npz (1500 sequences, 500/class; src/generate_synthetic_v2.py, seed 2026).
Split: stratified 70/15/15 (seed 42). Test set never used for training or model selection.
Input to networks: 4 channels [u1, u2, mask1, mask2].
Every network: 5 seeds (0-4), Adam 1e-3, batch 32, <=100 epochs, early stopping on val loss (patience 10).

E1  nominal test performance of CNN-BiLSTM, CNN-only, BiLSTM-only, CNN-Transformer
E2  classical baselines (single frame / sequence statistics / handcrafted temporal; LogReg, SVM, RF) on valid frames
E3  input ablation of the CNN-BiLSTM (retrained): binocular index only; u without masks; u + masks; + differences
E4  frozen-model robustness, ALL models (5 seeds) x 5 fresh test sets per condition (mean +- SD over 25 runs;
    RF: 5 seeds x 5 sets)
E5  quality-gate sensitivity on a heavy-missingness test set
E6  latency (compiled graph, batch 1)
Outputs: results/synthetic_v2.json, results/predictions/, results/figures/, results/models/ (seed-0 models)
"""
import os, sys, json, time, platform
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import generate_synthetic_v2 as G
from dh_measure import quality_gate

OUT = os.path.join(HERE, "results"); [os.makedirs(os.path.join(OUT, d), exist_ok=True) for d in ["predictions", "figures", "models"]]
CLASSES = G.CLASSES; T = G.T; SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS, PATIENCE, BATCH, LR = 100, 10, 32, 1e-3


def net_input(U, M, mode="default"):
    if mode == "default": return np.concatenate([U, M], -1)
    if mode == "no_mask": return U
    if mode == "binocular": return np.stack([U[..., 0] + U[..., 1], M[..., 0] * M[..., 1]], -1)
    if mode == "diffs":
        d1 = np.diff(U, axis=1, prepend=U[:, :1]) * M; d2 = np.diff(d1, axis=1, prepend=d1[:, :1]) * M
        return np.concatenate([U, M, d1, d2], -1)


# ---------------------------------------------------------------- models
def cnn_bilstm(c):
    return models.Sequential([layers.Input((T, c)), layers.Conv1D(64, 5, activation="relu"), layers.MaxPooling1D(2),
                              layers.Dropout(0.3), layers.Bidirectional(layers.LSTM(64)),
                              layers.Dense(32, activation="relu", name="latent"), layers.Dense(3, activation="softmax")])

def cnn_only(c):
    return models.Sequential([layers.Input((T, c)), layers.Conv1D(64, 5, activation="relu"), layers.MaxPooling1D(2),
                              layers.Dropout(0.3), layers.Conv1D(64, 5, activation="relu"), layers.GlobalAveragePooling1D(),
                              layers.Dense(32, activation="relu"), layers.Dense(3, activation="softmax")])

def bilstm_only(c):
    return models.Sequential([layers.Input((T, c)), layers.Bidirectional(layers.LSTM(64)),
                              layers.Dense(32, activation="relu"), layers.Dense(3, activation="softmax")])

class PosEmb(layers.Layer):
    def __init__(self, n, d, **kw): super().__init__(**kw); self.emb = layers.Embedding(n, d); self.n = n
    def call(self, x): return x + self.emb(tf.range(self.n))

def cnn_transformer(c, d=64):
    i = layers.Input((T, c)); x = layers.Conv1D(d, 5, activation="relu")(i); x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x); x = PosEmb(x.shape[1], d)(x)
    x = layers.LayerNormalization()(x + layers.MultiHeadAttention(4, 16, dropout=0.1)(x, x))
    x = layers.LayerNormalization()(x + layers.Dense(d)(layers.Dense(128, activation="relu")(x)))
    x = layers.Dense(32, activation="relu")(layers.GlobalAveragePooling1D()(x))
    return models.Model(i, layers.Dense(3, activation="softmax")(x))

ARCH = {"CNN-BiLSTM": cnn_bilstm, "CNN-only": cnn_only, "BiLSTM-only": bilstm_only, "CNN-Transformer": cnn_transformer}


def train(builder, Xtr, ytr, Xva, yva, seed):
    tf.keras.utils.set_random_seed(seed); m = builder(Xtr.shape[-1])
    m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=MAX_EPOCHS, batch_size=BATCH, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)])
    return m, len(h.history["loss"])


def met(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "weighted_f1": f1_score(y, p, average="weighted"),
            "per_class": {c: {"precision": pr[i], "recall": rc[i], "f1": f1[i], "support": int(sup[i])} for i, c in enumerate(CLASSES)},
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}


ms = lambda v: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]


# ---------------------------------------------------------------- handcrafted features (valid frames only)
def _nan(U, M): X = U.copy(); X[M == 0] = np.nan; return X

def chans(U, M):
    X = _nan(U, M); A = X[..., 0] + X[..., 1]
    return [X[..., 0], X[..., 1], A]

def f_single(U, M):
    j = (M[..., 0] * M[..., 1]) > 0; out = []
    for i in range(len(U)):
        v = np.flatnonzero(j[i]); k = v[np.argmin(np.abs(v - T // 2))] if v.size else T // 2
        out.append([U[i, k, 0], U[i, k, 1], U[i, k, 0] + U[i, k, 1]])
    return np.array(out)

def f_stats(U, M):
    return np.nan_to_num(np.stack(sum([[np.nanmean(c, 1), np.nanstd(c, 1)] for c in chans(U, M)], []), 1))

def f_temporal(U, M):
    feats = []
    for c in chans(U, M):
        th = np.array_split(c, 3, axis=1); t = np.arange(T)
        slope = np.array([np.polyfit(t[~np.isnan(r)], r[~np.isnan(r)], 1)[0] if (~np.isnan(r)).sum() > 2 else 0 for r in c])
        dif = np.abs(np.diff(c, axis=1))
        feats += [np.nanmean(c, 1), np.nanstd(c, 1), np.nanmin(c, 1), np.nanmax(c, 1),
                  np.nanmean(th[0], 1), np.nanmean(th[1], 1), np.nanmean(th[2], 1), slope, np.nanmax(dif, 1),
                  np.nanpercentile(c, 90, 1) - np.nanpercentile(c, 10, 1)]
    return np.nan_to_num(np.stack(feats, 1))

FEATS = {"Single frame": f_single, "Sequence mean/SD": f_stats, "Handcrafted temporal": f_temporal}
CLFS = {"LogReg": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=s)),
        "SVM": lambda s: make_pipeline(StandardScaler(), SVC(random_state=s)),
        "RF": lambda s: RandomForestClassifier(300, random_state=s)}


def latency(m, c):
    f = tf.function(lambda x: m(x, training=False)); x = tf.zeros((1, T, c))
    for _ in range(20): f(x)
    t0 = time.perf_counter()
    for _ in range(300): f(x)
    return (time.perf_counter() - t0) / 300 * 1000


def main():
    d = np.load(os.path.join(ROOT, "data", "synthetic_v2", "synthetic_v2.npz")); U, M, y = d["u"], d["mask"], d["y"]
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
    np.save(os.path.join(OUT, "predictions", "test_idx.npy"), te)
    R = {"data": {"n": len(y), "train": len(tr), "val": len(va), "test": len(te),
                  "test_per_class": np.bincount(y[te]).tolist(),
                  "missing_frame_pct": float(100 * (1 - M.mean()))},
         "env": {"python": platform.python_version(), "tensorflow": tf.__version__, "machine": platform.platform()}}
    X = net_input(U, M)

    # E1
    trained = {k: [] for k in ARCH}; R["E1"] = {}
    for name, b in ARCH.items():
        runs = []
        for s in SEEDS:
            m, ep = train(b, X[tr], y[tr], X[va], y[va], s); p = m.predict(X[te], verbose=0)
            np.save(os.path.join(OUT, "predictions", f"{name}_seed{s}.npy"), p)
            r = met(y[te], p.argmax(1)); r["epochs"] = ep; runs.append(r); trained[name].append(m)
            print("E1", name, s, round(r["weighted_f1"], 4), ep, flush=True)
        trained[name][0].save(os.path.join(OUT, "models", f"{name}_seed0.keras"))
        R["E1"][name] = {"params": int(trained[name][0].count_params()), "latency_ms": latency(trained[name][0], X.shape[-1]),
                         **{k: ms([r[k] for r in runs]) for k in ["accuracy", "macro_f1", "weighted_f1", "epochs"]},
                         "confusion_pooled": np.sum([r["confusion"] for r in runs], 0).tolist()}
    P = np.concatenate([np.load(os.path.join(OUT, "predictions", f"CNN-BiLSTM_seed{s}.npy")).argmax(1) for s in SEEDS])
    R["E1"]["CNN-BiLSTM"]["pooled_per_class"] = met(np.tile(y[te], 5), P)["per_class"]

    # E2
    R["E2"] = {}; rf_models = []
    for fn, f in FEATS.items():
        Ftr, Fte = f(U[tr], M[tr]), f(U[te], M[te])
        for cn, c in CLFS.items():
            runs = []
            for s in SEEDS:
                clf = c(s).fit(Ftr, y[tr]); runs.append(met(y[te], clf.predict(Fte)))
                if fn == "Handcrafted temporal" and cn == "RF": rf_models.append(clf)
            R["E2"][f"{fn} + {cn}"] = {k: ms([r[k] for r in runs]) for k in ["accuracy", "macro_f1", "weighted_f1"]}
            R["E2"][f"{fn} + {cn}"]["recall_intermittent_seed0"] = runs[0]["per_class"]["Intermittent"]["recall"]
            print("E2", fn, cn, R["E2"][f"{fn} + {cn}"]["weighted_f1"], flush=True)

    # E3
    R["E3"] = {}
    for mode, lab in [("binocular", "Binocular index u1+u2 + joint mask"), ("no_mask", "u1, u2 without masks"),
                      ("default", "u1, u2 + masks (default)"), ("diffs", "u1, u2 + masks + 1st/2nd differences")]:
        Xm = net_input(U, M, mode); runs = []
        for s in SEEDS:
            if mode == "default":
                p = np.load(os.path.join(OUT, "predictions", f"CNN-BiLSTM_seed{s}.npy")); runs.append(met(y[te], p.argmax(1))); continue
            m, _ = train(cnn_bilstm, Xm[tr], y[tr], Xm[va], y[va], s); runs.append(met(y[te], m.predict(Xm[te], verbose=0).argmax(1)))
        R["E3"][lab] = {"channels": int(Xm.shape[-1]), **{k: ms([r[k] for r in runs]) for k in ["accuracy", "weighted_f1"]}}
        print("E3", lab, R["E3"][lab]["weighted_f1"], flush=True)

    # E4
    conds = {"Nominal (fresh data)": {}, "Jitter sigma 0.03": dict(sigma=(0.03, 0.03)),
             "Jitter sigma 0.05": dict(sigma=(0.05, 0.05)), "Jitter sigma 0.08": dict(sigma=(0.08, 0.08)),
             "Missing frames (dropout 0.3)": dict(drop=(0.3, 0.3)),
             "Large gaze shifts (+-0.3)": dict(gaze=0.3, n_gaze_shifts=(1, 4)),
             "Low amplitude (0.04-0.08)": dict(amp_int=(0.04, 0.08), amp_const=(0.04, 0.08))}
    R["E4"] = {}
    for cn, kw in conds.items():
        sets = [G.generate_set(5000 + r, 100, **kw) for r in range(5)]; R["E4"][cn] = {}
        for name in ARCH:
            v = [f1_score(yy, m.predict(net_input(uu, mm), verbose=0).argmax(1), average="weighted")
                 for (uu, mm, yy, _) in sets for m in trained[name]]
            R["E4"][cn][name] = ms(v)
        v = [f1_score(yy, rf.predict(f_temporal(uu, mm)), average="weighted") for (uu, mm, yy, _) in sets for rf in rf_models]
        R["E4"][cn]["Handcrafted temporal + RF"] = ms(v)
        print("E4", cn, {k: round(x[0], 3) for k, x in R["E4"][cn].items()}, flush=True)

    # E5 quality gate sensitivity: heavy missingness test set, seed-0 CNN-BiLSTM
    Uh, Mh, yh, _ = G.generate_set(7000, 200, drop=(0.0, 0.6), burst_p=0.8, blinks=(0, 6))
    ph = trained["CNN-BiLSTM"][0].predict(net_input(Uh, Mh), verbose=0).argmax(1); R["E5"] = []
    for q in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for g in [10, 15, 30, 150]:
            acc = np.array([quality_gate(Mh[i, :, 0], Mh[i, :, 1], q, g)[0] for i in range(len(yh))])
            R["E5"].append({"min_joint": q, "max_gap": g, "accepted_pct": float(100 * acc.mean()),
                            "weighted_f1_accepted": float(f1_score(yh[acc], ph[acc], average="weighted")) if acc.any() else None,
                            "weighted_f1_rejected": float(f1_score(yh[~acc], ph[~acc], average="weighted")) if (~acc).any() else None})
    print("E5 done", flush=True)

    # figures: pooled confusion, t-SNE (seed 0), example sequences
    cm = np.array(R["E1"]["CNN-BiLSTM"]["confusion_pooled"]); fig, ax = plt.subplots(figsize=(4.4, 3.9)); ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "confusion_matrix_pooled.png"), dpi=300); plt.close(fig)
    m0 = trained["CNN-BiLSTM"][0]
    z = tf.keras.Model(m0.inputs, m0.get_layer("latent").output).predict(X[te], verbose=0)
    z2 = TSNE(2, perplexity=30, max_iter=1000, random_state=42, init="pca").fit_transform(z)
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); col = ["#2ca02c", "#d62728", "#1f77b4"]
    for k, c in enumerate(CLASSES): ax.scatter(*z2[y[te] == k].T, c=col[k], s=25, alpha=.8, edgecolors="w", label=c)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(fontsize=8); ax.grid(alpha=.3, ls="--")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "tsne_test.png"), dpi=300); plt.close(fig)
    rng = np.random.default_rng(3); fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True); t = np.arange(T) / 30
    for k, c in enumerate(CLASSES):
        i = rng.choice(np.flatnonzero(y == k));
        for e, ls in [(0, "-"), (1, "--")]:
            ax[k].plot(t, np.where(M[i, :, e] > 0, U[i, :, e], np.nan), ls, lw=1.4, label=f"Eye {e+1}")
        for tt in t[(M[i, :, 0] * M[i, :, 1]) == 0]: ax[k].axvspan(tt, tt + 1 / 30, color="grey", alpha=.2, lw=0)
        ax[k].set_title(f"({'ABC'[k]}) {c}"); ax[k].set_xlabel("Time (s)"); ax[k].grid(alpha=.3, ls="--")
    ax[0].set_ylabel("Signed normalized displacement u"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "figure2_trajectories.png"), dpi=300); plt.close(fig)
    R["tsne"] = {"n": int(len(te)), "perplexity": 30, "seed": 42}

    json.dump(R, open(os.path.join(OUT, "synthetic_v2.json"), "w"), indent=2, default=float); print("DONE", flush=True)


if __name__ == "__main__":
    main()
