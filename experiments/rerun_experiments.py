"""
Reproducible re-run of all synthetic experiments for the Dynamic Hirschberg manuscript.

Written 2026-11-25 to replace the earlier ad-hoc scripts (train_model.py,
experiment_ablation.py, experiment_noise_robustness*.py, Plot_Figure_5_*.py),
which used the test set as validation data, evaluated "ablations" without
retraining, and plotted figures from different datasets than the tables.

What this script does
---------------------
* Data: the 600 sequences in ../data/synthetic_dataset_robust (200 per class), produced by
  src/generate_synthetic_dataset.py. Input = (150 frames, 2 channels: L_ratio, R_ratio).
* Fixed stratified split 70/15/15 (seed 42). Test set never used for training,
  early stopping or model selection.
* Every neural model is trained with 5 seeds (0-4); mean +- SD reported.
* E1  Main CNN-BiLSTM (architecture identical to train_model.py).
* E2  Static vs temporal baselines + architecture comparison (CNN-only, BiLSTM-only,
      CNN-Transformer), all on the same test set.
* E4  Input-channel ablation, each configuration RETRAINED.
* E5  Frozen-model robustness on freshly generated test sets (same generator code,
      new RNG seed): jitter sweep, heavy blinks, low-amplitude (out-of-range) deviations.
* Timing: per-sequence inference latency (batch 1) on this machine.
* Figures: confusion matrix and t-SNE generated from the SAME saved test predictions.
All raw predictions are saved to predictions/.
"""
import json, os, platform, time
import numpy as np, pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "synthetic_dataset_robust")
OUT = HERE
os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

CLASSES = ["Healthy", "Intermittent", "Constant"]   # label index order used everywhere
T = 150
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 42
MAX_EPOCHS, PATIENCE, BATCH, LR = 100, 10, 32, 1e-3


# ----------------------------------------------------------------- data
def load_dataset():
    X, y, names = [], [], []
    for k, c in enumerate(CLASSES):
        d = os.path.join(DATA, c)
        for f in sorted(os.listdir(d)):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(d, f))
                X.append(df[["L_ratio", "R_ratio"]].values.astype("float32"))
                y.append(k); names.append(f)
    return np.stack(X), np.array(y), np.array(names)


# Exact re-implementation of synthetic_data_gen_w_blink_head_turn.py, with overrides
def gen_signal(rng, cond, noise=0.005, max_blinks=2, inter_amp=(0.10, 0.20),
               const_amp=(0.12, 0.25)):
    s = np.zeros(T) + rng.uniform(-0.02, 0.02)                 # angle kappa
    if cond == "Constant":
        s += rng.choice([1, -1]) * rng.uniform(*const_amp)
    elif cond == "Intermittent":
        start = rng.integers(30, 100); dur = rng.integers(30, 80)
        amp = rng.uniform(*inter_amp)
        for t in range(T):
            if start <= t < start + dur:
                s[t] += amp * (t - start) / dur
            elif t >= start + dur:
                s[t] += amp
    tt = np.linspace(0, 2 * np.pi, T)                          # slow head drift
    s += np.sin(tt + rng.uniform(0, 2 * np.pi)) * rng.uniform(0.001, 0.005)
    s += rng.normal(0, noise, T)                                # jitter
    for _ in range(rng.integers(0, max_blinks + 1)):            # blinks -> 0.0
        d = rng.integers(3, 10); st = rng.integers(0, T - d)
        s[st:st + d] = 0.0
    return s


def gen_set(seed, n_per_class=100, **kw):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(np.stack([gen_signal(rng, c, **kw), gen_signal(rng, "Healthy", **kw)], -1))
            y.append(k)
    return np.array(X, dtype="float32"), np.array(y)


def add_diffs(X):
    d1 = np.diff(X, axis=1, prepend=X[:, :1]); d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    return np.concatenate([X, d1, d2], -1)


# ----------------------------------------------------------------- models
def cnn_bilstm(n_ch):           # identical to train_model.py
    return models.Sequential([
        layers.Input((T, n_ch)),
        layers.Conv1D(64, 5, activation="relu"), layers.MaxPooling1D(2), layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(32, activation="relu", name="latent"),
        layers.Dense(3, activation="softmax")])


def cnn_only(n_ch):
    return models.Sequential([
        layers.Input((T, n_ch)),
        layers.Conv1D(64, 5, activation="relu"), layers.MaxPooling1D(2), layers.Dropout(0.3),
        layers.Conv1D(64, 5, activation="relu"), layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"), layers.Dense(3, activation="softmax")])


def bilstm_only(n_ch):
    return models.Sequential([
        layers.Input((T, n_ch)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(32, activation="relu"), layers.Dense(3, activation="softmax")])


class PosEmb(layers.Layer):
    def __init__(self, n, d, **kw):
        super().__init__(**kw); self.emb = layers.Embedding(n, d); self.n = n
    def call(self, x):
        return x + self.emb(tf.range(self.n))


def cnn_transformer(n_ch, d=64, heads=4, key_dim=16, ff=128):
    inp = layers.Input((T, n_ch))
    x = layers.Conv1D(d, 5, activation="relu")(inp); x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = PosEmb(x.shape[1], d)(x)
    a = layers.MultiHeadAttention(heads, key_dim, dropout=0.1)(x, x)
    x = layers.LayerNormalization()(x + a)
    f = layers.Dense(ff, activation="relu")(x); f = layers.Dense(d)(f)
    x = layers.LayerNormalization()(x + f)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    return models.Model(inp, layers.Dense(3, activation="softmax")(x))


ARCH = {"CNN-BiLSTM": cnn_bilstm, "CNN-only": cnn_only, "BiLSTM-only": bilstm_only,
        "CNN-Transformer": cnn_transformer}


def train_nn(builder, Xtr, ytr, Xva, yva, seed):
    tf.keras.utils.set_random_seed(seed)
    m = builder(Xtr.shape[-1])
    m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=MAX_EPOCHS, batch_size=BATCH,
              callbacks=[es], verbose=0)
    return m, len(h.history["loss"]), int(np.argmin(h.history["val_loss"])) + 1


def metrics(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    return {"accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "weighted_f1": f1_score(y, p, average="weighted"),
            "per_class": {c: {"precision": pr[i], "recall": rc[i], "f1": f1[i], "support": int(sup[i])}
                          for i, c in enumerate(CLASSES)},
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}


def summarize(runs, key):
    v = np.array([r[key] for r in runs]); return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ----------------------------------------------------------------- handcrafted features
def feat_single_frame(X, idx=T // 2):
    f = X[:, idx, :]; return np.c_[f, f[:, 0] - f[:, 1]]


def feat_stats(X):
    d = X[:, :, 0] - X[:, :, 1]
    return np.c_[X.mean(1), X.std(1), d.mean(1), d.std(1)]


def feat_temporal(X):
    out = []
    for ch in [X[:, :, 0], X[:, :, 1], X[:, :, 0] - X[:, :, 1]]:
        thirds = np.array_split(ch, 3, axis=1)
        slope = np.polyfit(np.arange(T), ch.T, 1)[0]
        out += [ch.mean(1), ch.std(1), ch.min(1), ch.max(1),
                thirds[0].mean(1), thirds[1].mean(1), thirds[2].mean(1), slope,
                np.abs(np.diff(ch, axis=1)).max(1)]
    return np.stack(out, 1)


# ================================================================== main
def main():
    X, y, names = load_dataset()
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SPLIT_SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SPLIT_SEED)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    # duplicate check
    flat = X.reshape(len(X), -1).round(6)
    n_dup = len(flat) - len(np.unique(flat, axis=0))
    split = {"n_total": len(y), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
             "test_per_class": {c: int((y[te] == k).sum()) for k, c in enumerate(CLASSES)},
             "duplicate_sequences": int(n_dup)}
    print("split", split)
    pd.DataFrame({"file": names[te], "label": y[te]}).to_csv(
        os.path.join(OUT, "predictions", "test_files.csv"), index=False)

    R = {"split": split, "env": {"python": platform.python_version(), "tensorflow": tf.__version__,
         "machine": platform.platform(), "processor": platform.processor()},
         "training": {"optimizer": "Adam", "lr": LR, "batch": BATCH, "max_epochs": MAX_EPOCHS,
                      "early_stopping": f"val_loss, patience {PATIENCE}, restore best"},
         "seeds": SEEDS}

    # ---------- E1 + E2 (architectures) ----------
    arch_res, frozen = {}, {}
    for name, b in ARCH.items():
        runs = []
        for s in SEEDS:
            m, ep, best = train_nn(b, X[tr], y[tr], X[va], y[va], s)
            p = m.predict(X[te], verbose=0); yp = p.argmax(1)
            met = metrics(y[te], yp); met.update(epochs_run=ep, best_epoch=best,
                                                   val_acc=accuracy_score(y[va], m.predict(X[va], verbose=0).argmax(1)))
            runs.append(met)
            np.save(os.path.join(OUT, "predictions", f"{name}_seed{s}_probs.npy"), p)
            if s == SEEDS[0]:
                frozen[name] = m
            print(name, s, round(met["weighted_f1"], 4), "epochs", ep)
        # latency, batch 1, after warm-up
        m = frozen[name]; x1 = X[te][:1]
        for _ in range(10): m(x1, training=False)
        t0 = time.perf_counter()
        for _ in range(200): m(x1, training=False)
        lat = (time.perf_counter() - t0) / 200 * 1000
        arch_res[name] = {"params": int(m.count_params()), "latency_ms": lat,
                          "accuracy": summarize(runs, "accuracy"), "macro_f1": summarize(runs, "macro_f1"),
                          "weighted_f1": summarize(runs, "weighted_f1"), "val_acc": summarize(runs, "val_acc"),
                          "epochs_run": summarize(runs, "epochs_run"), "best_epoch": summarize(runs, "best_epoch"),
                          "runs": runs}
    R["architectures"] = arch_res

    # ---------- E2 classical baselines ----------
    base = {}
    feats = {"Single frame (t=75)": feat_single_frame, "Sequence mean/SD": feat_stats,
             "Handcrafted temporal": feat_temporal}
    clfs = {"LogReg": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s)),
            "SVM": lambda s: make_pipeline(StandardScaler(), SVC(random_state=s)),
            "RF": lambda s: RandomForestClassifier(300, random_state=s)}
    for fn, f in feats.items():
        Ftr, Fte = f(X[tr]), f(X[te])
        for cn, c in clfs.items():
            runs = []
            for s in SEEDS:
                yp = c(s).fit(Ftr, y[tr]).predict(Fte); runs.append(metrics(y[te], yp))
            base[f"{fn} + {cn}"] = {"accuracy": summarize(runs, "accuracy"),
                                    "macro_f1": summarize(runs, "macro_f1"),
                                    "weighted_f1": summarize(runs, "weighted_f1"),
                                    "per_class_recall_seed0": {k: v["recall"] for k, v in runs[0]["per_class"].items()}}
            print(fn, cn, base[f"{fn} + {cn}"]["weighted_f1"])
    R["baselines"] = base

    # ---------- E4 input-channel ablation (retrained) ----------
    abl = {}
    configs = {"Affected eye only (L)": lambda Z: Z[:, :, :1],
               "Both eyes (L, R) [default]": lambda Z: Z,
               "Both eyes + 1st/2nd differences": add_diffs}
    for cn, fx in configs.items():
        runs = []
        for s in SEEDS:
            m, ep, _ = train_nn(cnn_bilstm, fx(X[tr]), y[tr], fx(X[va]), y[va], s)
            runs.append(metrics(y[te], m.predict(fx(X[te]), verbose=0).argmax(1)))
        abl[cn] = {"channels": int(fx(X[:1]).shape[-1]), "accuracy": summarize(runs, "accuracy"),
                   "weighted_f1": summarize(runs, "weighted_f1"), "macro_f1": summarize(runs, "macro_f1")}
        print("ablation", cn, abl[cn]["weighted_f1"])
    R["ablation"] = abl

    # ---------- E5 frozen-model robustness on fresh data ----------
    rob = {}
    conds = {f"jitter sigma={s}": dict(noise=s) for s in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]}
    conds["heavy blinks (0-6 per clip)"] = dict(max_blinks=6)
    conds["low amplitude (Int 0.03-0.08, Const 0.04-0.10)"] = dict(inter_amp=(0.03, 0.08), const_amp=(0.04, 0.10))
    for cn, kw in conds.items():
        rob[cn] = {}
        for an in ["CNN-BiLSTM", "CNN-Transformer"]:
            vals = []
            for rep in range(5):     # 5 fresh test sets per condition, frozen seed-0 model
                Xo, yo = gen_set(1000 + rep, 100, **kw)
                vals.append(f1_score(yo, frozen[an].predict(Xo, verbose=0).argmax(1), average="weighted"))
            rob[cn][an] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
        Xo, yo = gen_set(1000, 100, **kw)
        Ftr = feat_temporal(X[tr])
        rf = RandomForestClassifier(300, random_state=0).fit(Ftr, y[tr])
        rob[cn]["Handcrafted temporal + RF"] = (float(f1_score(yo, rf.predict(feat_temporal(Xo)), average="weighted")), 0.0)
        print("robust", cn, rob[cn])
    R["robustness"] = rob

    # ---------- figures from saved seed-0 CNN-BiLSTM predictions ----------
    m = frozen["CNN-BiLSTM"]
    p0 = np.load(os.path.join(OUT, "predictions", "CNN-BiLSTM_seed0_probs.npy")).argmax(1)
    cm = confusion_matrix(y[te], p0, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "confusion_matrix.png"), dpi=300); plt.close(fig)

    lat = tf.keras.Model(m.inputs, m.get_layer("latent").output).predict(X[te], verbose=0)
    z = TSNE(2, perplexity=20, max_iter=1000, random_state=42, init="pca").fit_transform(lat)
    sign = np.sign(X[te][:, 30:, 0].mean(1))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cols = {"Healthy": "#2ca02c", "Intermittent": "#d62728", "Constant": "#1f77b4"}
    for k, c in enumerate(CLASSES):
        if c == "Constant":
            for sg, mk, lab in [(1, "o", "Constant (positive offset)"), (-1, "^", "Constant (negative offset)")]:
                sel = (y[te] == k) & (sign == sg)
                ax.scatter(*z[sel].T, c=cols[c], marker=mk, s=40, alpha=.8, edgecolors="w", label=lab)
        else:
            sel = y[te] == k
            ax.scatter(*z[sel].T, c=cols[c], s=40, alpha=.8, edgecolors="w", label=c)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(fontsize=8); ax.grid(alpha=.3, ls="--")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "tsne_test.png"), dpi=300); plt.close(fig)
    R["tsne"] = {"n_points": int(len(te)), "perplexity": 20, "seed": 42, "layer": "Dense(32) 'latent'",
                 "constant_positive": int(((y[te] == 2) & (sign > 0)).sum()),
                 "constant_negative": int(((y[te] == 2) & (sign < 0)).sum())}

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    print("DONE")


if __name__ == "__main__":
    main()
