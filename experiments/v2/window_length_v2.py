"""
Observation-window length experiment (2.5, 5 and 10 s).

Why a separate generator: in STSG-v2 (src/generate_synthetic_v2.py) every Intermittent window is constructed to
contain at least 20 deviated frames, so the label is always visible inside the window. To study the window length,
labels must instead belong to the *subject*, and the window is a crop of a longer recording in which deviation
episodes occur at random times.

Subject-level generator (same components and parameter ranges as STSG-v2, expressed per 5 s and scaled with length):
  timeline of L = 600 frames (20 s at 30 fps) per subject
  baseline offset (kappa), conjugate gaze (0-3 shifts per 5 s) with slow drift, fixational drift, Gaussian noise
  sigma ~ U(0.005, 0.02), 0-3 blinks per 5 s, per-eye dropout U(0, 0.10), dropout burst p = 0.3 per 5 s per eye
  Healthy: no deviation; Constant: offset U(0.08, 0.25) throughout, +-5% slow modulation
  Intermittent: alternating aligned intervals and trapezoidal episodes (onset 5-30, plateau 15-75, recovery
  5-30 frames; amplitude U(0.08, 0.20)); aligned intervals U(a_min, a_max) s, started at a random phase.
  Scenario "frequent" (training and main test): aligned intervals 1-10 s.
  Scenario "infrequent" (test only): aligned intervals 5-20 s.
For each window length W in {75, 150, 300} frames, one window per subject is cropped at a random position.
A window of an Intermittent subject is "deviation visible" if it contains >= 20 frames with envelope > 0.5
(the criterion used in STSG-v2).

Training: 1500 subjects (500/class, seed 2027), stratified 70/15/15 split (seed 42), CNN-only network (the
recommended model, same architecture, input channels and training settings as run_synthetic_v2.py), 5 seeds per W.
Test: 600 fresh subjects (200/class) per scenario (seeds 2028 frequent, 2029 infrequent).
Reported per W: weighted F1, per-class recall, fraction of Intermittent windows with visible deviation,
Intermittent recall in visible and non-visible windows (mean +- SD over seeds), classifier latency.
Output: results/window_length_v2.json
"""
import os, sys, json, time, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import run_synthetic_v2 as S

L, FPS = 600, 30
WINDOWS = [75, 150, 300]
CLASSES = ["Healthy", "Intermittent", "Constant"]
SCEN = {"frequent": (1.0, 10.0), "infrequent": (5.0, 20.0)}


def trapezoid(n, start, up, plat, down):
    t = np.arange(n, dtype=float); a = np.zeros(n)
    a = np.where((t >= start) & (t < start + up), (t - start) / up, a)
    a = np.where((t >= start + up) & (t < start + up + plat), 1.0, a)
    e = start + up + plat
    return np.where((t >= e) & (t < e + down), 1 - (t - e) / down, a)


def per5s(rng, lo, hi, n):          # count drawn per 5-s block, summed over the timeline
    return int(sum(rng.integers(lo, hi + 1) for _ in range(max(1, round(n / 150)))))


def subject(rng, cls, gap_s):
    u = np.zeros((L, 2)); kappa = rng.uniform(-0.01, 0.05, 2)
    lev = rng.uniform(-0.15, 0.15); c = np.full(L, lev)
    for t0 in np.sort(rng.integers(5, L - 5, per5s(rng, 0, 3, L))):
        new = rng.uniform(-0.15, 0.15); c = c + (new - lev) * np.clip((np.arange(L) - t0) / 3.0, 0, 1); lev = new
    c = c + np.sin(np.linspace(0, 2 * np.pi * L / 150, L) + rng.uniform(0, 2 * np.pi)) * rng.uniform(0.001, 0.01)
    u[:, 0] = kappa[0] + c; u[:, 1] = kappa[1] - c
    for e in range(2): u[:, e] += np.cumsum(rng.normal(0, 0.0007, L))
    aff = int(rng.integers(0, 2)); sign = float(rng.choice([1.0, -1.0])); env = np.zeros(L)
    if cls == "Constant":
        A = rng.uniform(0.08, 0.25); env[:] = 1.0
        u[:, aff] += sign * A * (1 + 0.05 * np.sin(np.linspace(0, 2 * np.pi * L / 150, L) + rng.uniform(0, 6.3)))
    elif cls == "Intermittent":
        A = rng.uniform(0.08, 0.20); t = -int(rng.uniform(0, gap_s[1] * FPS + 165))   # random phase
        while t < L:
            t += int(rng.uniform(*gap_s) * FPS)
            up, plat, down = rng.integers(5, 31), rng.integers(15, 76), rng.integers(5, 31)
            env = np.maximum(env, trapezoid(L, t, up, plat, down)); t += up + plat + down
        u[:, aff] += sign * A * env
    u += rng.normal(0, rng.uniform(0.005, 0.02), (L, 2))
    m = np.ones((L, 2))
    for _ in range(per5s(rng, 0, 3, L)):
        d = rng.integers(3, 11); s = rng.integers(0, L - d); m[s:s + d, :] = 0
    for e in range(2):
        m[rng.random(L) < rng.uniform(0, 0.10), e] = 0
        for _ in range(round(L / 150)):
            if rng.random() < 0.3:
                d = rng.integers(5, 21); s = rng.integers(0, L - d); m[s:s + d, e] = 0
    u[m == 0] = 0.0
    return u.astype("float32"), m.astype("float32"), env.astype("float32")


def make_set(seed, n_per_class, gap_s):
    rng = np.random.default_rng(seed); U, M, E, Y = [], [], [], []
    for k, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            u, m, e = subject(rng, c, gap_s); U.append(u); M.append(m); E.append(e); Y.append(k)
    return np.stack(U), np.stack(M), np.stack(E), np.array(Y)


def crop(U, M, E, W, seed):
    s = np.random.default_rng(seed).integers(0, L - W + 1, len(U))
    take = lambda A: np.stack([a[i:i + W] for a, i in zip(A, s)])
    return take(U), take(M), take(E)


def cnn_only(W, c):
    return models.Sequential([layers.Input((W, c)), layers.Conv1D(64, 5, activation="relu"), layers.MaxPooling1D(2),
                              layers.Dropout(0.3), layers.Conv1D(64, 5, activation="relu"), layers.GlobalAveragePooling1D(),
                              layers.Dense(32, activation="relu"), layers.Dense(3, activation="softmax")])


def latency(m, W):
    f = tf.function(lambda x: m(x, training=False)); x = tf.zeros((1, W, 4))
    for _ in range(20): f(x)
    t0 = time.perf_counter()
    for _ in range(300): f(x)
    return (time.perf_counter() - t0) / 300 * 1000


def main():
    Ut, Mt, Et, yt = make_set(2027, 500, SCEN["frequent"])
    idx = np.arange(len(yt))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=yt, random_state=42)
    va, _ = train_test_split(tmp, test_size=0.50, stratify=yt[tmp], random_state=42)
    tests = {sc: make_set(seed, 200, g) for (sc, g), seed in zip(SCEN.items(), [2028, 2029])}
    out = {"design": {"timeline_frames": L, "fps": FPS, "windows_frames": WINDOWS, "scenarios_gap_s": SCEN,
                      "train_subjects": len(tr), "val_subjects": len(va), "test_subjects_per_scenario": 600,
                      "visible_criterion": ">=20 frames with envelope > 0.5", "model": "CNN-only", "seeds": S.SEEDS},
           "results": {}}
    for W in WINDOWS:
        Uc, Mc, _ = crop(Ut, Mt, Et, W, 100 + W); X = np.concatenate([Uc, Mc], -1)
        tf_models = []
        for s in S.SEEDS:
            tf.keras.utils.set_random_seed(s); m = cnn_only(W, 4)
            m.compile(optimizer=tf.keras.optimizers.Adam(S.LR), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            m.fit(X[tr], yt[tr], validation_data=(X[va], yt[va]), epochs=S.MAX_EPOCHS, batch_size=S.BATCH, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=S.PATIENCE, restore_best_weights=True)])
            tf_models.append(m)
        res = {"seconds": W / FPS, "latency_ms": latency(tf_models[0], W)}
        for sc, (Uq, Mq, Eq, yq) in tests.items():
            Uw, Mw, Ew = crop(Uq, Mq, Eq, W, 200 + W); Xq = np.concatenate([Uw, Mw], -1)
            vis = (Ew > 0.5).sum(1) >= 20; inter = yq == 1
            P = [m.predict(Xq, verbose=0).argmax(1) for m in tf_models]
            rec = lambda p, k, sel=None: float((p[(yq == k) if sel is None else sel] == k).mean()) if (sel is None or sel.any()) else None
            r = {"weighted_f1": S.ms([f1_score(yq, p, average="weighted") for p in P]),
                 "recall": {c: S.ms([rec(p, k) for p in P]) for k, c in enumerate(CLASSES)},
                 "intermittent_visible_pct": float(100 * vis[inter].mean()),
                 "n_intermittent_visible": int(vis[inter].sum()), "n_intermittent_not_visible": int((~vis[inter]).sum()),
                 "intermittent_recall_visible": S.ms([rec(p, 1, inter & vis) for p in P]),
                 "intermittent_recall_not_visible": S.ms([rec(p, 1, inter & ~vis) for p in P]) if (inter & ~vis).any() else None,
                 "intermittent_not_visible_called_healthy": S.ms([float((p[inter & ~vis] == 0).mean()) for p in P]) if (inter & ~vis).any() else None}
            res[sc] = r
            print(W, sc, json.dumps({k: v for k, v in r.items()}, default=float), flush=True)
        out["results"][str(W)] = res
    json.dump(out, open(os.path.join(HERE, "results", "window_length_v2.json"), "w"), indent=2, default=float)
    print("DONE")


if __name__ == "__main__":
    main()
