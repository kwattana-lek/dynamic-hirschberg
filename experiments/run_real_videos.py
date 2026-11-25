"""
Real-video feasibility check of the Dynamic Hirschberg pipeline (run 2026-11-25).

Videos: $DH_VIDEO_DIR/{Normal,Strabismus}/*.mp4 (Dreamstime clip IDs in real_video/dreamstime_video_ids.csv)
Processing is local only; nothing is uploaded. Per-frame values are saved, not images.

Pipeline (identical to inference_realtime_video.py / manuscript Section 2.2):
  MediaPipe Face Mesh (refine_landmarks=True, 1 face); pupil centre = iris landmark 468 / 473;
  iris diameter = |469-471| / |474-476|; glint = brightest pixel of 3x3-Gaussian-blurred grey
  patch within +-5 px of the iris centre, accepted if >= THRESH; r = |p - g| / d; 0 if missing.
Videos are resampled to 30 fps (nearest frame by timestamp). Non-overlapping 150-frame windows
are classified by the seed-0 CNN-BiLSTM, retrained here with exactly the split, seed and settings
of rerun_experiments.py (the earlier script did not save model weights).
"""
import os, sys, json, numpy as np, pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import cv2, mediapipe as mp, tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rerun_experiments as R

# Folder containing Normal/ and Strabismus/ sub-folders of .mp4 files (Dreamstime clips; not redistributed).
# Set with:  export DH_VIDEO_DIR=/path/to/videos
VID = os.environ.get("DH_VIDEO_DIR", "videos")
OUT = os.path.join(HERE, "real_video"); os.makedirs(OUT, exist_ok=True)
THRESHOLDS = [200, 180, 220]   # 200 = pipeline default; others = sensitivity check
FPS = 30
EYES = {"eye1": (468, 469, 471), "eye2": (473, 474, 476)}


def find_glint(img, c, thr, rad=5):
    h, w = img.shape[:2]; x, y = c
    x1, y1, x2, y2 = max(0, x - rad), max(0, y - rad), min(w, x + rad), min(h, y + rad)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return None
    g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, mx, _, loc = cv2.minMaxLoc(g)
    return None if mx < thr else (x1 + loc[0], y1 + loc[1])


def process(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok: break
        frames.append(im)
    cap.release()
    n = len(frames); dur = n / fps
    idx = np.round(np.arange(0, dur, 1 / FPS) * fps).astype(int); idx = idx[idx < n]
    rows = []
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                         static_image_mode=False) as fm:
        for k, i in enumerate(idx):
            im = frames[i]; h, w = im.shape[:2]
            res = fm.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            row = {"t": k / FPS, "face": res.multi_face_landmarks is not None}
            if row["face"]:
                pts = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark]).astype(int)
            for e, (c, a, b) in EYES.items():
                row[f"{e}_diam_px"] = float(np.linalg.norm(pts[a] - pts[b])) if row["face"] else np.nan
                for thr in THRESHOLDS:
                    r = 0.0
                    if row["face"] and row[f"{e}_diam_px"] > 0:
                        g = find_glint(im, tuple(pts[c]), thr)
                        if g is not None:
                            r = float(np.linalg.norm(pts[c] - np.array(g))) / row[f"{e}_diam_px"]
                    row[f"{e}_r_{thr}"] = r
            rows.append(row)
    return pd.DataFrame(rows), fps, dur, (frames[0].shape[1], frames[0].shape[0])


def train_model():
    X, y, _ = R.load_dataset(); idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=R.SPLIT_SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=R.SPLIT_SEED)
    m, ep, _ = R.train_nn(R.cnn_bilstm, X[tr], y[tr], X[va], y[va], 0)
    acc = float((m.predict(X[te], verbose=0).argmax(1) == y[te]).mean())
    m.save(os.path.join(OUT, "cnn_bilstm_seed0.keras"))
    return m, acc, ep


def main():
    model, acc, ep = train_model()
    print("model test acc", acc, "epochs", ep)
    per_video, windows = [], []
    for grp in ["Normal", "Strabismus"]:
        for f in sorted(os.listdir(os.path.join(VID, grp))):
            if not f.endswith(".mp4"): continue
            df, fps, dur, (W, H) = process(os.path.join(VID, grp, f))
            vid = f"{grp}/{f}"; df.to_csv(os.path.join(OUT, f"{grp}_{f[:-4]}_signal.csv"), index=False)
            rec = {"video": vid, "group": grp, "width": W, "height": H, "fps_native": round(fps, 2),
                   "duration_s": round(dur, 2), "frames_30fps": len(df),
                   "face_detected_pct": 100 * df["face"].mean(),
                   "iris_diam_px_median": float(np.nanmedian(df[["eye1_diam_px", "eye2_diam_px"]].values))}
            for thr in THRESHOLDS:
                both = (df[f"eye1_r_{thr}"] > 0) & (df[f"eye2_r_{thr}"] > 0)
                rec[f"valid_both_pct_{thr}"] = 100 * both.mean()
                rec[f"glint_eye1_pct_{thr}"] = 100 * (df[f"eye1_r_{thr}"] > 0).mean()
                rec[f"glint_eye2_pct_{thr}"] = 100 * (df[f"eye2_r_{thr}"] > 0).mean()
            # classify non-overlapping 150-frame windows at the default threshold
            sig = df[["eye1_r_200", "eye2_r_200"]].values.astype("float32")
            nw = len(sig) // R.T
            preds = []
            for k in range(nw):
                w = sig[k * R.T:(k + 1) * R.T]
                p = model.predict(w[None], verbose=0)[0]
                preds.append(int(p.argmax()))
                windows.append({"video": vid, "group": grp, "window": k, "pred": R.CLASSES[int(p.argmax())],
                                "p_healthy": p[0], "p_intermittent": p[1], "p_constant": p[2],
                                "valid_both_pct": 100 * ((w[:, 0] > 0) & (w[:, 1] > 0)).mean(),
                                "eye1_median_r": float(np.median(w[w[:, 0] > 0, 0])) if (w[:, 0] > 0).any() else np.nan,
                                "eye2_median_r": float(np.median(w[w[:, 1] > 0, 1])) if (w[:, 1] > 0).any() else np.nan})
            rec["n_windows"] = nw
            rec["pred_counts"] = {c: preds.count(i) for i, c in enumerate(R.CLASSES)}
            per_video.append(rec); print(rec)
    pv = pd.DataFrame(per_video); pv.to_csv(os.path.join(OUT, "per_video.csv"), index=False)
    wd = pd.DataFrame(windows); wd.to_csv(os.path.join(OUT, "windows.csv"), index=False)

    summary = {"model_test_acc_synthetic": acc, "n_videos": len(pv), "n_windows": len(wd),
               "by_group": {}}
    for g, d in pv.groupby("group"):
        w = wd[wd.group == g]
        summary["by_group"][g] = {
            "videos": len(d), "windows": len(w),
            "face_detected_pct_median": float(d.face_detected_pct.median()),
            "valid_both_pct_median_200": float(d.valid_both_pct_200.median()),
            "valid_both_pct_range_200": [float(d.valid_both_pct_200.min()), float(d.valid_both_pct_200.max())],
            "valid_both_pct_median_180": float(d.valid_both_pct_180.median()),
            "valid_both_pct_median_220": float(d.valid_both_pct_220.median()),
            "iris_diam_px_median": float(d.iris_diam_px_median.median()),
            "pred_distribution": w.pred.value_counts().to_dict(),
            "window_median_r_eye1": float(w.eye1_median_r.median()),
            "window_median_r_eye2": float(w.eye2_median_r.median())}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))

    # example trajectories: one Normal and one Strabismus video with the most valid frames
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for a, g in zip(ax, ["Normal", "Strabismus"]):
        best = pv[pv.group == g].sort_values("valid_both_pct_200", ascending=False).iloc[0]
        df = pd.read_csv(os.path.join(OUT, f"{g}_{os.path.basename(best.video)[:-4]}_signal.csv")).iloc[:R.T]
        for col, lab, ls in [("eye1_r_200", "Eye 1", "-"), ("eye2_r_200", "Eye 2", "--")]:
            v = df[col].values; a.plot(df.t, np.where(v > 0, v, np.nan), ls, lw=1.4, label=lab)
            for t in df.t[v == 0]: a.axvspan(t, t + 1 / FPS, color="grey", alpha=.15, lw=0)
        a.set_title(f"{g} video (first 5 s)"); a.set_xlabel("Time (s)"); a.grid(alpha=.3, ls="--")
    ax[0].set_ylabel("Normalized pupil--glint distance"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "real_trajectories.png"), dpi=300)


if __name__ == "__main__":
    main()
