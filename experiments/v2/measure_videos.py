"""
Step 1 of the v2 real-video analysis: signed measurement with validity masks for the 13 Dreamstime clips,
an estimate of real localization jitter, and a digital rescaling test of iris normalization.

Outputs (results/real_video/):
  <group>_<clipID>_signal.csv  per-frame (30 fps) u, raw signed pixel displacement, diameter, validity, peak grey
  per_video.csv                 per-clip measurement availability
  jitter_estimate.json          frame-to-frame noise of u in jointly valid stretches
  rescaling_test.csv/json       raw pixel vs normalized displacement across image scales (one clip)
Only numerical values are written; no images.
"""
import os, sys, json, numpy as np, pandas as pd, cv2, mediapipe as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from dh_measure import measure_frame

VID = os.environ.get("DH_VIDEO_DIR", "videos")
OUT = os.path.join(HERE, "results", "real_video"); os.makedirs(OUT, exist_ok=True)
FPS = 30


def read_resampled(path, scale=1.0):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); fr = []
    while True:
        ok, im = cap.read()
        if not ok: break
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        fr.append(im)
    n = len(fr); idx = np.round(np.arange(0, n / fps, 1 / FPS) * fps).astype(int); idx = idx[idx < n]
    return [fr[i] for i in idx], fps, n / fps


def measure_clip(frames, thr=200):
    rows = []
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
        for k, im in enumerate(frames):
            r = fm.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            m = measure_frame(im, r.multi_face_landmarks[0] if r.multi_face_landmarks else None, thr)
            m["t"] = k / FPS; rows.append(m)
    return pd.DataFrame(rows)


def main():
    per = []
    for g in ["Normal", "Strabismus"]:
        for f in sorted(os.listdir(os.path.join(VID, g))):
            if not f.endswith(".mp4"): continue
            frames, fps, dur = read_resampled(os.path.join(VID, g, f))
            df = measure_clip(frames); clip = f[:-4]
            df.to_csv(os.path.join(OUT, f"{g}_{clip}_signal.csv"), index=False)
            joint = (df.eye1_valid & df.eye2_valid).astype(bool)
            per.append({"group": g, "clip_id": clip, "fps_native": round(fps, 2), "duration_s": round(dur, 2),
                        "frames": len(df), "face_pct": 100 * df.face.mean(),
                        "iris_d_px_median": float(np.nanmedian(df[["eye1_d", "eye2_d"]].values)),
                        "peak_grey_median": float(np.nanmedian(df[["eye1_peak", "eye2_peak"]].values)),
                        "eye1_valid_pct": 100 * df.eye1_valid.mean(), "eye2_valid_pct": 100 * df.eye2_valid.mean(),
                        "joint_valid_pct": 100 * joint.mean(),
                        "eye1_u_median_valid": float(df.eye1_u[df.eye1_valid == 1].median()) if df.eye1_valid.any() else np.nan,
                        "eye2_u_median_valid": float(df.eye2_u[df.eye2_valid == 1].median()) if df.eye2_valid.any() else np.nan})
            print(per[-1])
    pv = pd.DataFrame(per); pv.to_csv(os.path.join(OUT, "per_video.csv"), index=False)

    # ---- jitter estimate from clips with >= 50% jointly valid frames
    est = {}
    for _, r in pv[pv.joint_valid_pct >= 50].iterrows():
        df = pd.read_csv(os.path.join(OUT, f"{r.group}_{r.clip_id}_signal.csv"))
        for e in ["eye1", "eye2"]:
            v = df[f"{e}_valid"].values.astype(bool); u = df[f"{e}_u"].values
            d = np.diff(u)[v[1:] & v[:-1]]                     # consecutive valid pairs only
            est[f"{r.clip_id}_{e}"] = {"pairs": int(d.size), "sigma_from_diff": float(np.std(d) / np.sqrt(2)),
                                       "robust_sigma_from_diff": float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)),
                                       "u_median": float(np.median(u[v])), "u_sd": float(np.std(u[v]))}
    json.dump(est, open(os.path.join(OUT, "jitter_estimate.json"), "w"), indent=2); print(json.dumps(est, indent=2))

    # ---- digital rescaling test on the clip with the most valid frames
    best = pv.sort_values("joint_valid_pct", ascending=False).iloc[0]
    rows = []
    for sc in [0.5, 0.75, 1.0, 1.5, 2.0]:
        frames, _, _ = read_resampled(os.path.join(VID, best.group, best.clip_id + ".mp4"), sc)
        df = measure_clip(frames)
        for e in ["eye1", "eye2"]:
            v = df[f"{e}_valid"] == 1
            rows.append({"scale": sc, "eye": e, "valid_pct": 100 * v.mean(),
                         "iris_d_px": float(df[f"{e}_d"][v].median()) if v.any() else np.nan,
                         "raw_px_median": float(df[f"{e}_px"][v].median()) if v.any() else np.nan,
                         "u_median": float(df[f"{e}_u"][v].median()) if v.any() else np.nan})
    rs = pd.DataFrame(rows); rs.to_csv(os.path.join(OUT, "rescaling_test.csv"), index=False); print(rs.round(4).to_string())
    summ = {"clip": f"{best.group}/{best.clip_id}"}
    for e in ["eye1", "eye2"]:
        d = rs[(rs.eye == e) & rs.raw_px_median.notna()]
        cv = lambda x: float(np.std(x) / abs(np.mean(x))) if len(x) > 1 and np.mean(x) != 0 else np.nan
        summ[e] = {"scales_with_signal": d.scale.tolist(), "cv_raw_px": cv(d.raw_px_median), "cv_u": cv(d.u_median),
                   "raw_px_range": [float(d.raw_px_median.min()), float(d.raw_px_median.max())],
                   "u_range": [float(d.u_median.min()), float(d.u_median.max())]}
    json.dump(summ, open(os.path.join(OUT, "rescaling_test.json"), "w"), indent=2); print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
