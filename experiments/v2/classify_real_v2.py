"""
Step 2 of the v2 real-video analysis: quality gate, then classification of accepted windows.

Input: results/real_video/*_signal.csv from measure_videos.py; seed-0 models from run_synthetic_v2.py.
Each clip is cut into non-overlapping 300-frame (10-s) windows. A window is classified only if it passes the
quality gate (src/dh_measure.quality_gate); otherwise the output is "Cannot assess" with the reason.
For transparency, the ungated CNN-BiLSTM output is also stored for rejected windows (not reported as a result).
"""
import os, sys, json, numpy as np, pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from dh_measure import quality_gate
import run_synthetic_v2 as S   # for PosEmb and net_input

RV = os.path.join(HERE, "results", "real_video"); T = S.T   # 300 frames = 10 s
MIN_JOINT = float(os.environ.get("DH_MIN_JOINT", 0.7)); MAX_GAP = int(os.environ.get("DH_MAX_GAP", 15))
models = {n: tf.keras.models.load_model(os.path.join(HERE, "results", "models", f"{n}_seed0.keras"),
                                        custom_objects={"PosEmb": S.PosEmb}) for n in S.ARCH}
pv = pd.read_csv(os.path.join(RV, "per_video.csv"), dtype={"clip_id": str}); rows = []
for _, r in pv.iterrows():
    df = pd.read_csv(os.path.join(RV, f"{r.group}_{r.clip_id}_signal.csv"))
    U = df[["eye1_u", "eye2_u"]].values.astype("float32"); M = df[["eye1_valid", "eye2_valid"]].values.astype("float32")
    for k in range(len(df) // T):
        u, m = U[k*T:(k+1)*T], M[k*T:(k+1)*T]
        ok, frac, gap, why = quality_gate(m[:, 0], m[:, 1], MIN_JOINT, MAX_GAP)
        x = S.net_input(u[None], m[None]); row = {"group": r.group, "clip_id": r.clip_id, "window": k,
               "joint_valid_pct": 100 * frac, "longest_gap": gap, "gate": "accepted" if ok else "Cannot assess", "reason": why}
        j = (m[:, 0] * m[:, 1]) > 0
        row["binocular_index_median"] = float(np.median(u[j, 0] + u[j, 1])) if j.any() else np.nan
        for n, mdl in models.items():
            p = mdl.predict(x, verbose=0)[0]
            row[f"{n}_pred"] = S.CLASSES[int(p.argmax())] if ok else "Cannot assess"
            row[f"{n}_pmax"] = float(p.max())
            if n == "CNN-BiLSTM": row["CNN-BiLSTM_ungated"] = S.CLASSES[int(p.argmax())]
        rows.append(row)
w = pd.DataFrame(rows); w.to_csv(os.path.join(RV, f"windows_gate_{MIN_JOINT}_{MAX_GAP}.csv"), index=False)
summ = {"gate": {"min_joint": MIN_JOINT, "max_gap": MAX_GAP}, "windows": len(w),
        "accepted": int((w.gate == "accepted").sum()),
        "by_group": {g: {"windows": int(len(d)), "accepted": int((d.gate == "accepted").sum()),
                         "accepted_clips": sorted(d[d.gate == "accepted"].clip_id.unique().tolist()),
                         "reasons_rejected": d[d.gate != "accepted"].reason.str.replace(r"\d+%|\d+", "#", regex=True).value_counts().to_dict(),
                         "CNN-BiLSTM_ungated_on_rejected": d[d.gate != "accepted"]["CNN-BiLSTM_ungated"].value_counts().to_dict()}
                     for g, d in w.groupby("group")},
        "accepted_windows": w[w.gate == "accepted"][["group", "clip_id", "window", "joint_valid_pct", "longest_gap",
                                                     "binocular_index_median"] + [f"{n}_pred" for n in S.ARCH]].to_dict("records")}
json.dump(summ, open(os.path.join(RV, f"summary_gate_{MIN_JOINT}_{MAX_GAP}.json"), "w"), indent=2, default=float)
print(json.dumps(summ, indent=2, default=float))
