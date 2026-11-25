"""Diagnostic: peak (blurred) grey level within +-5 px of each iris centre, every 5th frame."""
import os, numpy as np, cv2, mediapipe as mp, pandas as pd
# Folder containing Normal/ and Strabismus/ sub-folders of .mp4 files (Dreamstime clips; not redistributed).
# Set with:  export DH_VIDEO_DIR=/path/to/videos
VID = os.environ.get("DH_VIDEO_DIR", "videos")
rows = []
for g in ["Normal", "Strabismus"]:
    for f in sorted(os.listdir(os.path.join(VID, g))):
        if not f.endswith(".mp4"): continue
        cap = cv2.VideoCapture(os.path.join(VID, g, f)); k = 0; peaks = []; frame_max = []
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
            while True:
                ok, im = cap.read()
                if not ok: break
                k += 1
                if k % 5: continue
                h, w = im.shape[:2]; r = fm.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                if not r.multi_face_landmarks: continue
                L = r.multi_face_landmarks[0].landmark
                for c in (468, 473):
                    x, y = int(L[c].x * w), int(L[c].y * h)
                    roi = im[max(0, y-5):y+5, max(0, x-5):x+5]
                    if roi.size: peaks.append(cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0).max())
                frame_max.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).max())
        rows.append({"video": f"{g}/{f}", "peak_near_iris_median": float(np.median(peaks)) if peaks else np.nan,
                     "peak_near_iris_p90": float(np.percentile(peaks, 90)) if peaks else np.nan,
                     "frame_max_median": float(np.median(frame_max)) if frame_max else np.nan})
d = pd.DataFrame(rows); d.to_csv("real_video/glint_intensity_diagnostic.csv", index=False); print(d.round(0).to_string())
