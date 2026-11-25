"""Verification of the signed measurement (src/dh_measure.py) on a real clip with a visible reflex.
Test 1 (mirror invariance): u must be unchanged when the frame is flipped horizontally.
Test 2 (known displacements): a 3x3 white test glint is drawn k px temporal or nasal of the pupil
centre; the measured u must equal -k/d (glint temporal => pupil nasal) or +k/d (glint nasal)."""
import os, sys, json, numpy as np, cv2, mediapipe as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from dh_measure import measure_frame, IRIS
VID = os.environ.get("DH_VIDEO_DIR", "videos"); CLIP = os.path.join(VID, "Strabismus", "252652595.mp4")
cap = cv2.VideoCapture(CLIP); frames = []
while len(frames) < 90:
    ok, im = cap.read()
    if not ok: break
    frames.append(im)
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
res = {"mirror": [], "mirror_exact": [], "inject": []}
class _LM:  # landmark container with mirrored x (pixel x -> w-1-x)
    def __init__(self, L, w):
        self.landmark = [type("P", (), {"x": (w - 1) / w - q.x, "y": q.y})() for q in L.landmark]
for im in frames[::3]:
    r = fm.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if not r.multi_face_landmarks: continue
    L = r.multi_face_landmarks[0]; m0 = measure_frame(im, L)
    fl = cv2.flip(im, 1)
    mx = measure_frame(fl, _LM(L, im.shape[1]))
    for e in IRIS:
        if m0[f"{e}_valid"] and mx[f"{e}_valid"]:
            res["mirror_exact"].append(abs(m0[f"{e}_u"] - mx[f"{e}_u"]))
    rf = fm.process(cv2.cvtColor(fl, cv2.COLOR_BGR2RGB))
    if rf.multi_face_landmarks:
        m1 = measure_frame(fl, rf.multi_face_landmarks[0])
        a = sorted([m0["eye1_u"], m0["eye2_u"]]) if m0["eye1_valid"] and m0["eye2_valid"] else None
        b = sorted([m1["eye1_u"], m1["eye2_u"]]) if m1["eye1_valid"] and m1["eye2_valid"] else None
        if a and b: res["mirror"].append(float(np.max(np.abs(np.array(a) - np.array(b)))))
    h, w = im.shape[:2]; P = np.array([[p.x * w, p.y * h] for p in L.landmark])
    mid = (P[468, 0] + P[473, 0]) / 2
    for e, (c, _, _) in IRIS.items():
        p = P[c]; s = np.sign(p[0] - mid) or 1.0
        for k in (2, 3, 4):
            for side, expect_sign in (("temporal", -1), ("nasal", +1)):
                t = im.copy(); gx = int(round(p[0] + (s if side == "temporal" else -s) * k)); gy = int(round(p[1]))
                cv2.rectangle(t, (gx - 1, gy - 1), (gx + 1, gy + 1), (255, 255, 255), -1)
                m = measure_frame(t, L)
                if m[f"{e}_valid"]:
                    exp = expect_sign * abs(p[0] - gx) / m[f"{e}_d"]
                    res["inject"].append({"eye": e, "k": k, "side": side, "u": m[f"{e}_u"], "expected": exp,
                                          "sign_ok": np.sign(m[f"{e}_u"]) == expect_sign})
inj = res["inject"]
summary = {"frames_tested": len(frames[::3]),
           "mirror_exact_cases": len(res["mirror_exact"]), "mirror_exact_max_abs_diff": float(np.max(res["mirror_exact"])) if res["mirror_exact"] else None,
           "mirror_redetect_pairs": len(res["mirror"]), "mirror_max_abs_diff_median": float(np.median(res["mirror"])) if res["mirror"] else None,
           "inject_cases": len(inj), "inject_sign_correct_pct": 100 * float(np.mean([i["sign_ok"] for i in inj])),
           "inject_abs_error_median": float(np.median([abs(i["u"] - i["expected"]) for i in inj])),
           "inject_abs_error_max": float(np.max([abs(i["u"] - i["expected"]) for i in inj]))}
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(summary, open(os.path.join(HERE, "results", "measurement_verification.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
