"""
Dynamic Hirschberg measurement (v2): signed, horizontal, iris-normalized pupil-glint displacement
with explicit validity masks.

For each eye e and frame t:
    u_t = s_e * (p_x,t - g_x,t) / d_t
where p = pupil-centre estimate (MediaPipe iris-centre landmark), g = corneal reflex (glint),
d = iris diameter (distance between the two horizontal iris-boundary landmarks), and
    s_e = sign(p_x,e - m_x),  m_x = midpoint of the two iris centres.
s_e makes the positive direction TEMPORAL for each eye, so u > 0 means the pupil centre lies temporal
to the reflex (reflex displaced nasally, as in exotropia) and u < 0 means nasal (as in esotropia).
Because s_e is defined from the face geometry, u is unchanged when the image is mirrored.
In this temporal convention a conjugate (both-eye) horizontal gaze shift changes u1 and u2 by equal and
opposite amounts, whereas a misalignment changes them with the same sign. The binocular alignment index
A_t = u1_t + u2_t therefore cancels conjugate gaze, as in the clinical comparison of the two reflexes.
The glint search window is +-0.35 iris diameters (at least 3 px), so it scales with image size.

A frame is valid for an eye only if the face is detected, d > 0 and a glint is found; invalid frames
have valid = 0 and u = 0. The mask, not the zero, is what downstream code must use.
"""
import numpy as np, cv2

IRIS = {"eye1": (468, 469, 471), "eye2": (473, 474, 476)}   # centre, boundary A, boundary B


def find_glint(img, c, thr, rad=5, tol=2):
    """Glint = centroid of the pixels within `tol` grey levels of the maximum of the 3x3-Gaussian-blurred
    grey patch centred on c (half-width `rad`), accepted if the maximum >= thr. Using the centroid of the
    brightest region (instead of the first maximum in raster order) makes the estimate sub-pixel and
    symmetric under horizontal mirroring when the glint is saturated."""
    h, w = img.shape[:2]; x, y = int(round(c[0])), int(round(c[1]))
    x1, y1, x2, y2 = max(0, x - rad), max(0, y - rad), min(w, x + rad + 1), min(h, y + rad + 1)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0
    g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(float)
    mx = g.max()
    if mx < thr:
        return None, mx
    yy, xx = np.nonzero(g >= mx - tol)
    return (x1 + xx.mean(), y1 + yy.mean()), mx


def measure_frame(img, face_landmarks, thr=200):
    """Return dict with per-eye signed u, raw signed pixel displacement, diameter, validity, peak grey."""
    h, w = img.shape[:2]
    out = {"face": face_landmarks is not None}
    if face_landmarks is None:
        for e in IRIS:
            out.update({f"{e}_u": 0.0, f"{e}_px": 0.0, f"{e}_d": np.nan, f"{e}_valid": 0, f"{e}_peak": np.nan})
        return out
    P = np.array([[p.x * w, p.y * h] for p in face_landmarks.landmark])
    mid_x = (P[IRIS["eye1"][0], 0] + P[IRIS["eye2"][0], 0]) / 2
    for e, (c, a, b) in IRIS.items():
        p = P[c]; d = float(np.linalg.norm(P[a] - P[b]))
        s = np.sign(p[0] - mid_x) or 1.0
        g, peak = find_glint(img, p, thr, rad=max(3, int(round(0.35 * d))))   # window scales with iris size
        ok = g is not None and d > 0
        px = float(s * (p[0] - g[0])) if ok else 0.0
        out.update({f"{e}_u": px / d if ok else 0.0, f"{e}_px": px, f"{e}_d": d,
                    f"{e}_valid": int(ok), f"{e}_peak": float(peak)})
    return out


def quality_gate(valid1, valid2, min_joint=0.7, max_gap=15):
    """Accept a window only if both eyes are jointly valid in >= min_joint of frames and the longest
    run of jointly invalid frames is <= max_gap. Returns (accepted, joint_fraction, longest_gap, reason)."""
    j = np.asarray(valid1, bool) & np.asarray(valid2, bool)
    frac = float(j.mean())
    gap, run = 0, 0
    for v in j:
        run = 0 if v else run + 1; gap = max(gap, run)
    if frac < min_joint:
        return False, frac, gap, f"joint valid {frac:.0%} < {min_joint:.0%}"
    if gap > max_gap:
        return False, frac, gap, f"gap {gap} frames > {max_gap}"
    return True, frac, gap, "accepted"
