"""End-to-end processing time per frame and per 5-s window on the workstation CPU:
video decoding, MediaPipe Face Mesh + reflex localization (src/dh_measure.py), and classifier inference."""
import os, sys, time, json, numpy as np, cv2, mediapipe as mp
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from dh_measure import measure_frame
import run_synthetic_v2 as S
VID = os.environ.get("DH_VIDEO_DIR", "videos"); clip = os.path.join(VID, "Strabismus", "252652595.mp4")
cap = cv2.VideoCapture(clip); t0 = time.perf_counter(); frames = []
while len(frames) < 600:
    ok, im = cap.read()
    if not ok: break
    frames.append(im)
t_dec = (time.perf_counter() - t0) / len(frames)
fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
for im in frames[:20]: fm.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
t_mp, t_gl = [], []
for im in frames:
    a = time.perf_counter(); r = fm.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); b = time.perf_counter()
    measure_frame(im, r.multi_face_landmarks[0] if r.multi_face_landmarks else None); c = time.perf_counter()
    t_mp.append(b - a); t_gl.append(c - b)
m = tf.keras.models.load_model(os.path.join(HERE, "results", "models", "CNN-only_seed0.keras"))
f = tf.function(lambda x: m(x, training=False)); x = tf.zeros((1, 150, 4))
for _ in range(20): f(x)
a = time.perf_counter()
for _ in range(300): f(x)
t_cls = (time.perf_counter() - a) / 300
out = {"frames": len(frames), "resolution": [frames[0].shape[1], frames[0].shape[0]],
       "decode_ms_per_frame": 1000 * t_dec, "mediapipe_ms_per_frame": 1000 * float(np.mean(t_mp)),
       "reflex_measure_ms_per_frame": 1000 * float(np.mean(t_gl)),
       "measurement_ms_per_frame_total": 1000 * float(t_dec + np.mean(t_mp) + np.mean(t_gl)),
       "classifier_ms_per_window": 1000 * t_cls,
       "end_to_end_ms_per_5s_window": 1000 * float(150 * (t_dec + np.mean(t_mp) + np.mean(t_gl)) + t_cls)}
json.dump(out, open(os.path.join(HERE, "results", "latency_e2e.json"), "w"), indent=2); print(json.dumps(out, indent=2))
