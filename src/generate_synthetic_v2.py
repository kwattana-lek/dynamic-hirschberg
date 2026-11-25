"""
Synthetic temporal sequence generator, version 2 (STSG-v2).

Each sequence: T = 150 frames (5 s at 30 fps), two eyes, signed temporal-convention displacement u
(src/dh_measure.py) plus a validity mask per eye. Invalid frames have u = 0 AND mask = 0; models
receive the mask, so a missing observation is never presented as a genuine zero.

Components (all in iris-diameter units):
  kappa_e      per-eye baseline offset (angle kappa), U(-0.01, 0.05)
  c_t          conjugate gaze: piecewise-constant horizontal gaze level in U(-g, g) with 0-3 shifts
               (3-frame transitions) plus slow head drift; enters the eyes with OPPOSITE signs
               (+c for eye 1, -c for eye 2) because u is expressed in each eye's temporal direction
  dev_t        deviation of the affected eye (randomly eye 1 or 2), sign + (exo) or - (eso) at random
     Healthy:      none
     Constant:     A ~ U(0.08, 0.25) throughout, with a small slow wobble
     Intermittent: 1-2 trapezoidal episodes (onset 5-30, plateau 15-75, recovery 5-30 frames), amplitude
                   A ~ U(0.08, 0.20); episodes may start before or end after the window (truncated) and
                   may recover inside it; at least 20 aligned and 20 deviated frames inside the window
  noise        Gaussian, sigma ~ U(0.005, 0.02) per sequence (range of the frame-to-frame noise measured on
               the real clips with valid signals: 0.005-0.018)
  missing      0-3 blinks (both eyes, 3-10 frames), per-eye random dropout with rate U(0, 0.10), and with
               probability 0.3 one dropout burst of 5-20 frames per eye
Intermittent and Constant amplitudes overlap; the classes differ in temporal structure, not amplitude.
"""
import numpy as np

T = 150
CLASSES = ["Healthy", "Intermittent", "Constant"]

DEFAULT = dict(sigma=(0.005, 0.02), gaze=0.15, n_gaze_shifts=(0, 3), drop=(0.0, 0.10), burst_p=0.3,
               blinks=(0, 3), amp_int=(0.08, 0.20), amp_const=(0.08, 0.25), kappa=(-0.01, 0.05))


def _trapezoid(start, up, plat, down):
    t = np.arange(T, dtype=float); a = np.zeros(T)
    a = np.where((t >= start) & (t < start + up), (t - start) / up, a)
    a = np.where((t >= start + up) & (t < start + up + plat), 1.0, a)
    e = start + up + plat
    a = np.where((t >= e) & (t < e + down), 1 - (t - e) / down, a)
    return a


def _intermittent(rng, amp):
    for _ in range(1000):
        env = np.zeros(T)
        for _ in range(rng.integers(1, 3)):
            env = np.maximum(env, _trapezoid(rng.integers(-40, 141), rng.integers(5, 31),
                                             rng.integers(15, 76), rng.integers(5, 31)))
        if (env < 0.1).sum() >= 20 and (env > 0.5).sum() >= 20:
            return amp * env
    raise RuntimeError("could not satisfy intermittent constraints")


def _gaze(rng, g, shifts):
    lev = rng.uniform(-g, g); c = np.full(T, lev)
    for t0 in np.sort(rng.integers(5, T - 5, rng.integers(shifts[0], shifts[1] + 1))):
        new = rng.uniform(-g, g); ramp = np.clip((np.arange(T) - t0) / 3.0, 0, 1)
        c = c + (new - lev) * ramp; lev = new
    tt = np.linspace(0, 2 * np.pi, T)
    return c + np.sin(tt + rng.uniform(0, 2 * np.pi)) * rng.uniform(0.001, 0.01)


def generate_sequence(rng, cls, **over):
    cfg = {**DEFAULT, **over}
    u = np.zeros((T, 2)); kappa = rng.uniform(*cfg["kappa"], 2)
    c = _gaze(rng, cfg["gaze"], cfg["n_gaze_shifts"])
    u[:, 0] = kappa[0] + c; u[:, 1] = kappa[1] - c
    for e in range(2):                                           # small independent fixational drift
        u[:, e] += np.cumsum(rng.normal(0, 0.0007, T))
    aff = int(rng.integers(0, 2)); sign = float(rng.choice([1.0, -1.0])); A = 0.0
    if cls == "Constant":
        A = rng.uniform(*cfg["amp_const"])
        u[:, aff] += sign * A * (1 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, T) + rng.uniform(0, 6.3)))
    elif cls == "Intermittent":
        A = rng.uniform(*cfg["amp_int"]); u[:, aff] += sign * _intermittent(rng, A)
    sigma = rng.uniform(*cfg["sigma"]); u += rng.normal(0, sigma, (T, 2))
    m = np.ones((T, 2))
    for _ in range(rng.integers(cfg["blinks"][0], cfg["blinks"][1] + 1)):
        d = rng.integers(3, 11); s = rng.integers(0, T - d); m[s:s + d, :] = 0
    for e in range(2):
        m[rng.random(T) < rng.uniform(*cfg["drop"]), e] = 0
        if rng.random() < cfg["burst_p"]:
            d = rng.integers(5, 21); s = rng.integers(0, T - d); m[s:s + d, e] = 0
    u[m == 0] = 0.0
    return u.astype("float32"), m.astype("float32"), {"affected_eye": aff + 1, "sign": sign, "amplitude": A,
                                                     "sigma": sigma}


def generate_set(seed, n_per_class, **over):
    rng = np.random.default_rng(seed); U, M, Y, META = [], [], [], []
    for k, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            u, m, meta = generate_sequence(rng, c, **over); U.append(u); M.append(m); Y.append(k); META.append(meta)
    return np.stack(U), np.stack(M), np.array(Y), META


if __name__ == "__main__":
    import os, json
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "synthetic_v2")
    os.makedirs(out, exist_ok=True)
    U, M, Y, META = generate_set(2026, 500)
    np.savez_compressed(os.path.join(out, "synthetic_v2.npz"), u=U, mask=M, y=Y)
    json.dump(META, open(os.path.join(out, "meta.json"), "w"))
    print(U.shape, np.bincount(Y))
