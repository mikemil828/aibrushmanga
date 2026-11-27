"""Path geometry: splines, resampling, smooth noise, marching-squares outlines."""
import numpy as np


def catmull(pts, per=12):
    """Centripetal-free (uniform) Catmull-Rom through the rows of `pts` (N x D, any extra columns ride along)."""
    P = np.asarray(pts, float)
    if len(P) < 2:
        return P.copy()
    if len(P) == 2:
        t = np.linspace(0, 1, max(per, 2))[:, None]
        return P[0] + (P[1] - P[0]) * t
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    t = np.linspace(0, 1, per, endpoint=False)[:, None]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2][None])
    return np.vstack(out)


def arclen(P):
    d = np.hypot(*np.diff(P[:, :2], axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample(P, step):
    """Resample rows of P by arc length of the first two columns, at `step` px."""
    s = arclen(P)
    L = s[-1]
    if L < 1e-9:
        return P[:1].copy(), np.zeros(1)
    n = max(2, int(np.ceil(L / step)) + 1)
    ss = np.linspace(0, L, n)
    return np.stack([np.interp(ss, s, P[:, j]) for j in range(P.shape[1])], 1), ss


def normals(P):
    d = np.gradient(P[:, :2], axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    return n / np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-9)[:, None]


def smooth_noise(s, rng, scale, octaves=3):
    """Smooth 1-D noise in [-1, 1]-ish along arc length s, wavelength ~scale px."""
    out = np.zeros_like(s, dtype=float)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        f = (2 ** o) / max(scale, 1e-6)
        out += amp * np.sin(2 * np.pi * f * s * rng.uniform(0.8, 1.25) + rng.uniform(0, 2 * np.pi))
        tot += amp
        amp *= 0.5
    return out / tot


def outlines(field, level=0.5, min_len=8.0):
    """Iso-lines of a 2-D float field (marching squares). Returns a list of (N, 2) arrays in (x, y) px."""
    f = np.asarray(field, float)
    H, W = f.shape
    a, b, c, d = f[:-1, :-1], f[:-1, 1:], f[1:, 1:], f[1:, :-1]   # tl tr br bl
    idx = ((a > level) * 8 + (b > level) * 4 + (c > level) * 2 + (d > level) * 1).astype(np.uint8)
    ys, xs = np.nonzero((idx != 0) & (idx != 15))

    def interp(v0, v1):
        den = v1 - v0
        return np.where(np.abs(den) < 1e-12, 0.5, (level - v0) / np.where(np.abs(den) < 1e-12, 1, den))

    # edge points per cell: top (a-b), right (b-c), bottom (d-c), left (a-d), as keys for linking
    segs = []
    table = {1: [(3, 2)], 2: [(2, 1)], 3: [(3, 1)], 4: [(0, 1)], 5: [(3, 0), (2, 1)], 6: [(0, 2)], 7: [(3, 0)],
             8: [(3, 0)], 9: [(0, 2)], 10: [(0, 1), (3, 2)], 11: [(0, 1)], 12: [(3, 1)], 13: [(2, 1)], 14: [(3, 2)]}

    def edge_key(y, x, e):
        # unique key for the grid edge: ('h', y, x) horizontal edge between (y,x)-(y,x+1); ('v', y, x) vertical
        return {0: ('h', y, x), 1: ('v', y, x + 1), 2: ('h', y + 1, x), 3: ('v', y, x)}[e]

    def edge_pt(y, x, e):
        if e == 0:
            return (x + float(interp(f[y, x], f[y, x + 1])), float(y))
        if e == 1:
            return (float(x + 1), y + float(interp(f[y, x + 1], f[y + 1, x + 1])))
        if e == 2:
            return (x + float(interp(f[y + 1, x], f[y + 1, x + 1])), float(y + 1))
        return (float(x), y + float(interp(f[y, x], f[y + 1, x])))

    adj = {}
    pts = {}
    for y, x in zip(ys.tolist(), xs.tolist()):
        for e0, e1 in table[int(idx[y, x])]:
            k0, k1 = edge_key(y, x, e0), edge_key(y, x, e1)
            if k0 not in pts:
                pts[k0] = edge_pt(y, x, e0)
            if k1 not in pts:
                pts[k1] = edge_pt(y, x, e1)
            adj.setdefault(k0, []).append(k1)
            adj.setdefault(k1, []).append(k0)
    seen = set()
    lines = []
    starts = [k for k, v in adj.items() if len(v) == 1] + list(adj.keys())
    for k in starts:
        if k in seen:
            continue
        line = [k]
        seen.add(k)
        cur = k
        while True:
            nxt = [n for n in adj[cur] if n not in seen]
            if not nxt:
                break
            cur = nxt[0]
            seen.add(cur)
            line.append(cur)
        if len(line) > 2 and adj[line[-1]] and line[0] in adj[line[-1]]:
            line.append(line[0])
        arr = np.array([pts[q] for q in line])
        if len(arr) > 1 and arclen(arr)[-1] >= min_len:
            lines.append(arr)
    return lines


def simplify(P, tol=1.0):
    """Ramer-Douglas-Peucker on (N, 2)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = P[i + 1:j] - a
        d = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / L if L > 1e-9 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return P[keep]
