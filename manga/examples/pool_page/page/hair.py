"""Hair as an inker draws it: masses of strands that follow the flow, not a black shape.

Every stroke runs along the hair's own direction (read from the reference's strand lines), starts at the
root end and tapers out toward the tip. Dark parts get dense black strokes, lit parts thin out to paper,
and the deepest shadow is solid black that breaks up into tapered tips at its edge.
"""
import numpy as np
from scipy import ndimage as nd
from manga import tone, geom, prep


def _flow(gray, mask, s):
    ang = prep.orientation(gray, sigma=3.0 * s)
    return np.deg2rad(ang)


def _streamline(theta, m, x, y, step, nmax, ref_dir=None):
    H, W = m.shape
    out = {}
    for sign in (1, -1):
        px, py, prev, pts = x, y, None, []
        for _ in range(nmax):
            ix, iy = int(px), int(py)
            if not (0 <= ix < W and 0 <= iy < H) or not m[iy, ix]:
                break
            a = theta[iy, ix]
            v = np.array([np.cos(a), np.sin(a)])
            if prev is None:
                if ref_dir is not None and v @ ref_dir < 0:
                    v = -v
                v = v * sign
            elif v @ prev < 0:
                v = -v
            prev = v
            px, py = px + v[0] * step, py + v[1] * step
            pts.append((px, py))
        out[sign] = pts
    P = out[-1][::-1] + [(x, y)] + out[1]
    return np.array(P) if len(P) >= 3 else None


def draw_hair(sh, hair, gray, lines, box, s, style='strands', seed=7, root=None, density=0.12, deep_at=0.75, ring=None):
    """hair, gray, lines: panel-size arrays. root: (x, y) panel px of the crown, strokes taper away from it."""
    X0, Y0, X1, Y1 = box
    H, W = hair.shape
    rng = np.random.default_rng(seed)
    d = nd.gaussian_filter(1 - gray, 1.5 * s)
    theta = _flow(gray, hair, s)
    full = lambda a: _place(sh, a, box)                                     # noqa: E731
    # light and shadow read per lock (each connected piece of hair), so a braid in shade still shows its plaits
    t = np.ones_like(d)
    lab, nl = nd.label(hair)
    for i in range(1, nl + 1):
        mi = lab == i
        if mi.sum() < 30:
            continue
        lo, hi = np.percentile(d[mi], [15, 85])
        t[mi] = np.clip((d[mi] - lo) / max(hi - lo, 0.08), 0, 1)
    n = 0
    if style in ('strands', 'tone'):
        if style == 'tone':
            # mid tone for the mass, paper where the reference shines (the highlight band)
            lit = nd.binary_opening((t < 0.12) & hair, iterations=max(1, int(s)))
            if ring is not None:
                lit &= ring
            tone(sh, full(hair & ~lit), level=np.where(full((t < 0.12) & hair & ~lit), 0.3, 0.58).astype(np.float32),
                 lpi=60)
        # solid black in the deepest third
        deep = nd.binary_opening((t > deep_at) & hair, iterations=max(1, int(s)))
        sh.fill(full(deep))
        # strand strokes: dense where dark, sparse where lit. Seeds on a jittered grid.
        sp = 2.2 * s
        ys, xs = np.mgrid[0:H:sp, 0:W:sp]
        xs = (xs + rng.uniform(0, sp, xs.shape)).ravel(); ys = (ys + rng.uniform(0, sp, ys.shape)).ravel()
        for x, y in zip(xs, ys):
            ix, iy = int(x), int(y)
            if not (0 <= ix < W and 0 <= iy < H) or not hair[iy, ix] or deep[iy, ix]:
                continue
            p_keep = 0.15 + 0.85 * t[iy, ix] ** 1.3
            if rng.random() > p_keep * density:
                continue
            P = _streamline(theta, hair, x, y, 1.0 * s, int(rng.uniform(18, 45) * s))
            if P is None or geom.arclen(P)[-1] < 8 * s or _turn(P) > 1.3:
                continue
            # the root end is the one nearer the crown (or the darker end)
            if root is not None and np.hypot(*(P[0] - root)) > np.hypot(*(P[-1] - root)):
                P = P[::-1]
            pr = float(0.18 + 0.5 * t[iy, ix])
            Q = geom.simplify(P, 0.5) + [X0, Y0]
            sh.stroke(Q, pen='g_pen', pressure=pr, taper=(0.3 * s, 2.2 * s), seed=int(rng.integers(1 << 30)))
            n += 1
        # the drawn clump lines of the reference, as the inker's main strand lines
        from manga import trace
        cl = lines & nd.binary_erosion(hair, iterations=int(s))
        for pts, a, b in trace.trace(trace.thin(cl)):
            if len(pts) < 2 or geom.arclen(pts)[-1] < 10 * s:
                continue
            if root is not None and np.hypot(*(pts[0] - root)) > np.hypot(*(pts[-1] - root)):
                pts = pts[::-1]
            sh.stroke(geom.simplify(pts, 0.6) + [X0, Y0], pen='g_pen', pressure=0.45,
                      taper=(0.4 * s, 2.0 * s), seed=int(rng.integers(1 << 30)))
            n += 1
    elif style == 'black_white':
        # black mass, then white strands scraped in where the reference is lit: strands of light
        sh.fill(full(hair))
        sp = 2.0 * s
        ys, xs = np.mgrid[0:H:sp, 0:W:sp]
        xs = (xs + rng.uniform(0, sp, xs.shape)).ravel(); ys = (ys + rng.uniform(0, sp, ys.shape)).ravel()
        inner = nd.binary_erosion(hair, iterations=int(1.5 * s))
        for x, y in zip(xs, ys):
            ix, iy = int(x), int(y)
            if not (0 <= ix < W and 0 <= iy < H) or not inner[iy, ix]:
                continue
            lit = 1 - t[iy, ix]
            if rng.random() > density * (0.05 + 0.9 * lit ** 1.5):
                continue
            P = _streamline(theta, inner, x, y, 1.0 * s, int(rng.uniform(10, 30) * s * (0.5 + lit)))
            if P is None or geom.arclen(P)[-1] < 6 * s or _turn(P) > 1.3:
                continue
            sh.stroke(geom.simplify(P, 0.5) + [X0, Y0], pen='white', mode='white', pressure=float(0.03 + 0.12 * lit),
                      taper=(1.2 * s, 1.2 * s), seed=int(rng.integers(1 << 30)))
            n += 1
    return n


def _turn(P):
    """Total turning of a path in radians: hair strands bend gently, loops are a flow-field artefact."""
    v = np.diff(P[::3] if len(P) > 9 else P, axis=0)
    a = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    return float(np.abs(np.diff(a)).sum()) if len(a) > 1 else 0.0


def _place(sh, a, box):
    X0, Y0, X1, Y1 = box
    out = np.zeros((sh.H0, sh.W0), a.dtype)
    out[Y0:Y1, X0:X1] = a
    return out
