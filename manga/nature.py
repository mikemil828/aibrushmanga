"""Drawn vegetation: grass blades, grass along edges, tufts, and rice seedlings standing in water.

Tracing grass or seedlings from a reference gives fat blobs; an inker draws each blade as one tapered stroke.
Everything here works in source coordinates (reference px) and maps through `to_page` (identity by default),
so it can be called with `Frame.P`. `depth(y)` returns how near a point is: < 0 skips it (out of the drawn
zone), 0 is far and 1 is the front (write it with plain arithmetic so it also takes arrays); nearer blades are longer, heavier and denser.

Density matters more than anything: on the rice-field panel, 15k blades read as a black thicket and about 6.6k
read as grass. Check the stroke count and look at the sheet before going further.
"""
import numpy as np
from scipy import ndimage as nd

from . import geom


def _ident(x, y):
    return (x, y)


def blade(sh, root, angle, length, near, seed, to_page=None, bend=0.25, pen='g_pen', pressure=(0.16, 0.24),
          taper=(0.2, 1.0, 2.0)):
    """One blade from its root: curves as it rises, tapers to a point. angle in radians from vertical."""
    tp = to_page or _ident
    x, y = root
    tip = (x + np.sin(angle) * length, y - np.cos(angle) * length)
    mid = (x + np.sin(angle) * length * 0.5 + bend * length * np.sign(angle) * 0.3, y - np.cos(angle) * length * 0.55)
    sh.stroke([tp(x, y), tp(*mid), tp(*tip)], pen=pen, pressure=pressure[0] + pressure[1] * near,
              taper=(taper[0], taper[1] + taper[2] * near), seed=seed)


def edge_grass(sh, mask, depth, to_page=None, rng=None, every=(8.0, 3.5), count=(1, 1.5), length=(3.0, 16.0),
               jitter=0.8, lean_noise=0.35, seed=20000, blade_kw=None):
    """Blades along every edge of `mask` (a bank meeting water or a path), leaning out toward the open side.

    every  -- (a, b): one cluster every a - b * near px of edge.
    count  -- (a, b): a .. a + 1 + int(b * near) blades a cluster.
    length -- (a, b): blade length a + b * near, times a random 0.6 .. 1.2.
    Returns the number of blades.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    k = 0
    f = nd.gaussian_filter(np.asarray(mask, np.float32), 1.0)
    for Pl in geom.outlines(f, 0.5, 20):
        Q = geom.resample(Pl, 1.0)[0]
        nrm = geom.normals(Q)
        acc = 0.0
        for q, nv in zip(Q, nrm):
            x, y = q
            d = depth(y)
            if d < 0:
                continue
            near = min(d, 1.0)
            acc += 1.0
            if acc < every[0] - every[1] * near:
                continue
            acc = 0.0
            lean = np.clip(np.arctan2(nv[0], -nv[1]) * 0.5, -0.9, 0.9)
            for _ in range(rng.integers(count[0], count[0] + 2 + int(count[1] * near))):
                ln = (length[0] + length[1] * near) * rng.uniform(0.6, 1.2)
                blade(sh, (x + rng.normal(0, jitter), y), lean + rng.normal(0, lean_noise), ln, near, seed + k,
                      to_page, **(blade_kw or {}))
                k += 1
    return k


def tufts(sh, mask, depth, to_page=None, rng=None, prob=(0.0012, 0.002), blades=(3, 2), length=(4.0, 14.0),
          jitter=1.2, spread=0.45, seed=20000, blade_kw=None):
    """Tufts scattered inside `mask`: each pixel starts a tuft with probability prob[0] + prob[1] * near;
    a tuft is blades[0] + int(blades[1] * near) blades fanned around vertical. Returns the number of blades."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    ys, xs = np.nonzero(mask)
    d = np.asarray(depth(ys), float)                 # depth must accept an array (plain arithmetic does)
    ok = d >= 0
    xs, ys, near = xs[ok], ys[ok], np.clip(d[ok], 0, 1)
    pick = rng.random(len(xs)) < prob[0] + prob[1] * near
    k = 0
    for x, y, n_ in zip(xs[pick], ys[pick], near[pick]):
        for _ in range(blades[0] + int(blades[1] * n_)):
            ln = (length[0] + length[1] * n_) * rng.uniform(0.5, 1.1)
            blade(sh, (x + rng.normal(0, jitter), y), rng.normal(0, spread), ln, n_, seed + k, to_page,
                  **(blade_kw or {}))
            k += 1
    return k


def seedlings(sh, blobs, depth, to_page=None, rng=None, seed=41, leaf_seed=900, reflection_seed=5000):
    """Rice seedlings (or any tuft standing in water) drawn from a mask of small blobs, one per plant.

    Each plant: 2-5 leaves fanning up from the base, as wide as its blob; plants taller than 8 px get two or
    three short broken lines below for the reflection. depth(base_y) in 0..1. Returns the number of strokes.
    """
    tp = to_page or _ident
    rng = rng if rng is not None else np.random.default_rng(seed)
    lab, n = nd.label(blobs)
    k = 0
    for i, sl in enumerate(nd.find_objects(lab), 1):
        ys, xs = np.nonzero(lab[sl] == i)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        top, base_y = ys.min(), ys.max()
        h = base_y - top + 1
        if h < 3:
            continue
        bx = xs[ys >= base_y - max(1, h * 0.15)].mean()
        w = xs.max() - xs.min() + 1
        near = float(np.clip(depth(base_y), 0, 1))
        nleaf = 2 + (h > 10) + (h > 22) + (h > 40)
        spread = np.clip(0.6 * w / h, 0.25, 0.9)
        for j in range(nleaf):
            t = (j + 0.5) / nleaf * 2 - 1
            a = t * spread + rng.normal(0, 0.08)
            ln = h * rng.uniform(0.75, 1.05) * (1 - 0.25 * abs(t))
            tip = (bx + np.sin(a) * ln, base_y - np.cos(a) * ln)
            mid = (bx + np.sin(a * 0.6) * ln * 0.5, base_y - np.cos(a * 0.6) * ln * 0.5)
            pts = [tp(bx + rng.normal(0, 0.3), base_y), tp(*mid), tp(*tip)]
            sh.stroke(pts, pen='g_pen', pressure=0.25 + 0.4 * near, taper=(0.1, 1.2 + 1.5 * near), seed=leaf_seed + k)
            k += 1
        if h > 8:
            for j in range(2 + (h > 25)):
                a = (j - 1) * spread * 0.7
                ln = h * rng.uniform(0.35, 0.55)
                pts = [tp(bx + np.sin(a) * 2, base_y + 2), tp(bx + np.sin(a) * ln, base_y + np.cos(a) * ln)]
                sh.stroke(pts, pen='maru', pressure=0.35, taper=(0.3, 1.0), seed=reflection_seed + k)
                k += 1
    return k
