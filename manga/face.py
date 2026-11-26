"""Expression parts drawn by hand, for changing a face from the reference (closed eyes, a closed smile).

Points are in the reference's own pixels; `to_page(x, y)` maps them to the page (a Frame's P works). Erase what
they replace first (paint the eye or mouth area over with the skin's own tone), then draw these on top.

    face.closed_eye(sh, inner=(538, 387), outer=(615, 395), to_page=fr.P, sag=10)
    face.closed_smile(sh, [(455, 459), (480, 470), (518, 462)], to_page=fr.P)
"""
import numpy as np

from . import geom


def _frame(inner, outer):
    inner, outer = np.asarray(inner, float), np.asarray(outer, float)
    d = outer - inner
    L = float(np.hypot(*d))
    t = d / L
    n = np.array([-t[1], t[0]])
    if n[1] < 0:
        n = -n                                    # "down" on the face, whichever way the head tilts
    return inner, d, t, n


def closed_eye(sh, inner, outer, to_page, sag=8.0, thick=(1.0, 4.6), lashes=3, lash_len=10.0, crease=9.0,
               seed=32, widen=4.0):
    """A content closed eye: the lid bows downward like a soft smile from the inner corner to the outer one, a
    lash band thickening toward the outer corner, a few lashes falling down and outward there, and a thin
    broken crease above. `sag`, `thick`, `lash_len` and `crease` are in ref px. Returns the lid line (page px)."""
    rng = np.random.default_rng(seed)
    inner, d, t, n = _frame(inner, outer)
    if widen:
        d = d + t * widen
    u = np.linspace(0, 1, 40)
    base = inner + u[:, None] * d + (sag * np.sin(np.pi * u ** 0.9))[:, None] * n
    th = (thick[0] + (thick[1] - thick[0]) * u ** 1.3) * (1 - 0.6 * np.clip((u - 0.9) / 0.1, 0, 1))
    band = np.vstack([base - 0.5 * th[:, None] * n, (base + 0.5 * th[:, None] * n)[::-1]])
    sh.fill([to_page(x, y) for x, y in band])
    line = np.array([to_page(x, y) for x, y in base])
    sh.stroke(np.column_stack([line, 0.35 + 0.5 * u]), pen='g_pen', taper=(0.8, 0.3))
    for k in range(lashes):
        f = 0.78 + 0.21 * k / max(lashes - 1, 1)
        ang = 0.55 + 0.6 * k / max(lashes - 1, 1)
        ln = lash_len * (0.75 + 0.25 * k / max(lashes - 1, 1))
        i = int(f * 39)
        p = base[i] + 0.4 * th[i] * n
        dv = np.cos(ang) * t + np.sin(ang) * n
        sh.stroke([to_page(*p), to_page(*(p + dv * ln * 0.55 + t * 0.8)), to_page(*(p + dv * ln + t * 2.0))],
                  pen='g_pen', pressure=0.6, taper=(0.0, 1.2), seed=int(rng.integers(1 << 30)))
    if crease:
        cu = np.linspace(0.18, 0.86, 20)
        cr = inner + cu[:, None] * d + (sag * 0.6 * np.sin(np.pi * cu) - crease)[:, None] * n
        sh.stroke(np.column_stack([[to_page(x, y) for x, y in cr], np.full(len(cr), 0.3)]), pen='maru',
                  taper=(1.0, 1.4))
    return line


def closed_smile(sh, pts, to_page, tucks=4.0, lower_lip=True):
    """A closed smile through `pts` (corner, ..., corner; ref px): one line, heavier in the middle, a small tuck
    up at each corner, and a short hint of the lower lip under the middle."""
    P = np.array([to_page(x, y) for x, y in pts], float)
    sm = geom.catmull(P, per=8)
    u = np.linspace(0, 1, len(sm))
    sh.stroke(np.column_stack([sm, 0.45 + 0.35 * np.sin(np.pi * u)]), pen='g_pen', taper=(0.6, 0.6))
    a, b = np.asarray(pts[0], float), np.asarray(pts[-1], float)
    for c, sgn in ((a, -1), (b, 1)):
        sh.stroke([to_page(*c), to_page(c[0] + sgn * tucks, c[1] - tucks)], pen='maru', pressure=0.4,
                  taper=(0.2, 1.0), smooth=False)
    if lower_lip:
        mid = np.asarray(pts[len(pts) // 2], float)
        w = 0.13 * float(np.hypot(*(b - a)))
        sh.stroke([to_page(mid[0] - w, mid[1] + 11), to_page(mid[0], mid[1] + 13), to_page(mid[0] + w, mid[1] + 12)],
                  pen='maru', pressure=0.3, taper=(1.0, 1.0))
