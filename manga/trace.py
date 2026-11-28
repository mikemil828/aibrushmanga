"""From a line drawing to pen strokes, and from closed areas to flat stock tones.

`ink_line_art` thins a line mask to centre lines, follows them into paths, reads each line's width from the mask,
and lays every path as one pen stroke with matching pressure. Ends that stop in the open get the pen's taper;
ends that meet another line stay blunt, as an inker would leave them.

`area_tones` picks one stock tone per closed area (the spaces between the lines), plus at most one darker shade and
one scratched highlight inside it. That is how screentone is cut and laid by hand: by area, not per pixel.
"""
import numpy as np
from scipy import ndimage as nd

from .settings import configurable

from . import geom

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def thin(mask):
    """Zhang-Suen thinning: one-pixel-wide centre lines of a bool mask."""
    m = np.asarray(mask, bool).astype(np.uint8).copy()
    while True:
        changed = False
        for step in (0, 1):
            P = np.pad(m, 1)
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rem = (m == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rem.any():
                m[rem] = 0
                changed = True
        if not changed:
            return m.astype(bool)


def trace(skel):
    """Follow a one-pixel skeleton into paths. Returns [(points (N, 2) as (x, y), open_start, open_end)].

    open_* is True when that end is a free end (nothing else touches it), False when it stops at a junction.
    """
    S = np.pad(np.asarray(skel, bool), 1)
    H, W = S.shape
    flat = S.ravel()
    offs = np.array([dy * W + dx for dy, dx in _N8])
    idx = np.nonzero(flat)[0]
    nb = {}
    # a diagonal step counts only when no orthogonal pixel already joins the two: otherwise every staircase
    # corner looks like a junction and lines fall apart into 2-pixel pieces
    for i in idx.tolist():
        nb[i] = [i + o for (dy, dx), o in zip(_N8, offs.tolist()) if flat[i + o]
                 and not (dy and dx and (flat[i + dy * W] or flat[i + dx]))]
    deg = {i: len(v) for i, v in nb.items()}
    node = {i for i, k in deg.items() if k != 2}
    seen_edge = set()
    paths = []

    def walk(start, nxt):
        path = [start]
        prev, cur = start, nxt
        while True:
            path.append(cur)
            if cur in node:
                break
            cand = [j for j in nb[cur] if j != prev]
            if not cand:
                break
            prev, cur = cur, cand[0]
            if cur == start:
                path.append(cur)
                break
        return path

    for s in node:
        for n in nb[s]:
            if (s, n) in seen_edge:
                continue
            p = walk(s, n)
            for a, b in zip(p[:-1], p[1:]):
                seen_edge.add((a, b))
                seen_edge.add((b, a))
            paths.append((p, deg[s] == 1, deg[p[-1]] == 1))
    # pure loops (no nodes)
    used = set()
    for p, _, _ in paths:
        used.update(p)
    for i in idx.tolist():
        if i in used or i in node:
            continue
        n = nb[i][0]
        p = walk(i, n)
        used.update(p)
        paths.append((p, False, False))
    out = []
    for p, a, b in paths:
        arr = np.array(p)
        ys, xs = arr // W - 1, arr % W - 1
        out.append((np.stack([xs, ys], 1).astype(float), a, b))
    return out


def ink_line_art(sheet, lines, pens=('maru', 'g_pen'), split_mm=0.25, min_len_px=4.0, beta_px=3.0, tol=0.6,
                 seed=0, weight=1.3, min_px=1.3, even=0.0):
    """Draw a line mask (bool, base px) as pen strokes.

    pens     -- (thin pen, thick pen): lines narrower than split_mm get the first.
    beta_px  -- parts of the mask thicker than 2*beta_px are solid black areas, not lines: they are filled, not
                stroked. Returns (strokes, beta_mask).
    even     -- 0..1: how far each line's width is pulled to its own median. The traced mask swells where
                lines meet or blur; an inker's hand keeps a line's weight and changes it gradually.
    """
    from .pens import get as get_pen
    rng = np.random.default_rng(seed)
    m = np.asarray(lines, bool)
    dist = nd.distance_transform_edt(m)
    beta = nd.binary_dilation(dist > beta_px, iterations=int(np.ceil(beta_px)))
    beta &= m
    lines_only = m & ~nd.binary_erosion(beta, iterations=1)
    skel = thin(lines_only)
    ppm = sheet.ppm0
    n = 0
    for P, open_a, open_b in trace(skel):
        if len(P) < 2 or geom.arclen(P)[-1] < min_len_px:
            continue
        Q = geom.simplify(P, tol)
        if len(Q) > 2 or geom.arclen(Q)[-1] > 2:
            Qd, _ = geom.resample(P, 3.0)
            r = dist[np.clip(Qd[:, 1].round().astype(int), 0, m.shape[0] - 1),
                     np.clip(Qd[:, 0].round().astype(int), 0, m.shape[1] - 1)]
            w_px = np.maximum((2 * r - 1.0) * weight, min_px)
            w_mm = nd.uniform_filter1d(w_px, 3) / ppm
            if even > 0:
                med = np.median(w_mm)
                sm = nd.gaussian_filter1d(np.clip(w_mm, 0.75 * med, 1.25 * med), 4, mode='nearest')
                w_mm = (1 - even) * w_mm + even * sm
            pen = get_pen(pens[1] if np.median(w_mm) > split_mm else pens[0])
            p = np.clip((w_mm - pen.min_mm) / max(pen.max_mm - pen.min_mm, 1e-6), 0, 1) ** (1 / pen.gamma)
            pts = np.column_stack([Qd, np.clip(p, 0.05, 1.0)])
            L = geom.arclen(Qd)[-1] / ppm
            tin = min(pen.taper_in_mm, 0.35 * L) if open_a else 0.0
            tout = min(pen.taper_out_mm, 0.45 * L) if open_b else 0.0
            sheet.stroke(pts, pen=pen, taper=(tin, tout), seed=int(rng.integers(1 << 30)))
            n += 1
    if beta.any():
        sheet.fill(beta)
    return n, beta


STOCK = ((0.10, 0.0), (0.26, 0.12), (0.42, 0.25), (0.58, 0.40), (0.74, 0.55), (0.86, 0.72), (2.0, 1.0))


def _stock(v, stock):
    for upto, lv in stock:
        if v < upto:
            return lv
    return stock[-1][1]


@configurable('trace.area_tones')
def area_tones(darkness, barrier, stock=STOCK, shade=0.14, light=0.16, min_area=25, blur=1.5):
    """A tone level per pixel built area by area.

    darkness -- 0 paper .. 1 black (base px).  barrier -- bool: the ink lines and black areas that bound areas.
    Each area gets the stock tone nearest its median darkness; inside it, parts more than `shade` darker get the
    next stock tone (the shadow tone), parts more than `light` lighter get scratched to paper (highlights).
    Small specks between lines take the stock tone of their own darkness.
    """
    d = nd.gaussian_filter(np.asarray(darkness, np.float32), blur)
    free = ~np.asarray(barrier, bool)
    lab, n = nd.label(free)
    levels = np.zeros(d.shape, np.float32)
    if n == 0:
        return levels
    med = nd.median(d, lab, index=np.arange(1, n + 1))
    area = nd.sum(free, lab, index=np.arange(1, n + 1))
    lvl = np.array([_stock(v, stock) for v in med], np.float32)
    nxt = np.array([_stock(min(v + 0.17, 1.99), stock) for v in med], np.float32)
    big = area >= min_area
    base = np.concatenate([[0.0], np.where(big, lvl, 0)])[lab]
    up = np.concatenate([[0.0], np.where(big, nxt, 0)])[lab]
    medmap = np.concatenate([[0.0], med])[lab]
    bigmap = np.concatenate([[False], big])[lab]
    dark_part = nd.gaussian_filter(((d - medmap) > shade).astype(np.float32), 1.5) > 0.5
    light_part = nd.gaussian_filter(((medmap - d) > light).astype(np.float32), 1.5) > 0.5
    levels = np.where(bigmap, np.where(dark_part, up, np.where(light_part, 0.0, base)), 0.0)
    small = free & ~bigmap
    if small.any():
        sv = np.vectorize(lambda v: _stock(v, stock), otypes=[np.float32])(d[small])
        levels[small] = sv
    # under the lines themselves: the tone of the nearest area, so no paper halo shows beside a line
    idx = nd.distance_transform_edt(~free, return_distances=False, return_indices=True)
    levels = levels[tuple(idx)]
    return levels.astype(np.float32)
