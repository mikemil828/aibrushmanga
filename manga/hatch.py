"""Hatching and cross-hatching: pen strokes laid in parallel rows, densest where the tone map is darkest.

Every hatch line is an ordinary pen stroke (taper, pressure, wobble), journaled one by one, so a hatched area
replays and scales like hand inking. Layers stack the way an inker builds value: first one direction, then a
cross direction only where it must get darker, then a third.
"""
import numpy as np


def _sample(a, x, y):
    H, W = a.shape
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return np.where(inside, a[yi, xi], 0.0)


def hatch(sheet, tone, mask=None, pen='maru', angle=45.0, spacing_mm=0.7,
          layers=((0.2, 0.0), (0.45, 90.0), (0.7, 45.0)),
          pressure=(0.25, 0.85), stroke_mm=(5.0, 11.0), min_mm=1.2, jitter=0.12, angle_jitter=1.2,
          bow=0.15, seed=0, field=None):
    """Hatch `mask` (bool array or polygon, base px) following a darkness map `tone` (0 = paper, 1 = black).

    angle   -- degrees, direction of the first layer (0 = horizontal, 90 = vertical, y down).
    layers  -- (threshold, extra angle) per layer: the layer is drawn only where tone > threshold.
    pressure-- pressure range mapped from how far tone goes past the threshold (thicker lines = darker).
    stroke_mm -- a hand stroke is this long (random in range); long runs are broken with small overlaps/gaps.
    field   -- optional array of angles (degrees, base px): lines bend to follow it (curved hatching).
    Returns the number of strokes laid.
    """
    rng = np.random.default_rng(seed)
    H, W = sheet.H0, sheet.W0
    if mask is None:
        m = np.ones((H, W), bool)
    elif isinstance(mask, (list, tuple)):
        from .sheet import _poly_mask
        m = _poly_mask(mask, W, H) > 0.5
    else:
        m = np.asarray(mask, bool)
    T = np.broadcast_to(np.asarray(tone, np.float32), (H, W))
    ppm = sheet.ppm0
    sp = spacing_mm * ppm
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return 0
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    R = 0.5 * np.hypot(bx1 - bx0, by1 - by0) + 2
    n = 0
    for li, (thr, extra) in enumerate(layers):
        th = np.deg2rad(angle + extra)
        d = np.array([np.cos(th), np.sin(th)])
        nv = np.array([-d[1], d[0]])
        offs = np.arange(-R, R, sp)
        offs = offs + rng.uniform(-0.5, 0.5) * sp
        for o in offs:
            o2 = o + rng.normal(0, jitter * sp)
            if field is None:
                ang = np.deg2rad(rng.normal(0, angle_jitter))
                dd = np.array([np.cos(th + ang), np.sin(th + ang)])
                t = np.arange(-R, R, 1.0)
                px = cx + o2 * nv[0] + t * dd[0]
                py = cy + o2 * nv[1] + t * dd[1]
            else:
                px, py = _streamline(field, m, cx + o2 * nv[0], cy + o2 * nv[1], extra, R)
                if px is None:
                    continue
            inside = _sample(m.astype(np.float32), px, py) > 0.5
            tv = _sample(T, px, py)
            on = inside & (tv > thr)
            n += _runs_to_strokes(sheet, px, py, on, tv, thr, pen, pressure, stroke_mm, min_mm, bow, rng, ppm)
    return n


def _runs_to_strokes(sheet, px, py, on, tv, thr, pen, pressure, stroke_mm, min_mm, bow, rng, ppm):
    n = 0
    if not on.any():
        return 0
    e = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    starts, ends = np.nonzero(e == 1)[0], np.nonzero(e == -1)[0]
    for a, b in zip(starts, ends):
        if (b - a) < min_mm * ppm:
            continue
        i = a
        while i < b:
            L = int(rng.uniform(*stroke_mm) * ppm)
            j = min(b, i + L)
            if b - j < min_mm * ppm * 0.8:
                j = b
            seg = slice(i, j)
            sx, sy, st = px[seg], py[seg], tv[seg]
            if len(sx) >= 2:
                k = max(2, min(5, len(sx) // 25 + 2))
                idx = np.linspace(0, len(sx) - 1, k).round().astype(int)
                depth = np.clip((st[idx] - thr) / max(1 - thr, 1e-3), 0, 1)
                pr = pressure[0] + (pressure[1] - pressure[0]) * depth ** 0.7
                pts = np.stack([sx[idx], sy[idx], pr], 1)
                if bow and len(idx) > 2:   # a hand stroke arcs slightly
                    dx, dy = sx[idx[-1]] - sx[idx[0]], sy[idx[-1]] - sy[idx[0]]
                    Ln = max(np.hypot(dx, dy), 1e-6)
                    nx, ny = -dy / Ln, dx / Ln
                    tt = np.linspace(0, 1, len(idx))
                    amt = bow * ppm * rng.uniform(-1, 1) * 4 * tt * (1 - tt)
                    pts[:, 0] += nx * amt
                    pts[:, 1] += ny * amt
                sheet.stroke(pts, pen=pen, seed=int(rng.integers(1 << 30)))
                n += 1
            gap = int(rng.normal(0.0, 0.35) * ppm)   # small overlap or gap between strokes
            i = j + gap if j < b else b
            i = max(i, a)
            if j >= b:
                break
    return n


def _streamline(field, m, x, y, extra, R, step=1.0):
    F = np.asarray(field, np.float32)
    H, W = F.shape
    pts_f, pts_b = [], []
    for sign, out in ((1, pts_f), (-1, pts_b)):
        px, py = x, y
        prev = None
        for _ in range(int(2 * R)):
            if not (0 <= px < W and 0 <= py < H):
                break
            a = np.deg2rad(F[int(py), int(px)] + extra)
            v = np.array([np.cos(a), np.sin(a)]) * sign
            if prev is not None and v @ prev < 0:
                v = -v
            prev = v
            px, py = px + v[0] * step, py + v[1] * step
            out.append((px, py))
    pts = pts_b[::-1] + [(x, y)] + pts_f
    if len(pts) < 3:
        return None, None
    P = np.array(pts)
    return P[:, 0], P[:, 1]
