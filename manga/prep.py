"""Reading a reference photo: value maps, direction fields, and inking the outline of a shape you chose.

These are measuring tools, not a filter. You decide which shapes get a line, which areas get black, tone or
hatching; these helpers only give you the numbers to do it with.
"""
import numpy as np
from scipy import ndimage as nd
from PIL import Image

from . import geom
from .settings import configurable


def load(path, width, height):
    im = Image.open(path).convert('RGB').resize((width, height), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def darkness(rgb, blur=1.5, gamma=1.0, lo=0.0, hi=1.0, mode='lightness', mix=0.5):
    """0 = white paper, 1 = black: perceptual lightness inverted, stretched between lo..hi and gamma-shaped.

    mode='value' blends in the HSV value (max channel), so saturated colours (a red shirt, red seats) read as
    mid-tones the way an inker sees them, and only real shadows go dark. `mix` = share of value in the blend.
    """
    if mode == 'value':
        v = 1.0 - rgb.max(axis=2)
        dl = darkness(rgb, blur=0, gamma=1.0)
        d = (1 - mix) * dl + mix * v
        if blur:
            d = nd.gaussian_filter(d, blur)
        return (np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma).astype(np.float32)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    L = np.where(Y > 0.008856, 116 * np.cbrt(Y) - 16, 903.3 * Y) / 100.0
    d = 1.0 - L
    if blur:
        d = nd.gaussian_filter(d, blur)
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    return d.astype(np.float32)


def orientation(gray, sigma=6.0):
    """Dominant stroke direction in degrees (along edges, not across) from the structure tensor."""
    gy, gx = np.gradient(nd.gaussian_filter(gray, 1.0))
    jxx = nd.gaussian_filter(gx * gx, sigma)
    jyy = nd.gaussian_filter(gy * gy, sigma)
    jxy = nd.gaussian_filter(gx * gy, sigma)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)       # gradient direction
    return (np.rad2deg(ang) + 90.0).astype(np.float32)  # along the edge


def smooth_mask(mask, sigma=1.5):
    return nd.gaussian_filter(np.asarray(mask, np.float32), sigma)


@configurable('prep.ink_outline')
def ink_outline(sheet, mask, pen='g_pen', pressure=(0.35, 0.85), light=(-1.0, -1.0), sigma=2.0, tol=1.2,
                min_len=10.0, max_seg_mm=40.0, seed=0, holes=True, mode='ink'):
    """Ink the outline of a shape (bool mask, base px) with weight that follows the light.

    The side facing away from `light` (a direction, default from the upper left) gets the heavier line, the lit
    side the thinner one; long contours are broken into hand-length strokes that overlap a little.
    """
    rng = np.random.default_rng(seed)
    lv = np.asarray(light, float)
    lv = lv / np.hypot(*lv)
    f = smooth_mask(mask, sigma)
    lines = geom.outlines(f, 0.5, min_len)
    n = 0
    ppm = sheet.ppm0
    maxL = max_seg_mm * ppm
    for P in lines:
        P = geom.simplify(P, tol)
        if len(P) < 2:
            continue
        D, s = geom.resample(P, 2.0)
        if len(D) < 3:
            continue
        nrm = geom.normals(D)
        # make the normal point outward (toward lower mask value)
        probe = D + nrm * 2
        xi = np.clip(probe[:, 0].astype(int), 0, f.shape[1] - 1)
        yi = np.clip(probe[:, 1].astype(int), 0, f.shape[0] - 1)
        flip = np.where(f[yi, xi] > 0.5, -1.0, 1.0)
        nrm = nrm * flip[:, None]
        shade = np.clip(0.5 - 0.5 * (nrm @ lv), 0, 1)      # 1 = facing away from light
        shade = nd.gaussian_filter1d(shade, 6, mode='nearest')
        pr = pressure[0] + (pressure[1] - pressure[0]) * shade
        L = s[-1]
        i = 0
        while i < len(D) - 2:
            seg_len = rng.uniform(0.6, 1.0) * maxL
            j = int(np.searchsorted(s, s[i] + seg_len))
            j = min(j, len(D) - 1)
            if s[-1] - s[j] < 0.25 * maxL:
                j = len(D) - 1
            idx = np.unique(np.linspace(i, j, max(3, (j - i) // 6)).round().astype(int))
            pts = np.stack([D[idx, 0], D[idx, 1], pr[idx]], 1)
            sheet.stroke(pts, pen=pen, seed=int(rng.integers(1 << 30)), mode=mode)
            n += 1
            if j >= len(D) - 1:
                break
            i = max(j - int(rng.uniform(1.0, 2.5) * ppm / 2.0), i + 1)
    return n


def xdog_lines(rgb, sigma=1.2, k=1.6, p=20.0, eps=-0.02, phi=60.0, tidy=8):
    """Extended difference-of-Gaussians line map (0..1 ink): contour guides for figures and faces.

    Use it as a drawing aid (`sheet.apply(lines)` as a pencil-like layer, or threshold and outline it), not as
    the drawing: it finds every edge in the photo, and an inker picks few.
    """
    g = darkness(rgb, blur=0)
    g = 1.0 - g
    a = nd.gaussian_filter(g, sigma)
    b = nd.gaussian_filter(g, sigma * k)
    s = (1 + p) * a - p * b
    e = np.where(s >= eps, 1.0, 1.0 + np.tanh(phi * (s - eps)))
    ink = np.clip(1.0 - e, 0, 1).astype(np.float32)
    if tidy:
        m = ink > 0.5
        lab, n = nd.label(m)
        if n:
            sz = nd.sum(m, lab, range(1, n + 1))
            ink = ink * np.isin(lab, 1 + np.nonzero(sz >= tidy)[0])
    return ink


def ink_contour(sheet, field, level=0.5, min_len=12, keep=None, min_run=4, tol=0.6, simplify_first=False,
                piece=40, overlap=2, pen='maru', pressure=0.45, style=None, to_page=None, seed=0):
    """Ink selected stretches of a field's iso-lines as hand-length strokes.

    The common case: a ridge line where sky meets mountain, the lit underside of a cloud, the top edge of a
    grove. `keep(points) -> bool per point` picks the stretches (None keeps all); runs shorter than min_run
    points are dropped; long runs are cut into pieces of `piece` points overlapping by `overlap`.
    simplify_first -- simplify the whole line before selecting (use it when keep looks at the line's normals).
    style(points) -> (pen, pressure) per run, to vary the weight (e.g. heavier on near ridges).
    Coordinates are the field's own; `to_page` maps them onto the sheet. Returns the number of strokes.
    """
    tp = to_page or (lambda x, y: (x, y))
    n = 0
    for Pl in geom.outlines(field, level, min_len):
        base = geom.simplify(Pl, tol) if simplify_first else Pl
        if len(base) < 2:
            continue
        ok = np.ones(len(base), bool) if keep is None else np.asarray(keep(base), bool)
        runs = np.split(np.arange(len(base)), np.nonzero(np.diff(ok.astype(int)))[0] + 1)
        for run in runs:
            if not ok[run[0]] or len(run) < min_run:
                continue
            Q = base[run] if simplify_first else geom.simplify(Pl[run], tol)
            pn, pr = style(Q) if style else (pen, pressure)
            pts = [tp(x, y) for x, y in Q[:, :2]]
            for k in range(0, len(pts) - 1, piece):
                seg = pts[k:k + piece + overlap]
                if len(seg) > 1:
                    sheet.stroke(seg, pen=pn, pressure=pr, seed=seed + n)
                    n += 1
    return n
