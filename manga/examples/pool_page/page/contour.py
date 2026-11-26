"""The body's outer contour: one even G-pen line round the silhouette, heavier on the shadow side.

Weight changes only gradually (smoothed over several mm), strokes are hand-length and overlap with short tapers,
so there is no sudden thick-to-thin jump. Places where two strokes meet are buried under each other.
"""
import numpy as np
from scipy import ndimage as nd
from manga import geom


def draw_contour(sh, fig, box, s, base_mm=0.42, swing=0.35, light=(-1.0, -1.0), seg_mm=(28, 45), seed=12,
                 pen='g_pen', min_len=40):
    X0, Y0 = box[0], box[1]
    rng = np.random.default_rng(seed)
    f = nd.gaussian_filter(fig.astype(np.float32), 2.0 * s)
    lv = np.asarray(light, float); lv /= np.hypot(*lv)
    from manga.pens import get
    P_ = get(pen)
    ppm = sh.ppm0
    n = 0
    for P in geom.outlines(f, 0.5, min_len * s):
        P = geom.simplify(P, 0.6)
        if len(P) < 3:
            continue
        D, sarc = geom.resample(P, 2.0)
        if len(D) < 4:
            continue
        nrm = geom.normals(D)
        probe = D + nrm * 2
        xi = np.clip(probe[:, 0].astype(int), 0, f.shape[1] - 1); yi = np.clip(probe[:, 1].astype(int), 0, f.shape[0] - 1)
        nrm = nrm * np.where(f[yi, xi] > 0.5, -1.0, 1.0)[:, None]
        shade = np.clip(0.5 - 0.5 * (nrm @ lv), 0, 1)
        shade = nd.gaussian_filter1d(shade, 6 * ppm / 2.0, mode='wrap')        # ~6 mm: weight changes slowly
        w = base_mm * (1 - swing + 2 * swing * shade)
        pr = np.clip((w - P_.min_mm) / (P_.max_mm - P_.min_mm), 0, 1) ** (1 / P_.gamma)
        i = 0
        while i < len(D) - 2:
            j = min(int(np.searchsorted(sarc, sarc[i] + rng.uniform(*seg_mm) * ppm)), len(D) - 1)
            if sarc[-1] - sarc[j] < 10 * ppm:
                j = len(D) - 1
            idx = np.unique(np.linspace(i, j, max(3, (j - i) // 5)).round().astype(int))
            pts = np.column_stack([D[idx, 0] + X0, D[idx, 1] + Y0, pr[idx]])
            sh.stroke(pts, pen=pen, taper=(0.6, 0.9), seed=int(rng.integers(1 << 30)))
            n += 1
            if j >= len(D) - 1:
                break
            i = max(j - int(1.6 * ppm / 2.0), i + 1)                      # overlap ~1.6 mm
    return n
