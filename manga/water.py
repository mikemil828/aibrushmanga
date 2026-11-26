"""Still water in ink: calm line tone, the water level carried under things standing in it, mirrored
reflections, and broken ripple lines.

Use line tone with angle=90 for water (horizontal lines); angle=0 gives vertical lines, which read as rain.
"""
import numpy as np
from scipy import ndimage as nd

HORIZONTAL = 90.0


def fill_under(L, water, objects, sigma=4.0, grow=2):
    """The lightness of the water around each object, carried under it (a blur normalized over water pixels).

    Without this, seedlings or posts standing in bright water leave dark halos in the water tone.
    Returns (L with the objects' pixels replaced, the grown object mask)."""
    grown = nd.binary_dilation(objects, iterations=grow) if grow else np.asarray(objects, bool)
    ww = (np.asarray(water, bool) & ~grown).astype(np.float32)
    Lw = nd.gaussian_filter(L * ww, sigma) / np.maximum(nd.gaussian_filter(ww, sigma), 1e-3)
    return np.where(grown, Lw, L), grown


def calm_level(Lw, light=0.78, gain=1.0, cap=0.5, sigma=(2.5, 6.0), floor=0.04):
    """Water tone from water lightness, smoothed much more along the rows than across them, so nothing standing
    in the water prints into it. Levels under `floor` become bare paper."""
    lv = np.clip(gain * (light - nd.gaussian_filter(Lw, sigma)), 0, cap)
    return np.where(lv < floor, 0.0, lv)


def mirror(src, src_line, dst_line, squash, rows):
    """A reflection mask: `src` (bool) turned upside down about its base line `src_line` and laid in the water
    from `dst_line` down, foreshortened by `squash` (about 0.5 for a paddy seen from a standing height).
    rows -- (y0, y1): the rows to fill. Clip the result to the water yourself."""
    src = np.asarray(src, bool)
    out = np.zeros_like(src)
    ys = np.arange(*rows)
    sy = np.round(src_line - (ys - dst_line) / squash).astype(int)
    ok = (sy >= 0) & (sy < src.shape[0])
    out[ys[ok]] = src[sy[ok]]
    return out


def ripples(sh, mask, rows, to_page=None, rng=None, spacing=(8.0, 12.0), tighten=3.0, dash=(6.0, 22.0),
            gap=(10.0, 26.0), pressure=0.08, seed=9000):
    """Broken horizontal white lines across a reflection (or any dark water), closer together toward the viewer.
    rows -- (first row, last row). Returns the number of strokes."""
    tp = to_page or (lambda x, y: (x, y))
    rng = rng if rng is not None else np.random.default_rng(seed)
    y0, y1 = rows
    y = float(y0)
    k = 0
    while y < y1:
        xs = np.nonzero(mask[int(y)])[0]
        if len(xs):
            x = xs.min()
            while x < xs.max():
                ln = rng.uniform(*dash)
                sh.stroke([tp(x, y), tp(x + ln, y)], pen='white', mode='white', pressure=pressure, taper=(0.5, 0.5),
                          seed=seed + k)
                k += 1
                x += ln + rng.uniform(*gap)
        y += rng.uniform(*spacing) - tighten * (y - y0) / max(y1 - y0, 1)
    return k
