"""Screentone: dot, line and noise screens laid over a mask, as if cut from a tone sheet with a knife.

A tone sheet has a fixed grid. Two calls with the same lpi/angle/phase line up exactly, like one sheet of
tone cut into pieces. `level` (0-1) is the printed ink fraction: 0.1 = light 10 % dots, 0.5 = checkerboard,
0.8 = white holes in black. `lpi` is lines per inch, a physical size, so the dots keep their size in mm at any
scale. Classic tones: 50-65 lpi. On a screen-size draft (8 px/mm) 27-42 lpi reads better.
"""
import numpy as np

_LUT = {}


def _spot_lut(kind):
    if kind in _LUT:
        return _LUT[kind]
    g = (np.arange(256) + 0.5) / 256
    u, v = np.meshgrid(g, g)
    f = _spot(kind, u, v).ravel()
    f.sort()
    _LUT[kind] = f
    return f


def _spot(kind, u, v):
    if kind == 'dot':
        return 0.5 * (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v))
    if kind == 'line':
        return np.cos(2 * np.pi * u)
    raise ValueError(kind)


def _threshold(kind, level):
    f = _spot_lut(kind)
    q = np.clip(1.0 - np.asarray(level, np.float32), 0, 1)
    return np.interp(q, np.linspace(0, 1, len(f)), f).astype(np.float32)


def screen(level, shape, px_per_mm, lpi=42.0, angle=45.0, kind='dot', phase=(0.0, 0.0), origin=(0, 0), k=1.0,
           snap=False):
    """Ink coverage of a screen for a level map (scalar or array the size of `shape`).

    snap -- for 1-bit print: the screen is first nudged to a whole-pixel cell (snap_screen), and each
            pixel of the cell gets its own rank, so a level inks the same pixels in every cell (a
            threshold matrix). Coverage is then 0 or 1 and steps by one pixel per cell.
    """
    if snap:
        lpi, angle = snap_screen(lpi, angle, px_per_mm)
    H, W = shape
    period = 25.4 / lpi * px_per_mm
    a = np.deg2rad(angle)
    ca, sa = np.cos(a), np.sin(a)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    X += 0.5 + origin[0] * k
    Y += 0.5 + origin[1] * k
    u = (X * ca + Y * sa) / period + phase[0]
    v = (-X * sa + Y * ca) / period + phase[1]
    if snap:
        return _ranked(kind, u, v, level)
    f = _spot(kind, u, v).astype(np.float32)
    if kind == 'dot':
        g = (np.pi / period) * np.sqrt(np.sin(2 * np.pi * u) ** 2 + np.sin(2 * np.pi * v) ** 2)
    else:
        g = (2 * np.pi / period) * np.abs(np.sin(2 * np.pi * u))
    g = np.maximum(g, 0.35 * np.pi / period).astype(np.float32)   # flat spots of the spot function
    lv = np.asarray(level, np.float32)
    T = _threshold(kind, lv)
    cov = np.clip((f - T) / g + 0.5, 0, 1)
    cov = np.where(lv <= 0.003, 0.0, np.where(lv >= 0.997, 1.0, cov))
    return cov.astype(np.float32)


def _ranked(kind, u, v, level):
    # cell position of each pixel, rounded so that every cell repeats exactly
    ku = np.round(np.mod(u, 1.0) * 4096).astype(np.int64) % 4096
    kv = np.round(np.mod(v, 1.0) * 4096).astype(np.int64) % 4096
    if kind == 'line':
        kv[:] = 0                                   # a line inks whole rows at once
    key = ku * 4096 + kv
    uk, inv = np.unique(key, return_inverse=True)
    fu, fv = (uk // 4096) / 4096.0, (uk % 4096) / 4096.0
    f = _spot(kind, fu, fv)
    tie = ((uk * 2654435761) % 1000003) / 1000003.0   # fixed order among equal spot values
    order = np.lexsort((tie, -f))                     # highest spot value inks first
    rank = np.empty(len(uk), np.float64)
    rank[order] = (np.arange(len(uk)) + 0.5) / len(uk)
    thr = rank[inv.reshape(u.shape)]
    lv = np.asarray(level, np.float32)
    return (thr < lv).astype(np.float32)


def snap_screen(lpi, angle, px_per_mm):
    """Nudge a screen so its cell steps a whole number of pixels: (lpi, angle) closest to the asked ones.

    For 1-bit print output. A cell that steps a fractional number of pixels thresholds into dots of
    slowly varying shape, which a laser printer shows as a woven pattern; a whole-pixel cell repeats
    exactly. At 600 dpi a 60 lpi 45-degree screen becomes 60.6 lpi, a change no one can see.
    """
    period = 25.4 / lpi * px_per_mm
    a = np.deg2rad(angle)
    p, q = int(round(period * np.cos(a))), int(round(period * np.sin(a)))
    if p == 0 and q == 0:
        return lpi, angle
    return 25.4 * px_per_mm / float(np.hypot(p, q)), float(np.degrees(np.arctan2(q, p)))


def noise_screen(level, shape, px_per_mm, grain_mm=0.12, seed=0):
    """Sand / stipple tone: random grains, density = level."""
    H, W = shape
    rng = np.random.default_rng(seed)
    gpx = max(grain_mm * px_per_mm, 1.0)
    h, w = int(np.ceil(H / gpx)) + 1, int(np.ceil(W / gpx)) + 1
    r = rng.random((h, w)).astype(np.float32)
    yy = (np.arange(H) / gpx).astype(int)
    xx = (np.arange(W) / gpx).astype(int)
    r = r[yy][:, xx]
    return (r < np.asarray(level, np.float32)).astype(np.float32)


def tone(sheet, mask=None, level=0.3, lpi=42.0, angle=45.0, kind='dot', phase=(0.0, 0.0), mode='ink', seed=0,
         grain_mm=0.12):
    """Lay screentone over `mask` on the sheet.

    level -- a number, or an array at base resolution (a gradient or a tone map); a tuple (lo, hi, (x0, y0),
             (x1, y1)) makes a linear gradient from lo at the first point to hi at the second.
    kind  -- 'dot', 'line' or 'noise'. For 'line', angle is the direction the lines step across:
             angle=90 gives horizontal lines (still water), angle=0 vertical ones.
    mode  -- 'ink' lays the tone; 'white' scrapes tone and ink away (tone scratching for highlights).
    """
    lvl_arg = level
    if isinstance(level, tuple) and len(level) == 4:
        lvl_arg = {'gradient': [float(level[0]), float(level[1]), list(map(float, level[2])),
                                list(map(float, level[3]))]}
    elif isinstance(level, np.ndarray):
        lvl_arg = {'file': sheet._save_array(level.astype(np.float32), 'level')}
    else:
        lvl_arg = float(level)
    sheet._log({'op': 'tone', 'mask': sheet._arg_mask(mask, 'tonemask'), 'level': lvl_arg, 'lpi': lpi,
                'angle': angle, 'kind': kind, 'phase': list(phase), 'mode': mode, 'seed': seed,
                'grain_mm': grain_mm})
    sheet._n += 1
    sheet.stats['tones'] += 1
    _tone(sheet, mask, level, lpi, angle, kind, phase, mode, seed, grain_mm)


def _level_array(sheet, level):
    if isinstance(level, dict) and 'gradient' in level:
        level = tuple(level['gradient'])
    if isinstance(level, tuple) and len(level) == 4:
        lo, hi, (x0, y0), (x1, y1) = level
        Y, X = np.mgrid[0:sheet.H, 0:sheet.W].astype(np.float32)
        X /= sheet.k
        Y /= sheet.k
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((X - x0) * dx + (Y - y0) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
        return lo + (hi - lo) * t
    if isinstance(level, np.ndarray):
        return sheet.raster(level)
    return float(level)


def _tone(sheet, mask, level, lpi, angle, kind, phase, mode, seed, grain_mm):
    m = sheet.mask(mask)
    ys, xs = np.nonzero(m > 0.001)
    if len(ys) == 0:
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lv = _level_array(sheet, level)
    if isinstance(lv, np.ndarray):
        lv = lv[y0:y1, x0:x1]
    shape = (y1 - y0, x1 - x0)
    if kind == 'noise':
        cov = _noise_region(sheet, lv, grain_mm, seed, y0, y1, x0, x1)
    else:
        cov = screen(lv, shape, sheet.ppm, lpi, angle, kind, phase, origin=(x0 / sheet.k, y0 / sheet.k), k=sheet.k,
                     snap=getattr(sheet, 'snap_screens', False))
    cov = cov * m[y0:y1, x0:x1]
    sheet._put(cov, x0, y0, mode)


def _noise_region(sheet, lv, grain_mm, seed, y0, y1, x0, x1):
    # grains fixed in mm on the sheet, so the same seed gives the same sand at any scale
    gpx = max(grain_mm * sheet.ppm, 1.0)
    gy0, gy1 = int(y0 / gpx), int(np.ceil(y1 / gpx)) + 1
    gx0, gx1 = int(x0 / gpx), int(np.ceil(x1 / gpx)) + 1
    rng = np.random.default_rng(seed)
    full_h, full_w = int(np.ceil(sheet.H / gpx)) + 1, int(np.ceil(sheet.W / gpx)) + 1
    r = rng.random((full_h, full_w)).astype(np.float32)[gy0:gy1, gx0:gx1]
    yy = (np.arange(y0, y1) / gpx).astype(int) - gy0
    xx = (np.arange(x0, x1) / gpx).astype(int) - gx0
    r = r[np.clip(yy, 0, r.shape[0] - 1)][:, np.clip(xx, 0, r.shape[1] - 1)]
    return (r < lv).astype(np.float32)
