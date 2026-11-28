"""The manga sheet: a single ink-coverage buffer on paper, a journal of every call, and replay at any scale.

Coordinates passed to a Sheet are always in *base* pixels (the size the sheet was created with). A sheet created
with scale=3 is the same drawing three times larger: every coordinate is multiplied, pen widths stay in mm and
px_per_mm grows. That is how `python -m manga.replay page.jsonl --scale 3` re-renders a page for print.
"""
import json
import os
from contextlib import contextmanager

import numpy as np
from PIL import Image, ImageDraw

from . import geom
from .pens import Pen, get as get_pen
from .settings import configurable

PAPER = (247, 245, 239)
INK = (18, 17, 20)


def _poly_mask(poly, W, H, k=1.0):
    m = Image.new('L', (W, H), 0)
    ImageDraw.Draw(m).polygon([(float(x) * k, float(y) * k) for x, y in poly], fill=255)
    return np.asarray(m, np.float32) / 255.0


class Sheet:
    @configurable('sheet')
    def __init__(self, width, height, px_per_mm=8.0, paper=PAPER, ink=INK, journal=None, seed=0, scale=1.0):
        self.W0, self.H0 = int(width), int(height)
        self.k = float(scale)
        self.W, self.H = int(round(width * scale)), int(round(height * scale))
        self.ppm = float(px_per_mm) * self.k
        self.ppm0 = float(px_per_mm)
        self.paper = tuple(paper)
        self.ink_rgb = tuple(ink)
        self.seed = int(seed)
        self.ink = np.zeros((self.H, self.W), np.float32)
        self._clips = []
        self._n = 0
        self._journal = None
        self._jdir = None
        self._replaying = False
        self.stats = {'strokes': 0, 'tones': 0, 'fills': 0}
        if journal:
            self._journal = open(journal, 'w', encoding='utf-8')
            self._jdir = journal + '.d'
            os.makedirs(self._jdir, exist_ok=True)
            self._log({'op': 'sheet', 'width': self.W0, 'height': self.H0, 'px_per_mm': self.ppm0,
                       'paper': list(self.paper), 'ink': list(self.ink_rgb), 'seed': self.seed})

    # ------------------------------------------------------------------ journal
    def _log(self, d):
        if self._journal and not self._replaying:
            self._journal.write(json.dumps(d, ensure_ascii=False, separators=(',', ':')) + '\n')

    def _save_array(self, a, kind):
        """Store a raster argument next to the journal; returns its file name (or None when not journaling)."""
        if not self._journal or self._replaying:
            return None
        a = np.asarray(a)
        name = f'{self._n:06d}_{kind}.png'
        if a.dtype == bool:
            Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(self._jdir, name))
        else:
            Image.fromarray(np.clip(np.asarray(a, np.float32) * 65535 + 0.5, 0, 65535).astype(np.uint16)).save(
                os.path.join(self._jdir, name))
        return name

    def note(self, text, **kw):
        self._log({'op': 'note', 'text': text, **kw})

    def close(self):
        if self._journal:
            self._journal.close()
            self._journal = None

    # ------------------------------------------------------------------ rasters at sheet resolution
    def raster(self, a, smooth=True):
        """Bring a base-resolution array (mask or level, H0 x W0) to sheet resolution as float32."""
        a = np.asarray(a)
        if a.dtype == bool:
            a = a.astype(np.float32)
        a = a.astype(np.float32)
        if a.shape == (self.H, self.W):
            return a
        im = Image.fromarray(a, mode='F').resize((self.W, self.H), Image.BILINEAR if smooth else Image.NEAREST)
        return np.asarray(im, np.float32)

    def mask(self, m):
        """None -> everything; list of points -> polygon; array -> raster (base or sheet resolution)."""
        if m is None:
            return np.ones((self.H, self.W), np.float32)
        if isinstance(m, (list, tuple)):
            return _poly_mask(m, self.W, self.H, self.k)
        r = self.raster(m)
        if np.asarray(m).dtype == bool:
            r = np.clip((r - 0.5) * 2.0 + 0.5, 0, 1)   # keep cut edges crisp after resizing
        return r

    def _arg_mask(self, m, kind='mask'):
        """Journal form of a mask argument."""
        if m is None:
            return None
        if isinstance(m, (list, tuple)):
            return {'poly': [[float(x), float(y)] for x, y in m]}
        return {'file': self._save_array(m, kind)}

    # ------------------------------------------------------------------ clipping (panels)
    @contextmanager
    def clip(self, m):
        """Everything drawn inside the block is cut to this mask (a panel)."""
        self._log({'op': 'clip', 'mask': self._arg_mask(m, 'clip')})
        self._n += 1
        self._clips.append(self.mask(m))
        try:
            yield self
        finally:
            self._clips.pop()
            self._log({'op': 'unclip'})

    def _clip_region(self, x0, y0, x1, y1):
        if not self._clips:
            return None
        c = self._clips[-1][y0:y1, x0:x1]
        for o in self._clips[:-1]:
            c = np.minimum(c, o[y0:y1, x0:x1])
        return c

    # ------------------------------------------------------------------ compositing
    def _put(self, cov, x0, y0, mode):
        H, W = cov.shape
        x1, y1 = x0 + W, y0 + H
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, self.W), min(y1, self.H)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        c = cov[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        cl = self._clip_region(cx0, cy0, cx1, cy1)
        if cl is not None:
            c = c * cl
        dst = self.ink[cy0:cy1, cx0:cx1]
        if mode == 'ink':
            dst += c * (1.0 - dst)
        elif mode == 'white':
            dst *= (1.0 - c)
        else:
            raise ValueError(f"mode must be 'ink' or 'white', not {mode!r}")

    # ------------------------------------------------------------------ strokes
    def stroke(self, points, pen='g_pen', pressure=0.7, seed=None, mode='ink', taper=None, smooth=True):
        """One pen stroke through control points.

        points   -- [(x, y), ...] or [(x, y, pressure), ...] in base px. Two points make a straight line.
        pressure -- used when points carry none: the body pressure (the pen's taper still opens and closes it).
        taper    -- (in_mm, out_mm) to override the pen's entry/exit taper; (0, 0) for a blunt line.
        mode     -- 'ink' or 'white' (white-out over ink and tone).
        """
        pen = get_pen(pen)
        P = np.asarray(points, float)
        if P.ndim != 2 or len(P) < 1:
            raise ValueError('points must be a list of (x, y) or (x, y, pressure)')
        if seed is None:
            seed = (self.seed * 1000003 + self._n * 7919) & 0x7fffffff
        rec = {'op': 'stroke', 'points': np.round(P, 2).tolist(), 'pen': pen.to_dict(), 'pressure': pressure,
               'seed': int(seed), 'mode': mode, 'taper': list(taper) if taper else None, 'smooth': smooth}
        self._log(rec)
        self._n += 1
        self.stats['strokes'] += 1
        self._stroke(P, pen, pressure, int(seed), mode, taper, smooth)

    def _stroke(self, P, pen, pressure, seed, mode, taper, smooth):
        rng = np.random.default_rng(seed)
        k, ppm = self.k, self.ppm
        if P.shape[1] < 3:
            P = np.hstack([P[:, :2], np.full((len(P), 1), float(pressure))])
        P = P[:, :3].copy()
        P[:, :2] *= k
        if len(P) == 1:                                  # a dot
            P = np.vstack([P, P + [0.01, 0, 0]])
        if smooth and len(P) > 2:
            P = geom.catmull(P, per=10)
        r_max = 0.5 * pen.max_mm * ppm
        step = float(np.clip(0.5 * max(r_max, 1.0), 0.75, 3.0))
        Q, s = geom.resample(P, step)
        if len(Q) < 2:
            Q = np.vstack([Q, Q + [0.01, 0, 0]])
            s = np.array([0.0, 0.01])
        L = s[-1]
        # entry/exit taper on pressure
        tin, tout = (taper if taper is not None else (pen.taper_in_mm, pen.taper_out_mm))
        tin_px, tout_px = tin * ppm, tout * ppm
        if tin_px + tout_px > L and (tin_px + tout_px) > 0:
            f = L / (tin_px + tout_px)
            tin_px, tout_px = tin_px * f, tout_px * f
        env = np.ones_like(s)
        if tin_px > 0:
            env = np.minimum(env, np.clip(s / tin_px, 0, 1) ** 0.8)
        if tout_px > 0:
            env = np.minimum(env, np.clip((L - s) / tout_px, 0, 1) ** 0.8)
        p = np.clip(Q[:, 2] * env, 0, 1)
        w_mm = pen.width_mm(p)
        if pen.rough > 0:
            w_mm = w_mm * (1 + pen.rough * geom.smooth_noise(s, rng, 1.5 * ppm, 3))
        r = np.maximum(0.5 * w_mm * ppm, 0.0)
        xy = Q[:, :2]
        if pen.wobble_mm > 0 and len(xy) > 2:
            xy = xy + geom.normals(xy) * (pen.wobble_mm * ppm * geom.smooth_noise(s, rng, 12 * ppm, 2))[:, None]
        need_su = pen.dry > 0
        out = _raster_capsules(xy, r, s, need_su)
        if out is None:
            return
        cov, x0, y0, S, U = out
        if need_su:
            # dry brush: hairs run out of ink toward the end of a long stroke
            start = (1.0 - pen.dry) * max(L, 1.0)
            dry = np.clip((S - start) / max(L - start, 1.0), 0, 1)
            nh = 9 + int(rng.integers(0, 6))
            ph = rng.uniform(0, 100)
            hair = 0.5 + 0.5 * np.sin(U * nh * np.pi + ph + 1.5 * np.sin(S / (2.5 * ppm) + U * 2.3 + ph))
            hair = hair * 0.75 + 0.25 * rng.random(hair.shape)
            cov = cov * np.clip((hair + 0.2 - dry * 1.25) * 5, 0, 1)
        if pen.ink < 1:
            cov = cov * pen.ink
        self._put(cov, x0, y0, mode)

    # ------------------------------------------------------------------ flat fills (beta) and tone
    def fill(self, mask, mode='ink', amount=1.0):
        """Solid black (beta) or solid white over a mask."""
        self._log({'op': 'fill', 'mask': self._arg_mask(mask, 'fill'), 'mode': mode, 'amount': amount})
        self._n += 1
        self.stats['fills'] += 1
        self._put(self.mask(mask) * float(amount), 0, 0, mode)

    def apply(self, cov, mode='ink', kind='layer'):
        """Composite a precomputed coverage array (base or sheet resolution). Journals the array itself."""
        self._log({'op': 'apply', 'cov': {'file': self._save_array(np.asarray(cov, np.float32), kind)}, 'mode': mode})
        self._n += 1
        self._put(self.raster(cov), 0, 0, mode)

    # ------------------------------------------------------------------ output
    def image(self, bw=False):
        a = self.ink
        if bw:
            a = (a > 0.5).astype(np.float32)
        p = np.array(self.paper, np.float32)
        q = np.array(self.ink_rgb, np.float32)
        rgb = p * (1 - a[..., None]) + q * a[..., None]
        return Image.fromarray(np.clip(rgb + 0.5, 0, 255).astype(np.uint8))

    def png(self, path, bw=False, crop=None, dpi=None):
        im = self.image(bw)
        if crop:
            im = im.crop(tuple(int(round(v * self.k)) for v in crop))
        if bw:
            im = im.convert('L').point(lambda v: 255 if v > 127 else 0).convert('1')
        dpi = dpi or self.ppm * 25.4
        im.save(path, dpi=(dpi, dpi))
        return path


def _raster_capsules(xy, r, s, need_su=False):
    """Union of round-capped segments with linearly varying radius. Returns (cov, x0, y0, S, U) in a local box."""
    pad = float(r.max()) + 2.0
    x0 = int(np.floor(xy[:, 0].min() - pad))
    y0 = int(np.floor(xy[:, 1].min() - pad))
    x1 = int(np.ceil(xy[:, 0].max() + pad)) + 1
    y1 = int(np.ceil(xy[:, 1].max() + pad)) + 1
    if r.max() <= 0.02:
        return None
    H, W = y1 - y0, x1 - x0
    cov = np.zeros((H, W), np.float32)
    S = np.zeros((H, W), np.float32) if need_su else None
    U = np.zeros((H, W), np.float32) if need_su else None
    n = len(xy)
    for i in range(n - 1):
        a, b = xy[i], xy[i + 1]
        ra, rb = r[i], r[i + 1]
        rr_max = max(ra, rb)
        if rr_max <= 0.02:
            continue
        m = rr_max + 1.5
        bx0 = int(np.floor(min(a[0], b[0]) - m)) - x0
        by0 = int(np.floor(min(a[1], b[1]) - m)) - y0
        bx1 = int(np.ceil(max(a[0], b[0]) + m)) - x0 + 1
        by1 = int(np.ceil(max(a[1], b[1]) + m)) - y0 + 1
        bx0, by0 = max(bx0, 0), max(by0, 0)
        bx1, by1 = min(bx1, W), min(by1, H)
        X = np.arange(bx0, bx1, dtype=np.float32)[None, :] + x0 + 0.5
        Y = np.arange(by0, by1, dtype=np.float32)[:, None] + y0 + 0.5
        abx, aby = b[0] - a[0], b[1] - a[1]
        L2 = abx * abx + aby * aby
        px, py = X - a[0], Y - a[1]
        if L2 < 1e-12:
            t = np.zeros(np.broadcast(px, py).shape, np.float32)
        else:
            t = np.clip((px * abx + py * aby) / L2, 0, 1)
        dx, dy = px - t * abx, py - t * aby
        d = np.sqrt(dx * dx + dy * dy)
        rr = ra + (rb - ra) * t
        c = np.clip(np.maximum(rr, 0.5) + 0.5 - d, 0, 1) * np.minimum(1.0, 2.0 * rr)
        sl = cov[by0:by1, bx0:bx1]
        if need_su:
            upd = c > sl
            sl[upd] = c[upd]
            L = np.sqrt(L2) if L2 > 0 else 1.0
            cross = (abx * py - aby * px) / L
            S[by0:by1, bx0:bx1][upd] = (s[i] + (s[i + 1] - s[i]) * t)[upd]
            U[by0:by1, bx0:bx1][upd] = (cross / np.maximum(rr, 0.5))[upd]
        else:
            np.maximum(sl, c, out=sl)
    return cov, x0, y0, S, U
