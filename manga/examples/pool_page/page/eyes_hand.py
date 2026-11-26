"""The eyes, drawn by hand from points read off the reference (ref px). The face lives or dies by these."""
import numpy as np
from PIL import Image, ImageDraw
from manga import geom, tone

EYES = [
    dict(upper=[(372, 364), (383, 355), (400, 350), (420, 349), (437, 353), (450, 362), (454, 371)],
         lower=[(386, 388), (403, 391), (427, 389), (448, 383)],
         iris=(423, 372, 20, 22), pupil=(425, 372, 5.5, 8.5), lights=[(408, 370, 4.2), (421, 365, 2.2)],
         flick=[[(450, 362), (458, 357), (463, 356)], [(453, 368), (461, 368)]]),
    dict(upper=[(533, 384), (545, 375), (560, 370), (583, 370), (600, 377), (611, 388), (615, 397)],
         lower=[(548, 404), (565, 408), (585, 410), (604, 409)],
         iris=(564, 389, 20, 22), pupil=(566, 389, 5.5, 8.5), lights=[(551, 384, 4.2), (562, 380, 2.2)],
         flick=[[(611, 388), (619, 384), (625, 384)], [(614, 395), (622, 397)]]),
]


CHEEKS = [
    [(380, 430), (386, 447), (394, 466), (403, 486), (415, 505), (402, 492), (408, 481), (397, 476), (402, 467),
     (392, 461), (413, 451), (396, 444), (388, 435)],
    [(603, 448), (596, 468), (586, 486), (572, 503), (556, 516), (545, 521), (551, 509), (566, 500), (568, 492),
     (562, 483), (574, 480), (572, 470), (563, 463), (580, 458), (590, 452)],
]


def eye_region(W, H, pad=0):
    im = Image.new('L', (W, H), 0)
    dr = ImageDraw.Draw(im)
    for c in CHEEKS:
        dr.polygon(c, fill=1)
        if pad:
            dr.line(c + [c[0]], fill=1, width=int(2 * pad))
    for e in EYES:
        pts = e['upper'] + e['lower'][::-1]
        dr.polygon(pts, fill=1)
        if pad:
            dr.line(pts + [pts[0]], fill=1, width=int(2 * pad))
    return np.asarray(im, bool)


def _poly_mask(sh, pts):
    return [tuple(p) for p in pts]


def draw_eyes(sh, to_page, s):
    P = lambda pts: [to_page(x, y) for x, y in pts]                      # noqa: E731
    for e in EYES:
        up = geom.catmull(np.array(P(e['upper']), float), per=8)
        lo = geom.catmull(np.array(P(e['lower']), float), per=8)
        opening = [tuple(p) for p in np.vstack([up, lo[::-1]])]
        sh.fill(opening, mode='white')                                  # sclera is paper
        # iris: an ellipse cut by the lids, darker toward the top (the lid's shadow), outlined
        cx, cy = to_page(*e['iris'][:2]); rx, ry = e['iris'][2] * s, e['iris'][3] * s
        t = np.linspace(0, 2 * np.pi, 90)
        ell = np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])
        W, H = int(round(sh.W / sh.k)), int(round(sh.H / sh.k))
        im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon([tuple(p) for p in ell], fill=1)
        im2 = Image.new('L', (W, H), 0); ImageDraw.Draw(im2).polygon(opening, fill=1)
        iris = np.asarray(im, bool) & np.asarray(im2, bool)
        Y = np.arange(H, dtype=np.float32)[:, None]
        lv = np.clip(0.85 - 0.5 * (Y - (cy - ry)) / (2 * ry), 0.3, 0.85) * np.ones((1, W), np.float32)
        tone(sh, iris, level=lv, lpi=70, angle=45)
        # the upper lid's shadow across the white of the eye
        lid = Image.new('L', (W, H), 0)
        ImageDraw.Draw(lid).line([tuple(p) for p in up], fill=1, width=int(7 * s))
        tone(sh, np.asarray(lid, bool) & np.asarray(im2, bool) & ~iris, level=0.22, lpi=70, angle=45)
        with sh.clip(np.asarray(im2, bool)):
            sh.stroke(np.vstack([ell, ell[:1]]), pen='maru', pressure=0.7, taper=(0, 0))
            px, py = to_page(*e['pupil'][:2]); prx, pry = e['pupil'][2] * s, e['pupil'][3] * s
            sh.fill([(px + prx * np.cos(a), py + pry * np.sin(a)) for a in t])
            for lx, ly, r in e['lights']:
                qx, qy = to_page(lx, ly)
                sh.fill([(qx + r * s * np.cos(a), qy + r * s * 1.1 * np.sin(a)) for a in t], mode='white')
        # upper lid: a heavy black lash band, thick in the middle, tapering to both corners
        n = len(up)
        w = (np.sin(np.linspace(0, np.pi, n)) ** 0.6) * 4.5 * s + 0.6 * s
        nrm = geom.normals(up)
        band = np.vstack([up - nrm * 0.3 * w[:, None], (up + nrm * w[:, None])[::-1]])
        top = np.vstack([up + nrm * w[:, None]])
        # normals may point either way: put the band above the lid line (smaller y)
        if np.mean(top[:, 1]) > np.mean(up[:, 1]):
            band = np.vstack([up + nrm * 0.3 * w[:, None], (up - nrm * w[:, None])[::-1]])
        sh.fill([tuple(p) for p in band])
        sh.stroke(np.array(P(e['upper'])), pen='g_pen', pressure=0.55, taper=(0.4, 0.8))
        for f in e['flick']:
            sh.stroke(np.array(P(f)), pen='g_pen', pressure=0.5, taper=(0.0, 1.0))
        # lower lid: a light maru line, open at the inner corner
        sh.stroke(np.array(P(e['lower'])), pen='maru', pressure=0.45, taper=(0.6, 0.6))


def draw_cheek_scales(sh, to_page, s):
    """The scale patches on her cheeks: pale, clean-edged, with two or three inner scale lines."""
    W, H = int(round(sh.W / sh.k)), int(round(sh.H / sh.k))
    for c in CHEEKS:
        P = [to_page(x, y) for x, y in c]
        sh.fill(P, mode='white')
        im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(P, fill=1)
        tone(sh, np.asarray(im, bool), level=0.12, lpi=60)
        sh.stroke(np.array(P + [P[0]]), pen='maru', pressure=0.75, taper=(0, 0), smooth=False)
        # inner scale lines: from each inner notch a short arc toward the outer edge
        n_out = 5 if len(c) == 13 else 6
        outer, inner = c[:n_out], c[n_out:]
        for k, (nx, ny) in enumerate(inner[1::2]):
            j = min(len(outer) - 1, int(round((1 - (k + 1) / (len(inner) / 2 + 1)) * (len(outer) - 1))))
            ox, oy = outer[j]
            mx, my = (nx + ox) / 2 + (oy - ny) * 0.12, (ny + oy) / 2 - (ox - nx) * 0.12
            sh.stroke(np.array([to_page(nx, ny), to_page(mx, my), to_page(ox * 0.7 + nx * 0.3, oy * 0.7 + ny * 0.3)]),
                      pen='maru', pressure=0.4, taper=(0, 0.6))
