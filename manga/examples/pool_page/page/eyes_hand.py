"""The eyes, drawn by hand from points read off the reference (ref px). The face lives or dies by these."""
import numpy as np
from PIL import Image, ImageDraw
from manga import geom, tone


# Read off the reference at 8x (ref px). lid = lower edge of the lash band (where the eye opening starts),
# lash = top edge of the band, wing = the outer lashes sweeping past the corner, crease = the fold above.
EYES = [
    dict(lid=[(367.5, 376), (377.5, 362), (390, 358), (406, 357), (421, 357), (435, 359), (445, 365), (452.5, 372.5)],
         lash=[(366, 372), (376, 351), (390, 347.5), (406, 346.5), (421, 347), (435, 350), (446, 358), (455, 369)],
         wing=[[(376, 351), (367, 355), (359, 361)], [(370, 360), (362, 367), (358, 374)]],
         spikes=[(381, 349.5, -2.5), (391, 347.5, -2.25), (402, 346.5, -2.0)], outer=0,
         crease=[(400, 343), (420, 341.5), (438, 344), (451, 350)],
         lower=[(386, 386), (402, 389.5), (424, 388), (442, 384), (452, 376)],
         lower_lash=[(447, 381, 0.6), (450, 378, 0.35)],
         iris=(424, 373, 20.5, 16.5), pupil=(425.6, 372, 5, 7.2),
         lights=[(406.5, 370, 6.2, 4.4), (422.5, 365.5, 2.4, 2.4)], glint=(433, 380, 1.3)),
    dict(lid=[(538, 387), (550, 379), (566, 375.5), (582.5, 376), (595, 380), (606, 387.5), (615, 395)],
         lash=[(536, 379), (550, 370), (566, 367), (584, 367.5), (600, 371), (615, 381), (624, 392)],
         wing=[[(615, 381), (624, 389), (631, 397)], [(612, 386), (621, 395), (627, 402)]],
         spikes=[(589, 368, -1.1), (600, 371, -0.85), (609, 375, -0.6)], outer=1,
         crease=[(556, 362), (576, 359), (597, 361), (614, 368)],
         lower=[(543, 398), (556, 405), (575, 409.5), (595, 409.5), (607, 404)],
         lower_lash=[(603, 407, 0.7), (607, 404, 0.45)],
         iris=(566, 391, 20.5, 17), pupil=(564.5, 388, 5, 7.2),
         lights=[(548.5, 384, 6.2, 4.4), (565, 381.5, 2.4, 2.4)], glint=(575, 399, 1.3)),
]

CHEEKS = [
    [(380, 430), (386, 447), (394, 466), (403, 486), (415, 505), (402, 492), (408, 481), (397, 476), (402, 467),
     (392, 461), (413, 451), (396, 444), (388, 435)],
    [(603, 448), (596, 468), (586, 486), (572, 503), (556, 516), (545, 521), (551, 509), (566, 500), (568, 492),
     (562, 483), (574, 480), (572, 470), (563, 463), (580, 458), (590, 452)],
]


def _mirror_right_to_left(R, left_iris=(424.0, 374.0)):
    """The right eye reflected across her face's midline. Her head tilts, so the midline is not vertical: it is
    the perpendicular bisector of the line between the two irises, and the reflection turns the eye with it
    (a plain left-right flip left the left eye's corners at the wrong angle)."""
    Rc = np.array(R['iris'][:2], float); Lc = np.array(left_iris, float)
    M = (Rc + Lc) / 2; u = (Rc - Lc) / np.hypot(*(Rc - Lc))
    def P(x, y):
        v = np.array([x, y], float); v = v - 2 * np.dot(v - M, u) * u
        return float(v[0]), float(v[1])
    def A(a):                                            # a direction reflected the same way
        d = np.array([np.cos(a), np.sin(a)]); d = d - 2 * np.dot(d, u) * u
        return float(np.arctan2(d[1], d[0]))
    Ms = lambda pts: [P(x, y) for x, y in pts]          # noqa: E731
    L = dict(R)
    for key in ('lid', 'lash', 'crease', 'lower'):
        L[key] = Ms(R[key])[::-1]
    L['wing'] = [Ms(w) for w in R['wing']]
    L['spikes'] = [P(x, y) + (A(a),) for x, y, a in R['spikes']]
    L['lower_lash'] = [P(x, y) + (ln,) for x, y, ln in R['lower_lash']]
    L['iris'] = P(*R['iris'][:2]) + tuple(R['iris'][2:])
    L['pupil'] = P(*R['pupil'][:2]) + tuple(R['pupil'][2:])
    L['outer'] = 0
    return L


# the left eye was drawn a different shape from the right one; it now mirrors the right eye across her tilted
# face's midline
EYES[0] = _mirror_right_to_left(EYES[1])

# catch-lights placed on the iris itself (fractions of its radii), the same for both eyes: the light comes
# from the upper left, so they are not mirrored
LIGHTS = [(-0.44, -0.18, 0.26, 0.18), (0.18, -0.28, 0.10, 0.10)]
GLINT = (0.38, 0.42, 0.065)


# the scale-shaped diamonds on the swimsuit's chest: left out of the drawing on Michael's request (2026-11-25)
SUIT_PLAIN = [(430, 612), (512, 612), (512, 742), (430, 742)]


def eye_region(W, H, pad=0):
    """Where the hand-drawn eyes go (ref px): traced lines and hair are kept out of it."""
    im = Image.new('L', (W, H), 0)
    dr = ImageDraw.Draw(im)
    dr.polygon(SUIT_PLAIN, fill=1)
    for c in CHEEKS:
        dr.polygon(c, fill=1)
        if pad:
            dr.line(c + [c[0]], fill=1, width=int(2 * pad))
    for e in EYES:
        pts = e['lash'] + e['wing'][0][1:] + e['lower'][::-1]
        dr.polygon(pts, fill=1)
        if pad:
            dr.line(pts + [pts[0]], fill=1, width=int(2 * pad))
    return np.asarray(im, bool)


def _mask(W, H, pts):
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in pts], fill=1)
    return np.asarray(im, bool)


def _ell(cx, cy, rx, ry, n=90, rot=0.0):
    t = np.linspace(0, 2 * np.pi, n)
    c, s_ = np.cos(rot), np.sin(rot)
    x, y = rx * np.cos(t), ry * np.sin(t)
    return np.column_stack([cx + x * c - y * s_, cy + x * s_ + y * c])


IRIS_SCALE = 1.3


def draw_eyes(sh, to_page, s, style='detailed', iris_scale=None):
    """The eyes, layer by layer, as a manga artist builds them:
    sclera (paper) with the lid's cast shadow -> iris: dark upper half, light lower half, radial streaks and a
    reflected crescent -> heavy iris rim -> pupil -> catch-lights -> lash band with gloss specks, lash spikes and
    the outer wing -> crease -> lower lid and a few lower lashes."""
    if style == 'simple':
        import eyes_hand_v1
        return eyes_hand_v1.draw_eyes(sh, to_page, s)
    W, H = int(round(sh.W / sh.k)), int(round(sh.H / sh.k))
    P = lambda pts: np.array([to_page(x, y) for x, y in pts], float)        # noqa: E731
    sm = lambda pts, per=8: geom.catmull(P(pts), per=per)                    # noqa: E731
    rng = np.random.default_rng(31)
    k = IRIS_SCALE if iris_scale is None else iris_scale
    for e in EYES:
        lid, low = sm(e['lid']), sm(e['lower'])
        # a bigger iris needs a rounder eye: the lower lid drops most in the middle
        bow = np.sin(np.linspace(0, np.pi, len(low)))
        low = low + np.column_stack([np.zeros(len(low)), bow * (k - 1) * 22 * s])
        opening = np.vstack([lid, low[::-1]])
        open_m = _mask(W, H, opening)
        # 1. sclera: paper, with the upper lid's shadow lying across it
        sh.fill([tuple(p) for p in opening], mode='white')
        shadow = np.vstack([lid, (lid + [0, 3.5 * s])[::-1]])
        tone(sh, _mask(W, H, shadow) & open_m, level=0.28, lpi=70, angle=45)
        # 2. iris
        k = IRIS_SCALE if iris_scale is None else iris_scale
        cx, cy = to_page(*e['iris'][:2]); rx, ry = e['iris'][2] * s * k, e['iris'][3] * s * k * 1.08
        cy += (k - 1) * 4 * s                      # grow downward too: the upper lid hides the top
        iris = _mask(W, H, _ell(cx, cy, rx, ry)) & open_m
        Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
        v = (Y - (cy - ry)) / (2 * ry)                                     # 0 top .. 1 bottom of the iris
        band = _mask(W, H, np.vstack([lid - [0, 3 * s], (lid + [0, 0.85 * ry])[::-1]]))
        lv = np.where(band, 0.88, np.clip(0.62 - 0.55 * (v - 0.42) / 0.58, 0.22, 0.62)).astype(np.float32)
        tone(sh, iris, level=lv, lpi=75, angle=45)
        if style == 'detailed':
            # radial streaks in the lower half, from the pupil outward (the iris's fibres)
            px, py = to_page(*e['pupil'][:2])
            with sh.clip(iris):
                for a in np.linspace(0.15 * np.pi, 0.85 * np.pi, 9) + rng.normal(0, 0.05, 9):
                    r0, r1 = 0.45 * min(rx, ry) * 1.2, 0.92 * rx
                    sh.stroke([(px + r0 * np.cos(a), py + r0 * np.sin(a) * ry / rx),
                               (px + r1 * np.cos(a), py + r1 * np.sin(a) * ry / rx)],
                              pen='maru', pressure=0.28, taper=(0.3, 0.6), smooth=False)
                # reflected light: a soft crescent scraped back near the bottom of the iris
                cres = _mask(W, H, _ell(cx + 0.1 * rx, cy + 0.55 * ry, 0.62 * rx, 0.30 * ry)) & \
                    ~_mask(W, H, _ell(cx + 0.1 * rx, cy + 0.35 * ry, 0.70 * rx, 0.30 * ry))
                tone(sh, cres & iris, level=0.15, lpi=75, mode='white')
            # 3. rim: a heavy line round the iris, heavier at the top, cut by the lids
            with sh.clip(open_m):
                ring = _ell(cx, cy, rx, ry, 120)
                top = ring[:, 1] < cy
                pr = np.where(top, 1.0, 0.7)
                sh.stroke(np.column_stack([ring, pr]), pen='g_pen', taper=(0, 0))
        # 4. pupil
        px, py = to_page(*e['pupil'][:2])
        with sh.clip(open_m):
            sh.fill([tuple(p) for p in _ell(px, py, e['pupil'][2] * s * k, e['pupil'][3] * s * k)])
            # 5. catch-lights: a big one over the iris edge, a small one by the pupil, a glint below
            for fx, fy, frx, fry in LIGHTS:
                qx, qy = cx + fx * rx, cy + fy * ry
                sh.fill([tuple(p) for p in _ell(qx, qy, frx * rx, fry * rx, rot=-0.35)], mode='white')
                sh.stroke(_ell(qx, qy, frx * rx, fry * rx, 60, rot=-0.35), pen='maru', pressure=0.15,
                          taper=(0, 0))
            if style == 'detailed':
                gx, gy = cx + GLINT[0] * rx, cy + GLINT[1] * ry
                sh.fill([tuple(p) for p in _ell(gx, gy, GLINT[2] * rx, GLINT[2] * rx)], mode='white')
        # 6. lash band: solid black between lid and lash line, the wing sweeping past the corner
        lash = sm(e['lash'])
        n = min(len(lid), len(lash)); lid_, lash_ = lid[:n], lash[:n]
        u = np.linspace(0, 1, n) if e['outer'] == 1 else np.linspace(1, 0, n)    # 0 inner corner .. 1 outer
        prof = np.clip(0.30 + 1.4 * u, 0, 1) * (1 - 0.15 * np.clip((u - 0.85) / 0.15, 0, 1))
        lash = lid_ + (lash_ - lid_) * prof[:, None]
        sh.fill([tuple(p) for p in np.vstack([lid_, lash[::-1]])])
        sh.stroke(np.column_stack([lid, np.full(len(lid), 0.7)]), pen='g_pen', taper=(0.3, 0.6))
        for k, w in enumerate(e['wing']):
            sh.stroke(np.column_stack([P(w), [0.9, 0.7, 0.3]]), pen='g_pen', taper=(0.0, 1.6 - 0.4 * k))
        # lash spikes off the top edge, leaning toward the outer corner
        for x, y, lean in e['spikes']:
            bx, by = to_page(x, y)
            L = (7.0 + rng.uniform(0, 2.5)) * s
            dx, dy = np.cos(lean), np.sin(lean)
            out = 1 if e['outer'] == 1 else -1                     # bend toward the outer corner
            sh.stroke([(bx, by), (bx + dx * L * 0.5, by + dy * L * 0.5),
                       (bx + dx * L + out * 1.2 * s, by + dy * L + 1.0 * s)],
                      pen='g_pen', pressure=0.32, taper=(0.0, 1.0), seed=int(rng.integers(1 << 30)))
        if style == 'detailed':
            # gloss specks along the middle of the lash band (the shine on the lashes)
            mid = lid_ + (lash - lid_) * 0.55
            for i in sorted(rng.choice(np.arange(int(0.3 * n), int(0.8 * n)), 5, replace=False)):
                qx, qy = mid[i] + rng.normal(0, 0.3 * s, 2)
                sh.fill([tuple(p) for p in _ell(qx, qy, 0.45 * s, 0.32 * s, 10)], mode='white')
            # 7. crease above the lid, thin and broken
            sh.stroke(np.column_stack([sm(e['crease']), np.full(len(sm(e['crease'])), 0.35)]), pen='maru',
                      taper=(1.0, 1.4))
        # 8. lower lid: light, darker toward the outer corner; a couple of short lower lashes there
        uo = np.linspace(0, 1, len(low)) if e['outer'] == 1 else np.linspace(1, 0, len(low))
        sh.stroke(np.column_stack([low, 0.15 + 0.75 * uo ** 1.5]), pen='g_pen',
                  taper=(1.2, 0.2) if e['outer'] == 1 else (0.2, 1.2))
        # close the outer corner: the lid band and the lower line meet in a small black wedge
        oc = lid_[-1] if e['outer'] == 1 else lid_[0]
        ol = low[-1] if e['outer'] == 1 else low[0]
        mid_c = 0.5 * (oc + ol) + np.array([(1.5 if e['outer'] == 1 else -1.5) * s, 0])
        sh.fill([tuple(oc), tuple(mid_c), tuple(ol), tuple(0.5 * (oc + ol))])
        if style == 'detailed':
            for x, y, ln in e['lower_lash']:
                bx, by = to_page(x, y)
                sh.stroke([(bx, by), (bx + 1.8 * s * ln * 2, by + 2.6 * s * ln * 2)], pen='maru', pressure=0.5,
                          taper=(0.0, 0.8))


def draw_cheek_scales(sh, to_page, s, face=None):
    """The scale patches on her cheeks: pale, clean-edged, with two or three inner scale lines.

    face -- page-size bool mask of her skin: the patch's outer edge is pushed out past the cheek and cut to it,
    so the scales run right to the edge of the face (where the face contour line already is)."""
    W, H = int(round(sh.W / sh.k)), int(round(sh.H / sh.k))
    for ci, c in enumerate(CHEEKS):
        n_out = 5 if len(c) == 13 else 6
        sgn = -1 if ci == 0 else 1                                  # outward = away from the face's middle
        c_ext = [(x + sgn * 14, y) for x, y in c[:n_out]] + list(c[n_out:])
        P = [to_page(x, y) for x, y in c_ext]
        im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(P, fill=1)
        patch = np.asarray(im, bool)
        if face is not None:
            patch &= face
        sh.fill(patch, mode='white')
        tone(sh, patch, level=0.12, lpi=60)
        # outline: the jagged inner edge only; the outer edge is the face contour, inked separately
        Pin = [to_page(x, y) for x, y in [c[n_out - 1]] + list(c[n_out:]) + [c[0]]]
        sh.stroke(np.array(Pin), pen='maru', pressure=0.75, taper=(0, 0), smooth=False)
        # inner scale lines: from each inner notch a short arc toward the outer edge
        n_out = 5 if len(c) == 13 else 6
        outer, inner = c[:n_out], c[n_out:]
        for k, (nx, ny) in enumerate(inner[1::2]):
            j = min(len(outer) - 1, int(round((1 - (k + 1) / (len(inner) / 2 + 1)) * (len(outer) - 1))))
            ox, oy = outer[j]
            mx, my = (nx + ox) / 2 + (oy - ny) * 0.12, (ny + oy) / 2 - (ox - nx) * 0.12
            sh.stroke(np.array([to_page(nx, ny), to_page(mx, my), to_page(ox * 0.7 + nx * 0.3, oy * 0.7 + ny * 0.3)]),
                      pen='maru', pressure=0.4, taper=(0, 0.6))
