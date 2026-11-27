"""The parts an expression is made of, each drawn the way an inker draws it.

A face reads through a small vocabulary: the brows (their slant, weight and the knit between them), the eye (how
far the lids open, how much white shows round the iris, where the pupil sits, the catch-lights), the mouth (a
line, an open shape with teeth and tongue), and the marks laid over them (a blush, a sweat drop, the gloom band
of fright). Each has a function here. Shapes come as page-px masks or points, read off a reference or placed by
hand, so an expression is data that can be changed.

    face.brow(sh, pts, widths)                          # a tapered brow through points
    pts, w, line = face.brow_stroke(wedge, cand)        # a reference brow as one thick-to-thin stroke
    face.brow_ring(sh, pts, w)                          # its paper ring, laid before any brow is inked
    face.iris_to_rim(iris_mask, dark)                   # an iris filled out to the reference's own rim
    face.iris(sh, iris_mask, levels, pupil=m, lights=m) # cel-shaded iris, rim, streaks, pupil, catch-lights
    face.eye_white(sh, white_mask, lash_mask)           # the lid's cast shadow on the white
    face.manga_iris(sh, iris_mask, pupil=m, lights=m)   # mostly paper: black top, fading tone, small pupil
    face.open_mouth(sh, inside, tongue=m)               # black inside, toned tongue, clean teeth
    face.blush(sh, (cx, cy), rx, ry, n=7)               # slanted hatch on a cheek
    face.gloom(sh, mask)                                # vertical line tone and falling lines: dread
    face.sweat(sh, (x, y), size)                        # a drop with its highlight
    face.closed_eye(sh, inner, outer, to_page=fr.P)     # a content closed eye (ref px + to_page)
    face.closed_smile(sh, pts, to_page=fr.P)
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd

from . import geom
from .tone import tone


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


# ---------------------------------------------------------------------------------------------------------------
# parts drawn from page-px shapes


def _smooth(mask, sigma):
    return nd.gaussian_filter(np.asarray(mask, np.float32), sigma) > 0.5 if sigma else np.asarray(mask, bool)


def ink_shape(sh, mask, sigma=1.0):
    """A solid black shape (a lash band, a pupil, a thick brow) with its edge rounded at the page's scale."""
    m = _smooth(mask, sigma)
    if m.any():
        sh.fill(m)
    return m


def ribbon_of(mask, step=4.0):
    """Measure an elongated blob (a brow, a lid band) as a centre line with a width at each point.
    Returns (pts (N, 2) page px, widths (N,) px), running along the blob's long axis."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return np.zeros((0, 2)), np.zeros(0)
    c = np.array([xs.mean(), ys.mean()])
    d = np.column_stack([xs, ys]) - c
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    ax = vt[0] if vt[0][0] >= 0 else -vt[0]
    nrm = np.array([-ax[1], ax[0]])
    u, v = d @ ax, d @ nrm
    edges = np.arange(u.min(), u.max() + step, step)
    pts, w = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (u >= a) & (u < b)
        if sel.sum() < 2:
            continue
        lo, hi = v[sel].min(), v[sel].max()
        pts.append(c + ax * (a + b) / 2 + nrm * (lo + hi) / 2)
        w.append(hi - lo + 1)
    return np.array(pts), np.array(w, float)


def brow(sh, pts, widths, smooth=1.5):
    """A brow as an inker lays it: one filled, tapered shape through `pts` (page px) with `widths` (px) along
    it, pointed at both ends. Change the points to change the expression: inner ends down for anger, up for
    worry, the whole arc high for surprise."""
    P = np.asarray(pts, float)
    if len(P) < 2:
        return
    w = np.asarray(widths, float) * np.ones(len(P))
    if len(P) > 8 and smooth:                       # dense measured points carry pixel noise; a few hand-placed ones don't
        P = nd.gaussian_filter1d(P, smooth, axis=0, mode='nearest')
        w = nd.gaussian_filter1d(w, smooth, mode='nearest')
    Q = geom.catmull(P, per=6) if len(P) > 2 else P
    wq = np.interp(np.linspace(0, 1, len(Q)), np.linspace(0, 1, len(w)), w)
    u = np.linspace(0, 1, len(Q))
    wq = wq * np.clip(np.minimum(u, 1 - u) / 0.08, 0.15, 1)               # pointed ends
    n = geom.normals(Q)
    poly = np.vstack([Q + n * wq[:, None] / 2, (Q - n * wq[:, None] / 2)[::-1]])
    sh.fill([tuple(p) for p in poly])


def _follow(x0, y0, step, cand, reach, slope, win=3, max_gap=5):
    """Walk column by column from (x0, y0) in direction `step`, taking the candidate pixels nearest the predicted
    y each time: [(x, y, run_length)]. Stops after `max_gap` empty columns or `reach` px."""
    out, x, y, k, gap = [], x0, y0, slope, 0
    while gap <= max_gap and abs(x - x0) < reach:
        x += step
        if x < 0 or x >= cand.shape[1]:
            break
        yp = y + k * step
        lo, hi = max(int(yp - win), 0), int(yp + win + 1)
        hit = np.nonzero(cand[lo:hi, x])[0]
        if len(hit) == 0:
            gap += 1; y = yp; continue
        gap = 0
        yc = hit.mean() + lo
        k = 0.7 * k + 0.3 * (yc - y) / step
        y = yc
        out.append((x, yc, float(len(hit))))
    return out


def brow_stroke(wedge, cand=None, reach=200, every=3, tail_w=4.0, min_w=1.5, tip_len=9):
    """A reference brow as ONE stroke from thick to thin, never a thick piece plus a separate thin line.

    `wedge` is the brow's thick part (bool mask, any px space). `cand` (optional, same size) holds the pixels
    that may continue it (the reference's thin drawn tail: line colours, not the hair's mass); the stroke follows
    it column by column from the wedge's thin end, up to `reach` px. Widths are measured across the stroke (not
    down a column, which overstates a sloping brow), smoothed, and from the widest point on they only narrow.
    A steep, narrow fringe tip lying on the wedge's end is dropped first by a level opening `tip_len` px wide.
    Returns (pts, widths, line) with pts every `every` columns, or None when the wedge is too small; `line` is
    the mask the stroke covers, to take out of the traced passes (one owner per pixel)."""
    wm = np.asarray(wedge, bool)
    lv = nd.binary_opening(wm, structure=np.ones((1, tip_len), bool))
    if lv.sum() > 0.6 * wm.sum():
        lab, n = nd.label(lv)
        if n:
            sz = nd.sum(lv, lab, np.arange(1, n + 1))
            wm = np.isin(lab, 1 + np.nonzero(sz > 0.3 * lv.sum())[0])
    if not wm.any():
        return None
    cols = np.nonzero(wm.any(0))[0]
    ys = np.array([np.nonzero(wm[:, x])[0].mean() for x in cols])
    ws = wm[:, cols].sum(0).astype(float)
    if len(cols) < 6:
        return None
    q = max(len(cols) // 4, 2)
    to_left = ws[:q].mean() < ws[-q:].mean()       # walk towards the thin end...
    if cand is not None:                            # ...unless only the other end has a tail to follow
        c = np.asarray(cand, bool) & ~nd.binary_dilation(wm, iterations=1)
        y0, y1 = int(ys.min()) - 6, int(ys.max()) + 7
        near_l = c[max(y0, 0):y1, max(cols[0] - 10, 0):cols[0]].sum()
        near_r = c[max(y0, 0):y1, cols[-1] + 1:cols[-1] + 11].sum()
        if (near_l > 4) != (near_r > 4):
            to_left = near_l > 4
    if to_left:
        cols, ys, ws = cols[::-1], ys[::-1], ws[::-1]
    step = 1 if cols[-1] > cols[0] else -1
    tail = []
    if cand is not None:
        slope = np.polyfit(cols[-2 * q:], ys[-2 * q:], 1)[0]
        tail = _follow(cols[-1], ys[-1], step, np.asarray(cand, bool) & ~wm, reach, slope)
    X = np.concatenate([cols, [t[0] for t in tail]]).astype(float)
    Y = np.concatenate([ys, [t[1] for t in tail]])
    W = np.concatenate([ws, [min(t[2], tail_w) for t in tail]])
    W = W / np.sqrt(1 + np.gradient(nd.gaussian_filter1d(Y, 3.0, mode='nearest')) ** 2)   # across, not down
    sel = np.arange(0, len(X), every)
    pts = np.column_stack([X[sel], nd.gaussian_filter1d(Y, 2.0, mode='nearest')[sel]])
    w = np.maximum(nd.gaussian_filter1d(W, 2.0, mode='nearest')[sel], min_w)
    k = int(np.argmax(w))
    w[k:] = np.minimum.accumulate(w[k:])            # from the widest point on, a brow only narrows
    line = np.zeros_like(wm)
    for x_, y_ in zip(X.astype(int), Y.astype(int)):
        line[max(y_ - 4, 0):y_ + 5, x_] = True
    return pts, w, nd.binary_dilation(wm, iterations=2) | line


def brow_ring(sh, pts, widths, pad=4.0, smooth=2.0):
    """The paper ring round a brow: the same stroke `pad` px wider, in white. Lay every ring before any brow is
    inked, or one brow's ring cuts into its neighbour. Clip it (sh.clip) where the ring must not cut the hair."""
    P = np.asarray(pts, float)
    if len(P) < 2:
        return
    Q = geom.catmull(nd.gaussian_filter1d(P, smooth, axis=0, mode='nearest') if smooth else P, per=6)
    wq = np.interp(np.linspace(0, 1, len(Q)), np.linspace(0, 1, len(widths)), np.asarray(widths, float) + pad)
    n = geom.normals(Q)
    sh.fill([tuple(p) for p in np.vstack([Q + n * wq[:, None] / 2, (Q - n * wq[:, None] / 2)[::-1]])], mode='white')


def iris_to_rim(iris, dark, reach=6, inset=2, grow=(0.8, 1.6), min_solid=0.91):
    """The iris as the reference draws it: everything inside its own dark rim, nothing past it.

    A hull of the iris's colours stops short of the rim where the low crescent is pale (it passes for white or
    skin), and runs past it where the white's shading has the iris's colours. Here the dark pixels within
    `reach` px of `iris` are closed into the rim, the rim's inside is filled, and `inset` px are taken off so
    the rim's own ink stays the traced line's. The result is used only when it is a plausible iris (its area
    within `grow` of the input, solidity over `min_solid`, i.e. the rim really closed); otherwise None, and the
    caller keeps its mask. Don't use it on a half-closed iris: the lid cuts it, and there is no round rim."""
    im = np.asarray(iris, bool)
    rim = nd.binary_closing(np.asarray(dark, bool) & nd.binary_dilation(im, iterations=reach), iterations=1)
    filled = nd.binary_fill_holes(rim | nd.binary_erosion(im, iterations=3))
    enc = nd.binary_erosion(filled, iterations=inset) & nd.binary_dilation(im, iterations=reach)
    if not (grow[0] * im.sum() < enc.sum() < grow[1] * im.sum()):
        return None
    ys, xs = np.nonzero(enc)
    if len(xs) < 3:
        return None
    from scipy.spatial import ConvexHull
    try:
        hv = np.column_stack([xs, ys])[ConvexHull(np.column_stack([xs, ys])).vertices]
    except Exception:
        return None
    hm = Image.new('L', (im.shape[1], im.shape[0]), 0)            # the hull drawn in pixels, as the mask is
    ImageDraw.Draw(hm).polygon([tuple(map(float, p)) for p in hv], fill=1)
    return enc if enc.sum() > min_solid * np.asarray(hm, bool).sum() else None


def eye_white(sh, white, lash, depth=7.0, level=0.26, lpi=70):
    """The white of the eye stays paper; the upper lid casts a thin shadow on it. `white` is the white's mask,
    `lash` the lash band above it, `depth` the shadow's height in px. Pass the lash band only, not every dark
    pixel: the shadow falls below whatever `lash` holds, so an iris's lower rim in it puts tone under the iris."""
    d = int(round(depth))
    sh_m = np.zeros_like(white)
    acc = np.asarray(lash, bool).copy()
    for _ in range(d):
        acc = np.roll(acc, 1, axis=0); acc[0] = False
        sh_m |= acc
    sh_m &= white
    if sh_m.any():
        tone(sh, sh_m, level=level, lpi=lpi)
    return sh_m


def iris(sh, mask, levels, pupil=None, lights=None, rim=0.6, streaks=7, lpi=60, solid=0.8, seed=0):
    """An iris in cel layers. `mask` is the visible iris, `levels` a tone level per pixel (dark under the lid,
    lighter below, the bright crescent near the bottom). Layers at or above `solid` are inked black, as the
    lid's shadow on a manga eye is; the rest is tone. Over them: fine streaks fanning down from the pupil, a
    firm rim, the pupil in black, and the catch-lights scraped back to paper last of all."""
    from . import prep
    m = _smooth(mask, 0.8)
    if not m.any():
        return
    rng = np.random.default_rng(seed)
    levels = np.asarray(levels, np.float32)
    dark = _smooth(m & (levels >= solid), 1.0) & m
    lv = nd.gaussian_filter(np.where(m, np.where(levels >= solid, 0.5, levels), 0).astype(np.float32), 0.8)
    sh.fill(m, mode='white')
    tone(sh, m & ~dark, level=np.where(m, lv, 0).astype(np.float32), lpi=lpi)
    if dark.any():
        sh.fill(dark)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    rx, ry = (np.ptp(xs) + 1) / 2, (np.ptp(ys) + 1) / 2
    if pupil is not None and pupil.any():
        py_, px_ = np.nonzero(pupil); cx, cy = px_.mean(), py_.mean()
    if streaks and min(rx, ry) > 9:
        with sh.clip(m):
            for a in np.linspace(0.12 * np.pi, 0.88 * np.pi, streaks) + rng.normal(0, 0.05, streaks):
                r0, r1 = 0.45 * min(rx, ry), 1.1 * max(rx, ry)
                sh.stroke([(cx + r0 * np.cos(a), cy + r0 * np.sin(a)), (cx + r1 * np.cos(a), cy + r1 * np.sin(a))],
                          pen='maru', pressure=0.25, taper=(0.3, 0.6), smooth=False, seed=int(rng.integers(1 << 30)))
    if rim:
        prep.ink_outline(sh, m, pen='g_pen', pressure=(rim * 0.8, rim), sigma=1.0, min_len=6, seed=seed)
    if pupil is not None and pupil.any():
        sh.fill(_smooth(pupil, 0.8))
    if lights is not None and lights.any():
        L = _smooth(lights, 0.7)
        sh.fill(L, mode='white')


def manga_iris(sh, mask, pupil=None, lights=None, solid=0.26, fade=0.66, peak=0.7, pupil_r=0.13, sparkles=3,
               rim=0.65, lpi=60, seed=0):
    """An iris the way a manga inker draws it, not the colour art's cel layers: mostly paper. Black under the
    lid down to `solid` of its height, then a tone fading from `peak` to nothing by `fade`; a small round
    pupil (`pupil_r` of the iris's width, centred where the reference's pupil is); a few small sparkle dots
    on an arc low in the iris; a firm rim; catch-lights scraped back to paper last. Measured on a published
    page: the iris carries far less grey than the colour reference suggests, and that is what makes it read
    as drawn rather than traced."""
    from . import prep
    m = _smooth(mask, 0.8)
    if not m.any():
        return
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(m)
    y0, h = ys.min(), max(np.ptp(ys), 1)
    cx, cy = xs.mean(), ys.mean()
    rx, ry = (np.ptp(xs) + 1) / 2, (np.ptp(ys) + 1) / 2
    if pupil is not None and np.asarray(pupil).any():
        py_, px_ = np.nonzero(pupil); cx, cy = px_.mean(), py_.mean()
    yy = (np.arange(m.shape[0], dtype=np.float32)[:, None] - y0) / h
    yy = np.broadcast_to(yy, m.shape)
    sh.fill(m, mode='white')
    dark = m & (yy < solid)
    t = np.clip((yy - solid) / max(fade - solid, 1e-3), 0, 1)
    lv = np.where(m & ~dark, peak * (1 - t) ** 1.6, 0).astype(np.float32)
    if (lv > 0.02).any():
        tone(sh, m & ~dark & (lv > 0.02), level=lv, lpi=lpi)
    if dark.any():
        sh.fill(_smooth(dark, 1.0) & m)
    r = max(pupil_r * rx * 2, 1.6)
    yy2, xx2 = np.mgrid[:m.shape[0], :m.shape[1]]
    sh.fill(((xx2 - cx) ** 2 + (yy2 - cy) ** 2 <= r * r) & m)
    if sparkles and min(rx, ry) > 6:                 # small dots on an arc low in the iris
        for a in np.linspace(0.62 * np.pi, 0.38 * np.pi, sparkles) + rng.normal(0, 0.04, sparkles):
            px, py = cx + 0.62 * rx * np.cos(a), cy + 0.62 * ry * np.sin(a)
            d = max(0.07 * rx, 0.9)
            if m[int(py), int(px)]:
                sh.stroke([(px - d, py), (px + d, py)], pen='maru', pressure=0.3, taper=(0.4, 0.4), smooth=False)
    if rim:
        prep.ink_outline(sh, m, pen='g_pen', pressure=(rim * 0.8, rim), sigma=1.0, min_len=6, seed=seed)
    if lights is not None and np.asarray(lights).any():
        sh.fill(_smooth(lights, 0.7), mode='white')


def lid_dashes(mask, dash=10, gap=6):
    """A lower lid as an inker draws it: a few short dashes, not a closed line. Cuts `mask` (a thin lid line)
    into pieces `dash` px long with `gap` px between, along its longer axis, and keeps the middle of the run."""
    m = np.asarray(mask, bool)
    out = np.zeros_like(m)
    lab, n = nd.label(nd.binary_dilation(m, iterations=2) & m | m)
    for k, sl in enumerate(nd.find_objects(lab), 1):
        piece = lab[sl] == k
        w = piece.shape[1]
        cols = np.arange(w)
        keep = ((cols // 1) % (dash + gap)) < dash
        lo, hi = int(0.12 * w), int(0.88 * w)        # the ends of a lid line fade out
        keep[:lo] = False; keep[hi:] = False
        out[sl] |= piece & keep[None, :]
    return out


def open_mouth(sh, inside, tongue=None, teeth=None, tongue_level=0.3, lpi=60, outline=True, seed=0,
               tongue_core=None, core_level=0.34, teeth_shade=None, shade_level=0.18, lip=(0.45, 0.7)):
    """An open mouth: the inside in solid black, the tongue a light tone, the teeth left as clean paper, and one
    line round the whole opening. `tongue_core` is a darker middle of the tongue (its rim then reads as the
    lighter edge); `teeth_shade` is the part of the teeth in shadow (the corners). The tongue gets its own line
    only where it meets the inside of the mouth, so the lip is never drawn twice. All masks are page px."""
    from . import prep
    m = _smooth(inside, 1.0)
    whole = m.copy()
    tg = tt = None
    if tongue is not None and tongue.any():
        tg = _smooth(tongue, 1.0)
        whole |= tg
    if teeth is not None and teeth.any():
        tt = _smooth(teeth, 0.8)
        whole |= tt
    if whole.any():
        whole = nd.binary_fill_holes(nd.binary_closing(whole, iterations=2))
    if m.any():
        sh.fill(m)
    if tg is not None:
        sh.fill(tg, mode='white')
        tone(sh, tg, level=tongue_level, lpi=lpi)
        if tongue_core is not None and tongue_core.any():
            tone(sh, _smooth(tongue_core, 1.5) & tg, level=core_level, lpi=lpi)
        with sh.clip(nd.binary_erosion(whole, iterations=4)):
            prep.ink_outline(sh, tg, pen='maru', pressure=(0.5, 0.8), sigma=1.0, min_len=8, seed=seed)
    if tt is not None:
        sh.fill(tt, mode='white')
        if teeth_shade is not None and teeth_shade.any():
            tone(sh, _smooth(teeth_shade, 1.0) & tt, level=shade_level, lpi=lpi)
    if outline and whole.any():
        # the lips: one line round the whole opening, so the teeth are held in
        prep.ink_outline(sh, whole, pen='g_pen', pressure=lip, sigma=1.2, min_len=10, seed=seed + 1)


def blush(sh, center, rx, ry, n=7, slant=-62.0, pen='maru', pressure=0.32, length=0.75, seed=0, clip=None,
          soft=False):
    """A blush as manga draws it: `n` short slanted strokes across the cheek inside an ellipse (page px), longest
    in the middle. More strokes, closer together, for a deeper flush; `slant` is their angle in degrees.
    `soft` makes the strokes fade at both ends and toward the sides, for hatching laid over `blush_tone`."""
    rng = np.random.default_rng(seed)
    cx, cy = center
    a = np.deg2rad(slant)
    dx, dy = np.cos(a), np.sin(a)

    def run():
        for i in range(n):
            f = (i + 0.5) / n * 2 - 1                                   # -1 .. 1 across the cheek
            x = cx + f * rx * 0.92 + rng.normal(0, rx * 0.02)
            L = ry * length * np.sqrt(max(1 - f * f, 0.08)) * rng.uniform(0.8, 1.1)
            y = cy + rng.normal(0, ry * 0.06)
            if soft:
                # over a graded tone: each stroke thins out over most of its length and the strokes lighten
                # toward the ends of the band, so the hatching fades with the tone under it
                t = 2 * L / sh.ppm0 * 0.42
                sh.stroke([(x - dx * L, y - dy * L), (x + dx * L, y + dy * L)], pen=pen,
                          pressure=pressure * (1 - 0.55 * f * f), taper=(t, t), smooth=False,
                          seed=int(rng.integers(1 << 30)))
                continue
            sh.stroke([(x - dx * L, y - dy * L), (x + dx * L, y + dy * L)], pen=pen, pressure=pressure,
                      taper=(0.5, 0.7), smooth=False, seed=int(rng.integers(1 << 30)))
    if clip is not None:
        with sh.clip(clip):
            run()
    else:
        run()


def blush_tone(sh, mask, peak=0.24, soft_px=14, lpi=60, keep=None):
    """A deep flush as a graded tone: strongest in the middle of `mask` (page px) and fading to paper over
    `soft_px` at its edge, so the band has no outline. `keep` limits where it may fall (not on eyes or hair)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return
    d = nd.distance_transform_edt(m)
    t = np.clip(d / float(soft_px), 0, 1)
    lv = (peak * t * t * (3 - 2 * t)).astype(np.float32)                # smoothstep: no visible step either end
    lv = nd.gaussian_filter(lv, 2.0)
    area = m if keep is None else m & np.asarray(keep, bool)
    tone(sh, area & (lv > 0.02), level=lv, lpi=lpi)


def gloom(sh, mask, level=(0.42, 0.14), lpi=34, lines=18, seed=0):
    """The shadow that falls over a frightened face: vertical line tone, heaviest at the top and thinning
    downward, with a few drawn lines falling from its upper edge. `mask` is page px."""
    m = np.asarray(mask, bool)
    if not m.any():
        return
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(m)
    y0, y1 = ys.min(), ys.max()
    yy = np.arange(m.shape[0], dtype=np.float32)[:, None]
    lv = level[0] + (level[1] - level[0]) * np.clip((yy - y0) / max(y1 - y0, 1), 0, 1)
    tone(sh, m, level=np.where(m, lv, 0).astype(np.float32), lpi=lpi, kind='line', angle=0)
    with sh.clip(m):
        for x in np.linspace(xs.min(), xs.max(), lines) + rng.normal(0, 3, lines):
            col = m[:, int(np.clip(x, 0, m.shape[1] - 1))]
            if not col.any():
                continue
            top = int(np.argmax(col))
            L = (y1 - y0) * rng.uniform(0.25, 0.7)
            sh.stroke([(x, top), (x, top + L)], pen='maru', pressure=0.45, taper=(0.0, 1.5), smooth=False,
                      seed=int(rng.integers(1 << 30)))


def sweat(sh, at, size, tilt=0.0):
    """A sweat drop at `at` (its centre, page px), `size` px tall: pointed above, round below, paper inside,
    with a small curved highlight line. `tilt` in degrees leans it."""
    cx, cy = at
    h, w = size, size * 0.62
    t = np.linspace(0, 2 * np.pi, 40)
    x = w / 2 * np.sin(t) * (1 - 0.55 * (np.cos(t) + 1) / 2) ** 1.0
    y = -h / 2 * np.cos(t) * np.where(np.cos(t) > 0, 1.25, 0.85) + h * 0.1
    a = np.deg2rad(tilt)
    X = cx + x * np.cos(a) - y * np.sin(a)
    Y = cy + x * np.sin(a) + y * np.cos(a)
    pts = list(zip(X, Y))
    sh.fill(pts, mode='white')
    sh.stroke(np.column_stack([X, Y, np.full(len(X), 0.55)]), pen='maru', taper=(0, 0))
    hl = [(cx - w * 0.18, cy + h * 0.05), (cx - w * 0.2, cy + h * 0.22), (cx - w * 0.05, cy + h * 0.33)]
    sh.stroke(hl, pen='maru', pressure=0.3, taper=(0.4, 0.6))
