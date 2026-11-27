"""Eight expressions of one face, as a single manga panel drawn from ref.png.

    python3 prep.py && python3 panel.py        # writes panel.png and page.jsonl

Plan (decided before drawing).
  What matters: the eight faces are the same head, so everything that changes is the expression, and the
  expression lives in three places: brows, eyes, mouth. Those get the black and the care. Hair, neck and shirt
  are the frame and stay quiet.
  - Hair: one dark dot tone, not solid black, so the black brows that cross the fringe still read and the
    eyes stay the darkest things on the sheet. Strand lines in ink, a ring of scraped shine on the crown.
  - Skin: paper. The fringe's shadow, the neck under the chin, the lid fold and the nose get one light tone.
  - Brows: solid, tapered shapes (face.brow) from each brow's measured centre line and width.
  - Eyes: lash bands solid black; the white stays paper with the lid's thin shadow; the iris in cel layers,
    dark under the lid and light below, with rim, streaks, pupil and catch-lights (face.iris). The whites of
    the wide eyes (surprise, fear) are the biggest white on those faces: nothing goes on them.
  - Mouths: open ones are black inside with a toned tongue and paper teeth (face.open_mouth); closed ones are
    single lines traced with the maru pen.
  - Blush: slanted hatching, a few strokes for a flush, many and a tone under them for embarrassment.
  - Fear: the gloom band in vertical line tone with falling lines, and three sweat drops.
  - Thin lines (creases, nose, jaw, ears, shirt): maru and G-pen strokes following the reference's own lines.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
import numpy as np
from scipy import ndimage as nd
from manga import Sheet, tone, trace, effects, face
from manga import prep as mprep
import prep as ref

t0 = time.time()
PPM = 8.0
M = 88                                         # margin, px (11 mm)
W, H = ref.PW + 2 * M, ref.PH + 2 * M
LPI = 60
sh = Sheet(W, H, px_per_mm=PPM, journal=sys.argv[1] if len(sys.argv) > 1 else 'page.jsonl', seed=5)
if not os.path.exists('parts.npz'):
    rgb, P = ref.read(); np.savez_compressed('parts.npz', **P)
P = dict(np.load('parts.npz'))
PANEL = [(M, M), (W - M, M), (W - M, H - M), (M, H - M)]


def full(m):
    """A panel-sized array placed on the sheet."""
    out = np.zeros((H, W), m.dtype)
    out[M:M + ref.PH, M:M + ref.PW] = m
    return out


def cellmask(i, y0=0.0, y1=1.0, x0=0.0, x1=1.0):
    a, b, c, d = ref.cell(i)
    m = np.zeros((ref.PH, ref.PW), bool)
    m[int(b + (d - b) * y0):int(b + (d - b) * y1), int(a + (c - a) * x0):int(a + (c - a) * x1)] = True
    return m


def comps(m, min_area=0):
    lab, n = nd.label(m)
    out = []
    for k, sl in enumerate(nd.find_objects(lab), 1):
        piece = lab[sl] == k
        if piece.sum() >= min_area:
            out.append((sl, piece))
    return out


def hull(m):
    """Convex hull of a mask (an iris is convex; its catch-lights and pupil lie inside the hull)."""
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return m.copy()
    pts = np.column_stack([xs, ys])
    try:
        hv = pts[ConvexHull(pts).vertices]
    except Exception:
        return m.copy()
    im = Image.new('L', (m.shape[1], m.shape[0]), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in hv], fill=1)
    return np.asarray(im, bool)


def whole(sl, piece):
    m = np.zeros((ref.PH, ref.PW), bool); m[sl] = piece
    return m


# ---- the hair and the thick brows are the same colour in the reference. A brow is a thin, level piece lying on
#      skin; everything else of that colour is hair.
def big(m, area):
    lab, n = nd.label(m)
    sz = nd.sum(m, lab, np.arange(1, n + 1))
    return np.isin(lab, 1 + np.nonzero(sz > area)[0])


bulk = nd.binary_dilation(big(nd.binary_erosion(P['hair'], iterations=7), 20000), iterations=9) & P['hair']

BROWZ = np.zeros_like(bulk); EYEZ = BROWZ.copy(); MOUTHZ = BROWZ.copy(); CHEEKZ = BROWZ.copy(); FACE = BROWZ.copy()
SHIRT = BROWZ.copy()
for i in range(8):
    BROWZ |= cellmask(i, 0.26, 0.47, 0.24, 0.76)
    EYEZ |= cellmask(i, 0.27, 0.60, 0.20, 0.80)
    MOUTHZ |= cellmask(i, 0.47, 0.63, 0.36, 0.66)
    CHEEKZ |= cellmask(i, 0.36, 0.56, 0.20, 0.80)
    FACE |= cellmask(i, 0.20, 0.70, 0.20, 0.80)
    SHIRT |= cellmask(i, 0.74, 1.0)
seed_b = np.zeros_like(bulk)
for sl, piece in comps(nd.binary_opening(P['hair'] & ~bulk & BROWZ, iterations=1), 150):
    wm = whole(sl, piece)
    ring = nd.binary_dilation(wm, iterations=7) & ~wm
    level = (sl[1].stop - sl[1].start) > 1.6 * (sl[0].stop - sl[0].start)
    if level and (ring & P['hair']).sum() < 0.45 * ring.sum():
        seed_b |= wm
brows = nd.binary_dilation(seed_b, iterations=3) & (P['hair'] | P['black']) & ~bulk          # with its dark outline
brows = nd.binary_opening(brows, iterations=1)
mass = big(P['hair'] & ~brows, 5000)
hair_m = nd.gaussian_filter(nd.binary_closing(mass, iterations=1).astype(np.float32), 0.9) > 0.5
DARK = P['black'] | P['maroon']                    # lash bands and mouth insides are black and deep brown
IRIS = np.zeros_like(mass); LIGHTS = IRIS.copy(); MOUTH = IRIS.copy()
n_ink = 0

with sh.clip(PANEL):
    # 1. skin shadow: one light tone (fringe shadow, neck, lid folds, nose, ears)
    shadow = nd.gaussian_filter(nd.binary_opening(P['shadow'], iterations=1).astype(np.float32), 1.2) > 0.5
    tone(sh, full(shadow & ~hair_m), level=0.20, lpi=LPI)
    # 2. hair: a dark tone with an inked outline; strand lines and shine come later
    hair_t = nd.binary_dilation(hair_m, iterations=2)                 # out to the middle of its inked outline
    tone(sh, full(hair_t), level=0.62, lpi=LPI)
    # the shine: a ring of short scraped strokes round the crown, following the hair's fall from the parting
    rng = np.random.default_rng(3)
    for i in range(8):
        hm = hair_m & cellmask(i)
        ys, xs = np.nonzero(hm)
        cx, top, wid = xs.mean(), ys.min(), np.ptp(xs)
        px, py = cx + M, top + M - 0.10 * wid                         # the parting, a little above the crown
        ring = full(nd.binary_erosion(hm, iterations=6) & ~nd.binary_dilation(~hm & cellmask(i, 0.2, 0.6, 0.25, 0.75),
                                                                              iterations=10))
        with sh.clip(ring):
            for a in np.deg2rad(np.linspace(18, 162, 46)):
                if rng.random() < 0.22:
                    continue                                          # gaps: the shine is broken, not a band
                r0 = wid * (0.27 + 0.035 * np.sin(5 * a) + rng.normal(0, 0.012))
                L = wid * rng.uniform(0.035, 0.10)
                p0 = (px + r0 * np.cos(a), py + r0 * np.sin(a) * 1.05)
                p1 = (px + (r0 + L) * np.cos(a), py + (r0 + L) * np.sin(a) * 1.05)
                sh.stroke([p0, p1], pen='white', pressure=rng.uniform(0.10, 0.30), mode='white', taper=(0.8, 1.2),
                          smooth=False, seed=int(rng.integers(1 << 30)))
    # shirt: the folds are soft grey shapes in the reference; here one very light tone
    folds = nd.binary_opening(P['fold'] & SHIRT, iterations=2)
    folds = nd.gaussian_filter(folds.astype(np.float32), 1.5) > 0.5
    tone(sh, full(folds), level=0.14, lpi=LPI)

    for i in range(8):
        cm = cellmask(i)
        ez, mz, cz = EYEZ & cm, MOUTHZ & cm, CHEEKZ & cm
        # 3. fear: the gloom band
        gl = nd.binary_opening(P['gloom'] & FACE & cm, iterations=2)
        gl = np.any([whole(*c) for c in comps(gl, 2500)] or [np.zeros_like(gl)], axis=0)
        if gl.any():
            gl = nd.binary_closing(gl, iterations=5) & ~P['white'] & ~hair_m
            face.gloom(sh, full(gl), seed=40 + i)
        # 4. blush. A wide band of strong pink across the nose is embarrassment: a tone and dense hatching.
        #    Soft pink on each cheek is a flush: a few light strokes.
        strong = [c for c in comps(nd.binary_closing(P['pink'] & cz, iterations=4), 3000)]
        for sl, piece in strong:
            bm = nd.gaussian_filter(hull(whole(sl, piece)).astype(np.float32), 4) > 0.5
            bm &= ~nd.binary_dilation(P['white'] | P['amber'] | DARK, iterations=3) & ~hair_m
            tone(sh, full(bm), level=0.14, lpi=LPI)
            ys, xs = np.nonzero(bm)
            face.blush(sh, (xs.mean() + M, ys.mean() + M), np.ptp(xs) / 2, np.ptp(ys) / 2 * 1.15, n=26, pressure=0.45,
                       seed=20 + i, clip=full(nd.binary_dilation(bm, iterations=8) & ~P['white'] & ~hair_m))
        if not strong:
            for sl, piece in comps(nd.binary_opening(P['blush'] & cz, iterations=3), 1500):
                ys, xs = np.nonzero(whole(sl, piece))
                face.blush(sh, (xs.mean() + M, ys.mean() + M), np.ptp(xs) / 2 * 0.8, np.ptp(ys) / 2 * 0.9, n=6,
                           pressure=0.24, seed=20 + i)
        # 5. eyes: one per cluster of iris colour. The visible iris is the convex hull of its coloured layers
        #    (the lids cut it, so it is not an ellipse); pupil and catch-lights lie inside the hull.
        amber = P['amber'] & ez
        coloured = (P['amber'] | P['maroon'] | P['pink']) & ez
        seeds = nd.binary_dilation(amber, iterations=6)
        for sl, piece in comps(seeds, 150):
            core = amber & nd.binary_dilation(whole(sl, piece), iterations=2)
            cl, _ = nd.label(nd.binary_opening(coloured, iterations=1) | core)
            body = np.isin(cl, np.unique(cl[core])[np.unique(cl[core]) > 0])
            hl = hull(body)
            lights = np.zeros_like(hl)
            for wsl, wpiece in comps(P['white'] & nd.binary_dilation(hl, iterations=2)):
                wm = whole(wsl, wpiece)
                grown = nd.binary_dilation(wm, iterations=2) & P['white']        # part of something bigger?
                if wpiece.sum() < 600 and grown.sum() < wpiece.sum() + 30 and (wm & hl).sum() > 0.6 * wpiece.sum():
                    lights |= wm
            im = nd.binary_opening(hl & ~P['skin'] & ~(P['white'] & ~lights), iterations=2)
            if im.sum() < 60:
                continue
            IRIS |= im
            lv = np.zeros(im.shape, np.float32)
            lv[im] = 0.30
            lv[im & P['pink']] = 0.52
            ys_i = np.nonzero(im)[0]
            small = np.ptp(ys_i) < 30                    # a half-closed eye: all black would lose the iris
            lv[im & P['maroon']] = 0.70 if small else 0.86
            lv[im & P['yellow']] = 0.07
            pup = nd.binary_opening(P['black'] & nd.binary_erosion(im, iterations=3), iterations=2)
            pc = comps(pup, 12)
            pup = whole(*max(pc, key=lambda c: c[1].sum())) if pc else np.zeros_like(im)
            face.iris(sh, full(im), full(lv), pupil=full(pup), lights=full(lights), seed=60 + i)
            LIGHTS |= lights
        # the lower lid: a faint line under the white of each eye, so the eye holds its shape
        for sl, piece in comps(nd.binary_opening(P['white'] & ez & ~LIGHTS, iterations=1), 200):
            om = whole(sl, piece)
            ys_o = np.nonzero(om)[0]
            low = om & (np.arange(ref.PH)[:, None] > ys_o.min() + 0.55 * np.ptp(ys_o))
            edge = nd.binary_dilation(low, iterations=2) & ~nd.binary_erosion(om, iterations=2)
            with sh.clip(full(edge & ~nd.binary_dilation(IRIS, iterations=3))):
                mprep.ink_outline(sh, full(om), pen='maru', pressure=(0.3, 0.45), sigma=1.0, min_len=8, seed=90 + i)
        # the lid's shadow on the white of the eye (not on the frightened face: its whites stay stark)
        if not gl.any():
            face.eye_white(sh, full(P['white'] & ez & ~IRIS), full(DARK & ez), depth=4, level=0.2)
        # 6. mouth: black inside, toned tongue or lips, paper teeth
        tongue = nd.binary_opening(P['pink'] & mz, iterations=1) if not strong else np.zeros_like(mz)
        inside = nd.binary_closing(DARK & mz, iterations=2)
        inside = np.any([whole(*c) for c in comps(inside, 900)] or [np.zeros_like(inside)], axis=0)
        teeth = P['white'] & mz & nd.binary_fill_holes(nd.binary_closing(inside | tongue | P['white'] & mz &
                                                       nd.binary_dilation(inside, iterations=8), iterations=4))
        face.open_mouth(sh, full(inside & ~tongue & ~teeth), tongue=full(tongue), teeth=full(teeth), seed=80 + i)
        MOUTH |= inside | tongue | teeth
        # 7. thick brows: tapered solid shapes from their measured centre lines
        #    (a touch heavier than measured: the brow is the first thing read, and tone round it eats thin ink)
        for sl, piece in comps(nd.binary_closing(brows & cm, iterations=3), 150):
            if (sl[0].stop - sl[0].start) > (sl[1].stop - sl[1].start):
                continue                                    # a strand of the fringe, not a brow
            pts, wd = face.ribbon_of(whole(sl, piece), step=5)
            face.brow(sh, pts + [M, M], np.maximum(wd * 1.1, 3.5))

    # 8. solid blacks and drawn black lines (lash bands, pupils, outlines): thick parts filled, thin stroked
    blush_band = np.zeros_like(mass)
    # the hair's own outline is inked once (step 10): keep the reference's outline and edge lines out of it
    INNER = np.zeros_like(mass)
    for i in range(8):
        INNER |= cellmask(i, 0.22, 0.66, 0.24, 0.76)
    hair_edge = nd.binary_dilation(hair_t, iterations=4) & ~INNER
    open_m = nd.binary_dilation(MOUTH, iterations=4)
    lash = nd.binary_closing(DARK & EYEZ, iterations=2) & ~IRIS & ~P['white'] & ~P['skin']   # flecks in the bands
    blk = (P['black'] | P['maroon'] & ~IRIS | lash) & ~nd.binary_dilation(brows, iterations=1) & ~(MOUTHZ & ~open_m) & ~hair_edge
    n1, _ = trace.ink_line_art(sh, full(blk), pens=('maru', 'g_pen'), split_mm=0.22, min_len_px=6, beta_px=3.0,
                               weight=1.0, min_px=1.0, even=0.6, seed=7)
    # 9. the thin coloured lines, never filled. In the face they carry the expression (mouth, nose, creases,
    #    the knit of the brows), so even the faintest are kept and drawn with a firm hairline; mouths get the
    #    G-pen. The shirt's outline is one quiet line.
    free = ~nd.binary_dilation(DARK, iterations=2) & ~hair_m & ~brows & ~nd.binary_dilation(IRIS, iterations=2) & ~hair_edge
    thin = P['line'] & free
    brow_l = nd.binary_closing(thin & BROWZ, iterations=2)             # the thin brows of the easy faces
    nb, _ = trace.ink_line_art(sh, full(brow_l), pens=('maru', 'g_pen'), split_mm=0.12, min_len_px=14, beta_px=99,
                               weight=0.9, min_px=1.7, even=0.8, seed=13)
    thin &= ~BROWZ
    n2, _ = trace.ink_line_art(sh, full(thin & ~SHIRT & ~MOUTHZ), pens=('maru', 'saji'), split_mm=0.2, min_len_px=5,
                               beta_px=99, weight=0.8, min_px=1.2, even=0.8, seed=8)
    mouth_l = nd.binary_closing((P['line_soft'] | DARK) & MOUTHZ & ~open_m, iterations=2)
    n2m, _ = trace.ink_line_art(sh, full(mouth_l), pens=('maru', 'g_pen'), split_mm=0.12, min_len_px=8, beta_px=99,
                                weight=0.85, min_px=1.7, even=0.7, seed=12)
    n2b, _ = trace.ink_line_art(sh, full(nd.binary_closing(thin & SHIRT, iterations=2)), pens=('saji', 'saji'),
                                split_mm=0.2, min_len_px=30, beta_px=99, weight=0.8, min_px=1.6, even=0.9, seed=11)
    n2 += n2m + nb
    # 10. hair: strand lines in the tone, and the outline
    strands = nd.binary_opening(P['strand'] & nd.binary_erosion(hair_m, iterations=3), iterations=1)
    n3, _ = trace.ink_line_art(sh, full(strands), pens=('maru', 'maru'), min_len_px=28, beta_px=99, weight=0.6,
                               min_px=1.0, even=0.8, seed=9)
    with sh.clip(full(~INNER)):                   # round the face the reference's own fringe lines are enough
        n4 = mprep.ink_outline(sh, full(hair_t), pen='g_pen', pressure=(0.35, 0.6), sigma=1.5, min_len=30, seed=10)
    n_ink = n1 + n2 + n2b + n3 + n4
    sh.fill(full(nd.gaussian_filter(LIGHTS.astype(np.float32), 0.7) > 0.5), mode='white')    # catch-lights stay paper
    # 11. fear: three sweat drops, placed by hand from the reference (ref px)
    for (x, y, size, tilt) in [(174.2, 819.5, 13, 8), (181.5, 843, 14, 6), (311.8, 843, 15, -6)]:
        face.sweat(sh, (M + x * ref.S, M + y * ref.S), size * ref.S, tilt)
effects.border(sh, PANEL, 0.8)
sh.png('panel.png'); sh.close()
print('strokes', sh.stats['strokes'], 'traced lines', n_ink, round(time.time() - t0), 's')
