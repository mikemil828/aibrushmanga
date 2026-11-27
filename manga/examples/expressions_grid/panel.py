"""Nine expressions of one boy, as a single manga panel divided 3 x 3, drawn from ref.png.

    python3 prep.py && python3 panel.py        # writes panel.png and page.jsonl
    python3 review.py                          # eyes.png, mouths.png and more for checking

Plan (decided before drawing).
  What matters: the same head nine times, so everything that changes is the expression: brows, eyes, mouth,
  and on three faces sweat and blush. Those get the black and the care. Hair, ears, neck and the collar are
  the frame and stay quiet, but the collar is a big dark shape that must not out-shout the eyes.
  - Hair: spiky, dark teal with lighter teal shine. One dark dot tone; the shine a light tone inside it, so the
    hair keeps its painted streaks; strand lines in ink; one G-pen outline.
  - Collar: navy. A dark tone, with its deepest folds (the inside of the collar) solid black; the zip a
    hairline. Darker than the hair, so the head sits on it, but tone, not a black slab.
  - Skin: paper. Lid folds, nose, ear hollows, the neck under the jaw: one light tone.
  - Brows: the reference's own wedges (hair-coloured pieces on the skin), inked solid with a paper ring.
  - Eyes: lash bands solid black; the white stays paper with the lid's shadow; the iris in cel layers (dark
    under the lid, amber below, pupil, catch-lights). The shocked face's eyes are big whites with a tiny
    pupil: nothing goes on them.
  - Mouths: open ones black inside with a toned tongue and paper teeth; the gritted teeth are paper with their
    own lines; closed mouths are single maru lines.
  - Blush: a few soft hatch strokes for the laughing flush; for the embarrassed face a graded tone band with
    soft hatching over it.
  - Sweat: the reference's own drops, outlined, left paper.
  - The dividers: the reference's thin lines, kept as the panel's inner borders.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
import numpy as np
from scipy import ndimage as nd
from PIL import Image
from manga import Sheet, tone, trace, effects, face, tools
from manga import prep as mprep
import prep as ref

t0 = time.time()
PPM = 8.0
M = 88                                         # margin, px (11 mm)
W, H = ref.PW + 2 * M, ref.PH + 2 * M
LPI = 60
N = 9
sh = Sheet(W, H, px_per_mm=PPM, journal=sys.argv[1] if len(sys.argv) > 1 else 'page.jsonl', seed=5)
if not os.path.exists('parts.npz'):
    rgb, P = ref.read(); np.savez_compressed('parts.npz', **P)
P = dict(np.load('parts.npz'))
PANEL = [(M, M), (W - M, M), (W - M, H - M), (M, H - M)]
LUM = np.asarray(Image.open('ref.png').convert('L').crop((ref.X0, ref.Y0, ref.X1, ref.Y1))
                 .resize((ref.PW, ref.PH), Image.LANCZOS), np.float32)


def full(m):
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
    from PIL import ImageDraw
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


def big(m, area):
    lab, n = nd.label(m)
    sz = nd.sum(m, lab, np.arange(1, n + 1))
    return np.isin(lab, 1 + np.nonzero(sz > area)[0])


Z = {k: np.zeros((ref.PH, ref.PW), bool) for k in ('BROWZ', 'EYEZ', 'MOUTHZ', 'CHEEKZ', 'FACE', 'INNER', 'LOW')}
for i in range(N):
    Z['BROWZ'] |= cellmask(i, 0.17, 0.37, 0.22, 0.78)
    Z['EYEZ'] |= cellmask(i, 0.26, 0.54, 0.20, 0.80)
    Z['MOUTHZ'] |= cellmask(i, 0.55, 0.80, 0.32, 0.68)
    Z['CHEEKZ'] |= cellmask(i, 0.40, 0.68, 0.20, 0.80)
    Z['FACE'] |= cellmask(i, 0.15, 0.80, 0.20, 0.80)
    Z['INNER'] |= cellmask(i, 0.16, 0.80, 0.26, 0.74)
    Z['LOW'] |= cellmask(i, 0.76, 1.0)
BROWZ, EYEZ, MOUTHZ, CHEEKZ, FACE, INNER, LOW = (Z[k] for k in ('BROWZ', 'EYEZ', 'MOUTHZ', 'CHEEKZ', 'FACE',
                                                                 'INNER', 'LOW'))

# ---- brows are hair-coloured pieces lying on the skin, longer than tall, not joined to the hair's mass
bulk = nd.binary_dilation(big(nd.binary_erosion(P['hair'], iterations=6), 15000), iterations=8) & P['hair']
brows = np.zeros_like(bulk)
for sl, piece in comps(nd.binary_opening(P['hair'] & ~bulk & BROWZ, iterations=1), 120):
    wm = whole(sl, piece)
    ring = nd.binary_dilation(wm, iterations=6) & ~wm
    level = (sl[1].stop - sl[1].start) > 1.5 * (sl[0].stop - sl[0].start)
    if level and (ring & (P['skin'] | P['shadow'] | P['line'])).sum() > 0.5 * ring.sum():
        brows |= wm
brows = brows | (P['black'] & nd.binary_dilation(brows, iterations=2))
# the hair's deepest shadow (under the locks, by the ears) is near black: outside the face it is still hair
under = nd.binary_dilation(P['hair'] & ~brows, iterations=6, mask=P['black'] & ~INNER
                           & ~nd.binary_dilation(P['skin'] | P['shadow'], iterations=3))   # not the ears' lines
mass = big((P['hair'] | under) & ~brows, 500)      # small pieces too: the locks hanging by the jaw
hair_m = nd.gaussian_filter(nd.binary_closing(mass, iterations=1).astype(np.float32), 0.9) > 0.5
hair_t = nd.binary_dilation(hair_m, iterations=2)
jacket = nd.gaussian_filter(nd.binary_closing(P['jacket'] & LOW, iterations=2).astype(np.float32), 1.0) > 0.5
DARK = P['black'] | P['brown']
IRIS = np.zeros_like(mass); BAND = IRIS.copy(); LIGHTS = IRIS.copy(); MOUTH = IRIS.copy(); SMALL = IRIS.copy(); TEETH = IRIS.copy()

with sh.clip(PANEL):
    # 1. skin shadow: one light tone
    # the blush fades through shadow-coloured pixels at its edge: no shadow tone round a blush
    blush_zone = nd.binary_dilation(nd.binary_closing(P['blush'], iterations=6), iterations=12)
    shadow = (P['shadow'] | P['brown'] & ~EYEZ & ~MOUTHZ) & ~blush_zone
    # the neck's shadow is speckled in the reference (two close skin tones): close it into one shape
    shadow = nd.binary_opening(shadow, iterations=1) | nd.binary_opening(nd.binary_closing(shadow & LOW, iterations=4),
                                                                          iterations=3)
    shadow = nd.gaussian_filter(shadow.astype(np.float32), 1.2) > 0.5
    tone(sh, full(shadow & ~hair_m & ~jacket), level=0.20, lpi=LPI)
    # 2. hair: dark tone, its shine a light tone; the collar darker still, its deepest folds black
    shine = nd.binary_opening(P['shine'] & hair_m, iterations=1)
    tone(sh, full(hair_t & ~shine), level=0.62, lpi=LPI)
    tone(sh, full(shine), level=0.30, lpi=LPI)
    tone(sh, full(jacket), level=0.74, lpi=LPI)
    deep = nd.binary_opening(jacket & (LUM < 36), iterations=2)
    sh.fill(full(nd.gaussian_filter(deep.astype(np.float32), 1.0) > 0.5))

    for i in range(N):
        cm = cellmask(i)
        ez, mz, cz = EYEZ & cm, MOUTHZ & cm, CHEEKZ & cm
        # 3. blush: a wide band is embarrassment (graded tone and soft hatching); patches are a flush
        bl = nd.binary_opening(P['blush'] & cz, iterations=2)
        band = [c for c in comps(nd.binary_closing(bl, iterations=6), 9000) if (c[0][1].stop - c[0][1].start) > 0.3 * (ref.cell(i)[2] - ref.cell(i)[0])]
        eyes_off = ~nd.binary_dilation(big(P['white'] & ez, 150) | P['amber'] | DARK & ez, iterations=3) & ~hair_m
        for sl, piece in band:
            bm = nd.gaussian_filter(whole(sl, piece).astype(np.float32), 4) > 0.5
            BAND |= nd.binary_dilation(bm, iterations=6)
            face.blush_tone(sh, full(nd.binary_dilation(bm, iterations=14)), peak=0.36, soft_px=30, lpi=LPI,
                            keep=full(eyes_off))
            for side in (0, 1):                          # the band's hatching sits on each cheek
                ys, xs = np.nonzero(bm)
                mid = np.median(xs)
                half = bm & ((np.arange(ref.PW)[None, :] < mid) if side == 0 else (np.arange(ref.PW)[None, :] >= mid))
                hy, hx = np.nonzero(half)
                if len(hx) < 50:
                    continue
                face.blush(sh, (hx.mean() + M, hy.mean() + M), np.ptp(hx) / 2 * 0.75, np.ptp(hy) / 2 * 1.1,
                           n=11, pressure=0.42, soft=True, seed=20 + i + 10 * side,
                           clip=full(nd.binary_dilation(bm, iterations=8) & eyes_off))
        if not band:
            for sl, piece in comps(nd.binary_closing(bl, iterations=3), 700):
                ys, xs = np.nonzero(whole(sl, piece))
                face.blush(sh, (xs.mean() + M, ys.mean() + M), np.ptp(xs) / 2 * 0.8, np.ptp(ys) / 2 * 0.9, n=5,
                           pressure=0.24, seed=20 + i)
        # 4. eyes: one per cluster of iris colour; the visible iris is the hull of its coloured layers
        amber = P['amber'] & ez
        coloured = (P['amber'] | P['brown'] | P['shadow']) & ez & ~nd.binary_dilation(P['skin'], iterations=1)
        for sl, piece in comps(nd.binary_dilation(amber, iterations=6), 60):
            core = amber & nd.binary_dilation(whole(sl, piece), iterations=2)
            cl, _ = nd.label(nd.binary_opening(coloured, iterations=1) | core)
            ids = np.unique(cl[core]); body = np.isin(cl, ids[ids > 0])
            hl = hull(body)
            half = np.ptp(np.nonzero(hl)[0]) < 0.065 * (ref.cell(i)[3] - ref.cell(i)[1])
            lights = np.zeros_like(hl)
            for wsl, wpiece in comps(P['white'] & nd.binary_dilation(hl, iterations=2)):
                wm = whole(wsl, wpiece)
                grown = nd.binary_dilation(wm, iterations=2) & P['white']
                if wpiece.sum() < 500 and grown.sum() < wpiece.sum() + 30 and (wm & hl).sum() > 0.5 * wpiece.sum():
                    lights |= wm
            # the pale yellow low in the iris passes for skin: only skin at the hull's edge is cut away
            im = nd.binary_opening(hl & ~(P['white'] & ~lights) & ~(P['skin'] & ~nd.binary_erosion(hl, iterations=4)),
                                   iterations=2)
            if im.sum() < 60 or (nd.binary_dilation(im, iterations=6) & P['black']).sum() < 40 \
                    or (im & P['blush']).sum() > 0.2 * im.sum():
                continue                                 # an iris sits against a lash band, not in a blush
            IRIS |= im
            ys_i = np.nonzero(im)[0]
            # the reference's iris is a gradient: dark brown under the lid, amber low down. By position, not by
            # colour (its middle tones fall between the classes): solid under the lid, then tone fading down.
            yy = (np.arange(ref.PH, dtype=np.float32)[:, None] - ys_i.min()) / max(np.ptp(ys_i), 1)
            yy = np.broadcast_to(yy, im.shape)
            lv = np.zeros(im.shape, np.float32)
            if half:
                lv[im] = np.where(yy[im] < 0.36, 0.55, 0.22)
            else:
                lv[im] = np.where(yy[im] < 0.16, 0.86, 0.62 - 0.40 * np.clip((yy[im] - 0.16) / 0.7, 0, 1))
                lv[im & P['amber'] & (yy > 0.5)] = 0.16
                lv[im & P['skin'] & (yy > 0.5)] = 0.08
            # the pupil: the black piece nearest the iris's middle, not the lash band lying across its top
            pup = nd.binary_opening(P['black'] & nd.binary_erosion(im, iterations=2)
                                    & (yy > 0.2), iterations=1)
            cy, cx = np.nonzero(im)[0].mean(), np.nonzero(im)[1].mean()
            pc = [c for c in comps(pup, 10) if c[0][0].start > ys_i.min() + 2]
            pup = whole(*min(pc, key=lambda c: abs((c[0][0].start + c[0][0].stop) / 2 - cy) +
                             abs((c[0][1].start + c[0][1].stop) / 2 - cx))) if pc else np.zeros_like(im)
            if half:
                SMALL |= nd.binary_dilation(im, iterations=1) & ~pup
            face.iris(sh, full(im), full(lv), pupil=full(pup), lights=full(lights), seed=60 + i,
                      rim=0.3 if half else 0.6, streaks=3 if half else 6)
            LIGHTS |= lights
        # (the reference draws its own lower lids: they come with the thin lines)
        face.eye_white(sh, full(P['white'] & ez & ~IRIS & ~LIGHTS), full(DARK & ez), depth=4, level=0.2)
        # 5. mouth: the opening is the hull of the dark inside, tongue and teeth; white pieces are teeth
        # the inside of an open mouth is a thick dark piece; the jaw's outline crossing the zone is a thin one
        inside = nd.binary_closing(DARK & mz, iterations=2)
        inside = np.any([whole(*c) for c in comps(inside, 500) if nd.binary_erosion(c[1], iterations=4).sum() > 80]
                        or [np.zeros_like(inside)], axis=0)
        pinkm = nd.binary_opening(P['pink'] & mz, iterations=1)
        whitem = P['white'] & mz
        teethy = nd.binary_opening(whitem, iterations=2)
        if not (inside.any() or teethy.sum() > 1500):
            continue                                     # a closed mouth: a line, drawn with the thin lines
        opening = hull(nd.binary_fill_holes(nd.binary_closing(
            inside | pinkm | whitem & nd.binary_dilation(inside | pinkm, iterations=10), iterations=4)))
        if not inside.any():                             # gritted teeth: the opening is the white itself
            opening = nd.binary_fill_holes(nd.binary_closing(teethy | DARK & mz & nd.binary_dilation(teethy, iterations=3),
                                                             iterations=3))
            opening = np.any([whole(*c) for c in comps(opening, 1500)] or [np.zeros_like(opening)], axis=0)
        tongue = np.zeros_like(mz); teeth = tongue.copy()
        for sl, piece in comps(nd.binary_opening(opening & ~inside, iterations=1), 40):
            pm = whole(sl, piece)
            if (pm & P['white']).sum() > 0.45 * pm.sum():
                teeth |= pm
            else:
                tongue |= pm
        shade = nd.binary_opening(teeth & (LUM < 222), iterations=1)
        face.open_mouth(sh, full(opening & ~tongue & ~teeth), tongue=full(tongue), teeth=full(teeth), seed=80 + i,
                        tongue_level=0.42, tongue_core=full(pinkm & tongue), core_level=0.26,
                        teeth_shade=full(shade), shade_level=0.22, lip=(0.35, 0.55))
        MOUTH |= opening
        TEETH |= teeth
        if not inside.any():                             # the gritted teeth's own divisions
            with sh.clip(full(nd.binary_erosion(opening, iterations=2))):
                trace.ink_line_art(sh, full(P['line_soft'] & opening & ~P['white']), pens=('maru', 'maru'),
                                   min_len_px=4, beta_px=99, weight=0.6, min_px=1.0, even=0.9, seed=85 + i)

    # 6. solid blacks and drawn black lines: lash bands, pupils, the face's outline
    hair_edge = nd.binary_dilation(hair_t, iterations=4) & ~INNER
    open_m = nd.binary_dilation(MOUTH, iterations=7)
    lash = nd.binary_closing(DARK & EYEZ, iterations=2) & ~IRIS & ~P['white'] & ~P['skin']
    ll, ln = nd.label(lash)
    lash_big = np.isin(ll, 1 + np.nonzero(nd.sum(lash, ll, np.arange(1, ln + 1)) > 500)[0])
    blk = (P['black'] | P['brown'] & ~IRIS & EYEZ | lash) & ~nd.binary_dilation(brows, iterations=1)
    blk &= ~(MOUTHZ & ~open_m) & ~hair_edge & ~SMALL & ~open_m & ~jacket & ~nd.binary_erosion(IRIS, iterations=2)   # face.iris owns it
    tips = blk & nd.binary_dilation(hair_t, iterations=5) & ~lash_big
    blk &= ~tips
    n1, _ = trace.ink_line_art(sh, full(blk), pens=('maru', 'g_pen'), split_mm=0.22, min_len_px=6, beta_px=3.0,
                               weight=1.0, min_px=1.0, even=0.6, seed=7)
    nt, _ = trace.ink_line_art(sh, full(tips), pens=('maru', 'maru'), min_len_px=8, beta_px=99, weight=0.45,
                               min_px=1.0, even=0.95, seed=14)
    # 6b. brows over the lashes: each the reference's own wedge with a paper ring
    wedges = []
    for i in range(N):
        for sl, piece in comps(nd.binary_closing(brows & cellmask(i), iterations=2), 120):
            if (sl[0].stop - sl[0].start) > (sl[1].stop - sl[1].start):
                continue
            wedges.append(nd.gaussian_filter(nd.binary_dilation(whole(sl, piece), iterations=1)
                                             .astype(np.float32), 0.7) > 0.5)
    for bm in wedges:
        sh.fill(full(nd.binary_dilation(bm, iterations=2) & ~hair_t), mode='white')
    for bm in wedges:
        face.ink_shape(sh, full(bm), sigma=0)
    # 7. thin lines: creases, nose, mouth lines, ears, jaw, sweat drops, the zip
    free = ~nd.binary_dilation(DARK, iterations=2) & ~hair_m & ~brows & ~nd.binary_dilation(IRIS, iterations=2) \
        & ~hair_edge & ~jacket & ~P['bg']
    thin = P['line'] & free & ~open_m & ~(BAND & ~P['black'])      # the band's edge is not a line
    n2, _ = trace.ink_line_art(sh, full(thin & ~MOUTHZ), pens=('maru', 'saji'), split_mm=0.2, min_len_px=5,
                               beta_px=99, weight=0.8, min_px=1.2, even=0.8, seed=8)
    mouth_l = nd.binary_closing((P['line_soft'] | DARK) & MOUTHZ & ~open_m & ~P['bg'], iterations=2)
    n2m, _ = trace.ink_line_art(sh, full(mouth_l), pens=('maru', 'g_pen'), split_mm=0.12, min_len_px=8, beta_px=99,
                                weight=0.85, min_px=1.7, even=0.7, seed=12)
    zipl = nd.binary_dilation(jacket, iterations=3) & (P['zip'] | P['line_soft'] & ~P['jacket']) & ~P['bg'] & ~hair_t
    with sh.clip(full(nd.binary_dilation(jacket, iterations=3))):
        nz, _ = trace.ink_line_art(sh, full(zipl & ~P['skin']), pens=('maru', 'maru'), min_len_px=10, beta_px=99,
                                   weight=0.5, min_px=1.0, even=0.9, seed=17)
    # 8. hair strand lines and the outlines of hair and collar
    strands = nd.binary_opening(P['strand'] & nd.binary_erosion(hair_m, iterations=3) & ~shine, iterations=1)
    n3, _ = trace.ink_line_art(sh, full(strands), pens=('maru', 'maru'), min_len_px=24, beta_px=99, weight=0.6,
                               min_px=1.0, even=0.8, seed=9)
    with sh.clip(full(~INNER)):
        n4 = mprep.ink_outline(sh, full(hair_t), pen='g_pen', pressure=(0.35, 0.6), sigma=1.5, min_len=30, seed=10)
    n5 = mprep.ink_outline(sh, full(jacket), pen='g_pen', pressure=(0.4, 0.65), sigma=1.5, min_len=30, seed=11)
    sh.fill(full(nd.gaussian_filter(LIGHTS.astype(np.float32), 0.7) > 0.5), mode='white')

# 9. the reference's dividers, as the panel's inner borders
for x in ref.COLS[1:-1]:
    px = M + (x - ref.X0) * ref.S
    sh.stroke([(px, M), (px, H - M)], pen='maru', pressure=0.55, taper=(0, 0), smooth=False)
for y in ref.ROWS[1:-1]:
    py = M + (y - ref.Y0) * ref.S
    sh.stroke([(M, py), (W - M, py)], pen='maru', pressure=0.55, taper=(0, 0), smooth=False)
tools.keep(brows=brows, iris=IRIS, small=SMALL, lights=LIGHTS, blk=blk, tips=tips, mouth=MOUTH, teeth=TEETH,
           jacket=jacket)
effects.border(sh, PANEL, 0.8)
sh.png('panel.png'); sh.close()
print('strokes', sh.stats['strokes'], 'traced', n1 + nt + n2 + n2m + nz + n3 + n4 + n5, round(time.time() - t0), 's')
