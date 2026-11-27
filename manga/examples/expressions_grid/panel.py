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
  - Brows: one tapered stroke each, from the thick wedge out along its thin tail, with a paper ring.
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

# STYLE=cel (default, Michael's choice) keeps the colour art's layers: every shade in the reference becomes tone.
# STYLE=manga draws as a manga inker does, learned from a published page: a mostly paper iris, the
# heaviest stroke on the upper lashes, skin left white, a white gap round the figure, one outline round it.
STYLE = os.environ.get('STYLE', 'cel')
EYES = STYLE

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
    Z['BROWZ'] |= cellmask(i, 0.08, 0.37, 0.22, 0.78)       # a raised or shocked brow sits high
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
# a brow that touches a fringe tip is joined to the hair's mass. Find those by their vertical runs: a brow is a
# short run of hair colour with skin (not hair) just above it; a fringe tip's run comes down from the hair.
runs = np.zeros_like(bulk)
hz = P['hair'] & BROWZ
above_ok = P['skin'] | P['shadow'] | P['line'] | P['brown']
for x in np.nonzero(hz.any(0))[0]:
    col = hz[:, x]
    ys_ = np.nonzero(col)[0]
    starts = ys_[np.r_[True, np.diff(ys_) > 1]]
    ends = ys_[np.r_[np.diff(ys_) > 1, True]]
    for y0_, y1_ in zip(starts, ends):
        if y1_ - y0_ <= 24 and y0_ >= 3 and above_ok[y0_ - 3:y0_, x].any() and not P['hair'][y0_ - 3:y0_, x].all():
            runs[y0_:y1_ + 1, x] = True
for sl, piece in comps(nd.binary_opening(runs & ~brows, iterations=1), 150):
    wm = whole(sl, piece)
    ring = nd.binary_dilation(wm, iterations=6) & ~wm & ~P['hair']
    level = (sl[1].stop - sl[1].start) > 2.5 * (sl[0].stop - sl[0].start)
    below = nd.binary_dilation(wm, iterations=6) & ~wm & (np.arange(ref.PH)[:, None] > (sl[0].start + sl[0].stop) / 2)
    if level and (below & (P['skin'] | P['shadow'] | P['line'] | P['brown'])).sum() > 0.6 * (below & ~wm).sum():
        brows |= wm
# a brow drawn darker than the hair falls in the line class, and its tip may touch a fringe tip. Cut it free of the
# hair mass by a 3 px gap and take the thick, level, mostly-not-black pieces that lie on skin (the lash bands are black)
_d = (P['hair'] | P['line'] | P['black'] | P['strand']) & BROWZ & ~nd.binary_dilation(bulk, iterations=3) & ~brows
_dt = nd.distance_transform_edt(_d)
for sl, piece in comps(_d, 500):
    wm = whole(sl, piece)
    ring = nd.binary_dilation(wm, iterations=5) & ~wm
    level = (sl[1].stop - sl[1].start) > 1.4 * (sl[0].stop - sl[0].start)
    if (level and _dt[wm].max() >= 4 and (wm & P['black']).sum() < 0.5 * wm.sum()
            and (ring & (P['skin'] | P['shadow'])).sum() > 0.65 * ring.sum()):
        # a thin fringe tip touching the brow's end is not brow: open it away, keep the brow's own body
        body = nd.binary_opening(wm, iterations=2)
        brows |= nd.binary_dilation(body, iterations=2, mask=wm) if body.any() else wm
brows = brows | (P['black'] & nd.binary_dilation(brows, iterations=2))
# the hair's deepest shadow (under the locks, by the ears) is near black: outside the face it is still hair
under = nd.binary_dilation(P['hair'] & ~brows, iterations=6, mask=P['black'] & ~INNER
                           & ~nd.binary_dilation(P['skin'] | P['shadow'], iterations=3))   # not the ears' lines
# behind the ears the hair is in deep shadow, near black, and reaches the skin: by darkness, not by colour
under |= nd.binary_dilation(P['hair'] & ~brows, iterations=10, mask=(LUM < 42) & ~INNER & ~LOW) & (LUM < 42) & ~INNER
mass = big((P['hair'] | under) & ~brows, 500)      # small pieces too: the locks hanging by the jaw
hair_m = nd.gaussian_filter(nd.binary_closing(mass, iterations=1).astype(np.float32), 0.9) > 0.5
hair_t = nd.binary_dilation(hair_m, iterations=2)
# the collar: its navy plus the near-black inside of the collar behind the neck, as one shape
jk = P['jacket'] & LOW
jk |= nd.binary_dilation(jk, iterations=30, mask=(P['black'] | P['jacket']) & LOW)
jacket = nd.gaussian_filter(nd.binary_fill_holes(nd.binary_closing(jk, iterations=3)).astype(np.float32), 1.0) > 0.5
DARK = P['black'] | P['brown']

# ---- each brow as one stroke: its thick wedge, then followed column by column along the thin drawn tail that
#      continues it (staying within a few px of where the line is heading), with widths that only narrow
BROW_STROKES = []
BROWALL = np.zeros_like(brows)
_cand = (P['line'] | P['black'] | P['brown'] | P['hair'] & ~bulk) & BROWZ


for i in range(N):
    cw_i = ref.cell(i)[2] - ref.cell(i)[0]
    for sl, piece in comps(nd.binary_closing(brows & cellmask(i), iterations=2), 120):
        if (sl[0].stop - sl[0].start) > (sl[1].stop - sl[1].start):
            continue
        wm = whole(sl, piece)
        got = face.brow_stroke(wm, _cand & cellmask(i), reach=0.22 * cw_i)
        if got:
            BROW_STROKES.append(got[:2])
            BROWALL |= got[2]
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
    if STYLE == 'manga':
        # an inker leaves skin white: only the shadow under the chin and the one the hair casts stay. Not the lids,
        # the nose or the ears (the ears get their lines)
        ears_z = np.zeros_like(shadow)
        for i in range(N):
            ears_z |= cellmask(i, 0.30, 0.72, 0.10, 0.33) | cellmask(i, 0.30, 0.72, 0.67, 0.90)
        cast = shadow & nd.binary_dilation(hair_t, iterations=3) & ~EYEZ & ~ears_z
        cast = nd.binary_propagation(cast, mask=shadow & ~EYEZ & ~ears_z & nd.binary_dilation(hair_t, iterations=16))
        shadow &= LOW | cast
    tone(sh, full(shadow & ~hair_m & ~jacket), level=0.20, lpi=LPI)
    # 2. hair: dark tone, its shine a light tone; the collar darker still, its deepest folds black
    shine = nd.binary_opening(P['shine'] & hair_m, iterations=1)
    tone(sh, full(hair_t & ~shine), level=0.62, lpi=LPI)
    tone(sh, full(shine), level=0.30, lpi=LPI)
    # the collar's folds: a tone graded by the reference's own light (its rim and fold tops are lighter navy)
    jl = nd.gaussian_filter(LUM, 2.0)
    jlv = np.clip(0.84 - (jl - 40) * 0.020, 0.40, 0.84).astype(np.float32)   # folds: stronger contrast
    tone(sh, full(jacket), level=full(np.where(jacket, jlv, 0).astype(np.float32)), lpi=LPI)
    deep = nd.binary_opening(jacket & (jl < 34), iterations=3)
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
            # the pupil can split the iris's colours in two: take every coloured piece within reach of the amber
            cl, _ = nd.label(nd.binary_opening(coloured, iterations=1) | core)
            reach = nd.binary_dilation(core, iterations=5)
            ids = np.unique(cl[reach]); body = np.isin(cl, ids[ids > 0])
            body &= nd.binary_dilation(core, iterations=int(0.06 * (ref.cell(i)[3] - ref.cell(i)[1])))
            hl = hull(body)
            # half-closed: short and wide under a low lid. A small round iris (the worried face's) is a whole one
            half = np.ptp(np.nonzero(hl)[0]) < 0.065 * (ref.cell(i)[3] - ref.cell(i)[1]) \
                and np.ptp(np.nonzero(hl)[1]) > 1.25 * np.ptp(np.nonzero(hl)[0])
            lights = np.zeros_like(hl)
            for wsl, wpiece in comps(P['white'] & nd.binary_dilation(hl, iterations=2)):
                wm = whole(wsl, wpiece)
                grown = nd.binary_dilation(wm, iterations=2) & P['white']
                if wpiece.sum() < 500 and grown.sum() < wpiece.sum() + 30 and (wm & hl).sum() > 0.5 * wpiece.sum():
                    lights |= wm
            # the pale yellow low in the iris passes for skin: only skin at the hull's edge is cut away
            im = nd.binary_opening(hl & ~(P['white'] & ~lights) & ~(P['skin'] & ~nd.binary_erosion(hl, iterations=4)),
                                   iterations=2)
            # the reference's own dark rim bounds the iris: fill out to it (the pale crescent low in the iris passes
            # for white and skin, so the hull stops short of the rim) and cut away anything beyond it
            enclosed = None if half else face.iris_to_rim(im, DARK)
            if enclosed is not None:
                im = enclosed
            if core.sum() < 40 or (nd.binary_dilation(im, iterations=6) & P['white']).sum() < 30:
                continue                                 # no amber, no white beside it: hair or a brow, not an eye
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
            if EYES == 'manga':
                face.manga_iris(sh, full(im), pupil=full(pup), lights=full(lights), seed=60 + i,
                                solid=0.40 if half else 0.26, fade=0.85 if half else 0.66, sparkles=0 if half else 3)
            else:
                face.iris(sh, full(im), full(lv), pupil=full(pup), lights=full(lights), seed=60 + i,
                          rim=0.3 if half else 0.6, streaks=3 if half else 6)
            LIGHTS |= lights
        # (the reference draws its own lower lids: they come with the thin lines)
        # the lid's shadow falls from the lash band only: the iris's own dark rim casts none, or tone shows below it
        face.eye_white(sh, full(P['white'] & ez & ~IRIS & ~LIGHTS),
                       full(DARK & ez & ~nd.binary_dilation(IRIS & ez, iterations=7)), depth=4, level=0.2)
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
    # a lash band lies on an eye white; a big dark piece away from any white is hair shadow over the ear
    _near_w = np.unique(ll[nd.binary_dilation(P['white'] & EYEZ, iterations=8) & lash])
    lash_big &= np.isin(ll, _near_w[_near_w > 0])
    _ears = np.zeros_like(lash)
    for i in range(N):
        _ears |= cellmask(i, 0.30, 0.72, 0.10, 0.33) | cellmask(i, 0.30, 0.72, 0.67, 0.90)
    lash_big &= ~_ears | nd.binary_dilation(P['white'] & EYEZ, iterations=12)   # out by the ears: hair shadow
    blk = (P['black'] | P['brown'] & ~IRIS & EYEZ | lash) & ~nd.binary_dilation(brows, iterations=1)
    blk &= ~(MOUTHZ & ~open_m) & ~hair_edge & ~SMALL & ~open_m & ~jacket & ~nd.binary_erosion(IRIS, iterations=2)   # face.iris owns it
    blk &= ~BROWALL
    tips = blk & nd.binary_dilation(hair_t, iterations=5) & ~lash_big
    # round the ears (outside the face) every black is a drawn line: hairlines, never filled blobs
    EARS = np.zeros_like(blk)                       # the ears reach in past INNER's edge
    for i in range(N):
        EARS |= cellmask(i, 0.30, 0.72, 0.10, 0.33) | cellmask(i, 0.30, 0.72, 0.67, 0.90)
    # (the eye zone reaches the ears too: there only the blacks next to an eye white stay solid)
    tips |= blk & (~INNER | EARS) & ~LOW & ~lash_big & ~(EYEZ & ~EARS) \
        & ~nd.binary_dilation(P['white'] & EYEZ, iterations=12)
    # the dark rim round an eye white, away from the lash band, is a lid line: a hairline, not a heavy ink
    whites_b = big(P['white'] & EYEZ & ~LIGHTS, 200)
    lidb = blk & nd.binary_dilation(whites_b, iterations=4) & EYEZ & ~lash_big & ~nd.binary_dilation(IRIS, iterations=2) \
        & ~nd.binary_erosion(nd.binary_fill_holes(nd.binary_dilation(whites_b, iterations=2)), iterations=3)   # a pupil inside the white stays solid
    blk &= ~tips & ~lidb
    # the figure's silhouette against the background: in manga style one smooth outline (drawn in step 8), so the
    # traced pieces along it go
    figure = nd.binary_fill_holes(nd.binary_opening(~P['bg'], iterations=2))
    figure = nd.gaussian_filter(figure.astype(np.float32), 1.0) > 0.5
    sil_band = nd.binary_dilation(figure & ~nd.binary_erosion(figure, iterations=1), iterations=3)
    if STYLE == 'manga':
        blk &= ~sil_band; tips &= ~sil_band
    if EYES == 'manga':                              # the upper lash band is the heaviest mark on the face
        # from the lash band itself: a brow's tail line running into it must not cut notches out of it
        _lb = nd.binary_closing(nd.binary_dilation(lash_big & ~nd.binary_dilation(brows, iterations=1), iterations=1),
                                iterations=2)
        _lb = nd.binary_opening(_lb, structure=np.ones((1, 11), bool)) | (_lb & ~nd.binary_dilation(
            nd.binary_opening(_lb, structure=np.ones((1, 11), bool)), iterations=4))   # drop lumps on the band's edge
        _lb = nd.gaussian_filter(_lb.astype(np.float32), 1.6) > 0.5          # one smooth band, not a lumpy one
        blk = (blk & ~lash_big) | (_lb & EYEZ & ~nd.binary_erosion(IRIS, iterations=1))
    n1, _ = trace.ink_line_art(sh, full(blk), pens=('maru', 'g_pen'), split_mm=0.22, min_len_px=6, beta_px=3.0,
                               weight=1.0, min_px=1.0, even=0.6, seed=7)
    nlb, _ = trace.ink_line_art(sh, full(lidb), pens=('maru', 'maru'), min_len_px=6, beta_px=99, weight=0.8,
                                min_px=1.2, even=0.85, seed=19)
    nt, _ = trace.ink_line_art(sh, full(tips), pens=('maru', 'maru'), min_len_px=8, beta_px=99, weight=0.45,
                               min_px=1.0, even=0.95, seed=14)
    # 6b. brows over the lashes: one tapered stroke each, from thick to thin, with a paper ring round it
    ring_clip = ~hair_t & ~nd.binary_dilation(lash_big | blk & EYEZ, iterations=3) if STYLE == 'manga' else ~hair_t
    with sh.clip(full(ring_clip)):                   # paper ring: the same stroke, 4 px wider, in white
        for pts, w in BROW_STROKES:
            face.brow_ring(sh, pts + M, w)
    for pts, w in BROW_STROKES:
        face.brow(sh, [(x + M, y + M) for x, y in pts], w, smooth=2.0)
    # 7. thin lines: creases, nose, mouth lines, ears, jaw, sweat drops, the zip
    free = ~nd.binary_dilation(DARK, iterations=2) & ~hair_m & ~brows & ~nd.binary_dilation(IRIS, iterations=2) \
        & ~hair_edge & ~jacket & ~P['bg']
    thin = P['line'] & free & ~open_m & ~(BAND & ~P['black'])      # the band's edge is not a line
    thin &= ~BROWALL
    if STYLE == 'manga':
        thin &= ~sil_band
        for i in range(N):                           # the nose is one stroke (drawn below), not its shadow's edges
            thin &= ~cellmask(i, 0.44, 0.68, 0.40, 0.60)
    # the eye white's own grey rim: only its lower half is a lid, drawn as a light hairline; the rest of the rim
    # (round the corners, past the lash band) is left out, or every eye gets a ring
    whites = big(P['white'] & EYEZ & ~LIGHTS, 200)
    rim = thin & nd.binary_dilation(whites, iterations=4) & EYEZ
    lower = np.zeros_like(rim)
    for sl, piece in comps(whites, 200):
        ys_w = np.nonzero(piece)[0] + sl[0].start
        xs_w = np.nonzero(piece)[1] + sl[1].start
        yc, x0w, x1w = ys_w.mean(), xs_w.min(), xs_w.max()
        span = np.zeros_like(rim)
        span[int(yc):, int(x0w + 0.08 * (x1w - x0w)):int(x1w - 0.08 * (x1w - x0w))] = True
        lower |= rim & span & nd.binary_dilation(whole(sl, piece), iterations=4)
    thin &= ~rim
    n2, _ = trace.ink_line_art(sh, full(thin & ~MOUTHZ), pens=('maru', 'saji'), split_mm=0.2, min_len_px=5,
                               beta_px=99, weight=0.8, min_px=1.2, even=0.8, seed=8)
    mouth_l = nd.binary_closing((P['line_soft'] | DARK) & MOUTHZ & ~open_m & ~P['bg'], iterations=2)
    nl, _ = trace.ink_line_art(sh, full(face.lid_dashes(lower) if EYES == 'manga' else lower), pens=('maru', 'maru'), min_len_px=8, beta_px=99, weight=0.7,
                               min_px=1.1, even=0.9, seed=18)
    n2 += nl
    n2m, _ = trace.ink_line_art(sh, full(mouth_l), pens=('maru', 'g_pen'), split_mm=0.12, min_len_px=8, beta_px=99,
                                weight=0.85, min_px=1.7, even=0.7, seed=12)
    zipl = nd.binary_dilation(jacket, iterations=3) & (P['zip'] | P['line_soft'] & ~P['jacket']) & ~P['bg'] & ~hair_t
    with sh.clip(full(nd.binary_dilation(jacket, iterations=3))):
        nz, _ = trace.ink_line_art(sh, full(zipl & ~P['skin']), pens=('maru', 'maru'), min_len_px=10, beta_px=99,
                                   weight=0.5, min_px=1.0, even=0.9, seed=17)
    if STYLE == 'manga':
        # the nose: one short stroke at its tip, where the colour art has a shadow patch
        for i in range(N):
            nose_m = nd.binary_opening(P['shadow'] & cellmask(i, 0.44, 0.66, 0.40, 0.60), iterations=1)
            pcs = comps(nose_m, 40)
            if not pcs:
                continue
            sl, piece = max(pcs, key=lambda c: c[1].sum())
            ys_n, xs_n = np.nonzero(piece); ys_n = ys_n + sl[0].start; xs_n = xs_n + sl[1].start
            yb = np.percentile(ys_n, 92)
            xl, xr = np.percentile(xs_n, 25), np.percentile(xs_n, 90)
            sh.stroke([(xl + M, yb + M + 1.5), (xr + M, yb + M - 1.5)],
                      pen='maru', pressure=0.7, taper=(0.5, 0.6), seed=90 + i)
    # 8. hair strand lines and the outlines of hair and collar
    strands = nd.binary_opening(P['strand'] & nd.binary_erosion(hair_m, iterations=3) & ~shine, iterations=1)
    n3, _ = trace.ink_line_art(sh, full(strands), pens=('maru', 'maru'), min_len_px=24, beta_px=99, weight=0.6,
                               min_px=1.0, even=0.8, seed=9)
    if STYLE == 'manga':
        # one continuous outline round the whole figure, the heaviest line on it; the hair's own outline only
        # where the hair meets the face or the collar, not the background
        with sh.clip(full(~INNER & ~nd.binary_dilation(P['bg'], iterations=5))):
            n4 = mprep.ink_outline(sh, full(hair_t), pen='g_pen', pressure=(0.35, 0.6), sigma=1.5, min_len=30, seed=10)
        div = np.zeros_like(figure)
        for x in ref.COLS:
            px = int(round((x - ref.X0) * ref.S)); div[:, max(px - 12, 0):px + 13] = True
        for y in ref.ROWS:
            py = int(round((y - ref.Y0) * ref.S)); div[max(py - 12, 0):py + 13, :] = True
        with sh.clip(full(~div)):
            ne = mprep.ink_outline(sh, full(figure), pen='g_pen', pressure=(0.62, 0.9), sigma=2.0, min_len=20, seed=21)
    else:
        with sh.clip(full(~INNER)):
            n4 = mprep.ink_outline(sh, full(hair_t), pen='g_pen', pressure=(0.35, 0.6), sigma=1.5, min_len=30, seed=10)
        # the ears' outer edge: the figure's silhouette against the background, where it is not hair
        with sh.clip(full(EARS & ~nd.binary_dilation(hair_t, iterations=1))):
            ne = mprep.ink_outline(sh, full(figure), pen='g_pen', pressure=(0.5, 0.7), sigma=1.2, min_len=12, seed=21)
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
print('strokes', sh.stats['strokes'], 'traced', ne + n1 + nt + n2 + n2m + nz + n3 + n4 + n5, round(time.time() - t0), 's')
