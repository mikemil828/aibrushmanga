"""A single manga panel drawn from the rice-field reference (ref.png, 1152 x 1728).

Decided before drawing:
  * One tall panel on a B4-size sheet, 8 px/mm, 10 mm margins: the reference maps in at x1.597.
  * Primary: the walker (small, but the eye goes to him) and the path that leads to him. Secondary: the paddies
    and banks in front. Background: the village, trees, mountains and sky, drawn lighter and sparser with distance.
  * One dot screen (60 lpi, 45 deg) for everything solid: sky, mountains, banks, path. Water gets its own
    horizontal line tone, so the paddies read as still water and never as ground.
  * Sky: a dot gradient, darkest at the top, opening to paper around the sun; clouds one tone step darker with
    their sunlit rims scraped white. The sun is bare paper.
  * Mountains: flat tones by distance (far ranges light, the near right mountain darker), mist as tone fading to
    paper, maru ridge lines, a few hatching lines on the right mountain's darkest forest.
  * Trees, pine and house walls: solid black silhouettes against the mist; lit windows left white; thatch in a
    mid tone with short strokes down the slope.
  * Paddies: line tone by the reflected light; the brightest glints scraped white. Seedlings inked one tuft at a
    time from the reference, G-pen near, maru far.
  * Banks: dot tone, darker on the shaded side; grass blades inked in the foreground; dew scraped white.
  * The walker: hand-drawn: straw hat in light tone with weave lines, black haori with white fold lines and the
    crest, grey hakama with pleat lines, black swords. His long shadow falls across the path into the water.
"""
import os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from manga import Sheet, Frame, tone, hatch, effects, trace, prep, geom, nature, water

W, H, PPM = 2000, 2920, 8.0
OX, OY = 80, 80
RW, RH = 1152, 1728
S = 1840 / RW
LPI = 60

ref = np.asarray(Image.open('ref.png'), np.float32) / 255
L = ref.mean(2)
Ls = nd.gaussian_filter(L, 1.0)
Z = dict(np.load('zones.npz'))
YY, XX = np.mgrid[:RH, :RW]


FR = Frame((W, H), (RW, RH), (OX, OY), S, lpi=LPI)
up, upm, P, lay = FR.up, FR.upm, FR.P, FR.lay


# ------------------------------------------------------------------ sky
def sky(sh):
    zone = Z['sky'] | Z['cloud'] | Z['sun']
    grad = np.clip(0.30 - 0.26 * YY / 560, 0.04, 0.30)
    dsun = np.hypot(XX - 336, YY - 486)
    grad = grad * np.clip((dsun - 55) / 330, 0, 1) ** 0.7
    lv = grad.copy()
    # clouds one step darker; their sun-lit rims (warm) back to paper
    cl = nd.gaussian_filter(nd.binary_opening(Z['cloud'], iterations=2).astype(np.float32), 3.0) > 0.5
    lv = np.where(cl, np.clip(grad + 0.16, 0.14, 0.42), lv)
    warm = nd.gaussian_filter(ref[..., 0] - ref[..., 2], 1.5)
    rim = cl & (warm > 0.02) & (nd.gaussian_filter(L, 1.5) > 0.8)
    lv = np.where(rim, 0.0, lv)
    # chimney smoke: a soft pale tone
    smoke = zone & (XX < 140) & (YY > 430) & (Ls < 0.79)
    lv = np.where(smoke, np.maximum(lv, np.clip(1.6 * (0.84 - Ls), 0, 0.22)), lv)
    lv[Z['sun']] = 0
    lv = nd.gaussian_filter(lv, 0.8)
    lay(sh, zone, lv)
    # a thin broken line along the clouds' lower edges, where the low sun lights them
    prep.ink_contour(sh, nd.gaussian_filter(cl.astype(np.float32), 1.0), min_len=25, tol=0.5, simplify_first=True,
                     keep=lambda Q: geom.normals(Q)[:, 1] > 0.35, min_run=4, piece=10 ** 6, pen='maru',
                     pressure=0.3, to_page=P, seed=200)


# ------------------------------------------------------------------ mountains and mist
def mountains(sh):
    far, right = Z['mtn_far'], Z['mtn_right']
    lf = np.clip(0.30 + 0.9 * (0.47 - Ls), 0.05, 0.40)
    lr = np.clip(0.52 + 1.0 * (0.28 - Ls), 0.06, 0.66)
    lv = np.where(right, lr, lf)
    lay(sh, far | right, lv)
    # the right mountain's darkest forest: a few hatched lines down the slope
    dark = np.where(right & (Ls < 0.24), 1.0, 0.0)
    hatch(sh, up(nd.gaussian_filter(dark, 1.5)), upm(right & (Ls < 0.26)), pen='maru', angle=-58,
          spacing_mm=0.9, layers=((0.45, 0.0),), pressure=(0.2, 0.5), stroke_mm=(3, 7), seed=21)
    # lit ridges (green, sunlit) scraped back
    lit = right & (ref[..., 1] > ref[..., 2] + 0.01) & (Ls > 0.30)
    lit = nd.binary_opening(lit, iterations=1)
    sh.fill(upm(lit), mode='white', amount=0.45)
    # ridge lines: where sky meets a mountain
    skyf = nd.gaussian_filter((Z['sky'] | Z['cloud'] | Z['sun']).astype(np.float32), 1.0)
    near_m = nd.binary_dilation(far | right, iterations=3)
    near_r = nd.binary_dilation(right, iterations=3)
    inside = lambda m, Q: m[np.minimum(Q[:, 1].astype(int), RH - 1), np.minimum(Q[:, 0].astype(int), RW - 1)]
    prep.ink_contour(sh, skyf, min_len=12, keep=lambda Q: inside(near_m, Q) & (Q[:, 1] > 270), min_run=15, tol=0.6,
                     piece=40, style=lambda Q: ('saji', 0.35) if inside(near_r, Q[len(Q) // 2:len(Q) // 2 + 1])[0]
                     else ('maru', 0.45), to_page=P, seed=300)


# ------------------------------------------------------------------ village: trees, pine, houses
def village(sh):
    t = Z['trees']
    right = XX > 700
    # left of the path the trees frame the near village: black against the mist, as before
    tl = t & ~right
    blk = nd.binary_opening(tl & (Ls < 0.22), iterations=1)
    lay(sh, tl & ~blk, np.full(L.shape, 0.62))
    sh.fill(upm(blk))
    # right: layered by distance. The two front cedars stay black; the trees behind them and the broadleaf
    # tree step back in tone, so the mass reads as a grove in the mist rather than one black block
    tr = t & right
    Lt = nd.gaussian_filter(L, 2.0)
    lv = np.clip(0.6 + 2.0 * (0.22 - Lt), 0.55, 0.82)
    lay(sh, tr, lv)
    Lf = nd.gaussian_filter(L, 1.0)
    front = nd.binary_closing(tr & (Lf < 0.14), iterations=1) & tr
    lab, n = nd.label(front)
    if n:
        sizes = nd.sum(front, lab, np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:] = sizes > 60
        front = keep[lab]
    sh.fill(upm(front))
    # a thin line where a darker layer stands in front of a lighter one, so the layers separate
    prep.ink_contour(sh, nd.gaussian_filter(tr.astype(np.float32), 1.0), min_len=12, keep=lambda Q: Q[:, 1] < 676,
                     min_run=4, tol=0.5, piece=20, pressure=0.35, to_page=P, seed=640)
    pine(sh)
    lay(sh, Z['houses'], np.full(L.shape, 0.5))          # whatever of the old house shapes the drawn ones leave
    for k, hs in enumerate(HOUSES):
        house(sh, hs, k)


TRUNK = [(197, 686), (203, 640), (210, 590), (216, 548), (223, 505), (227, 482)]
BRANCHES = [[(206, 618), (186, 604), (172, 596)], [(209, 600), (232, 590), (248, 585)],
            [(214, 556), (196, 545), (180, 540)], [(217, 540), (238, 530), (252, 526)],
            [(222, 505), (204, 494), (188, 490)]]


def pine(sh):
    """The pine against the sunset: pads in black with needle tips scraped along their sunlit tops, a trunk and
    branches with a lit edge on the side toward the sun."""
    m = nd.binary_fill_holes(nd.binary_closing(Z['pine'], iterations=2))
    m = nd.gaussian_filter(m.astype(np.float32), 1.0) > 0.5
    pm_ = upm(m)
    sh.fill(pm_)
    for k, br in enumerate(BRANCHES):
        sh.stroke([P(*q) for q in br], pen='g_pen', pressure=0.55, seed=600 + k)
    sh.stroke([P(*q) for q in TRUNK], pen='brush', pressure=0.55, seed=620)
    # the trunk's lit edge (the sun is low on the right of it) and a few bark marks
    pads = upm(nd.binary_erosion(m, iterations=2) & (YY < 650))
    with sh.clip(~pads):
        lit = [(x + 1.6, y) for x, y in TRUNK[:3]]
        sh.stroke([P(*q) for q in lit], pen='white', mode='white', pressure=0.1, seed=621)
    rng = np.random.default_rng(62)
    for k in range(0):
        t = rng.uniform(0.03, 0.25)
        y = 686 - t * 220
        x = np.interp(-y, [-q[1] for q in TRUNK], [q[0] for q in TRUNK])
        sh.stroke([P(x - 1.0, y), P(x + 1.2, y - 1.5)], pen='white', mode='white', pressure=0.08, seed=630 + k)
    # needle tips: short white strokes fanning down from each pad's upper edge
    f = nd.gaussian_filter(m.astype(np.float32), 1.0)
    k = 0
    for Pl in geom.outlines(f, 0.5, 15):
        Q = geom.resample(Pl, 1.6)[0]
        nrm = geom.normals(Q)
        for q, nv in zip(Q, nrm):
            if nv[1] > -0.45 or abs(q[0] - np.interp(-q[1], [-t[1] for t in TRUNK], [t[0] for t in TRUNK])) < 3:
                continue
            x, y = q
            if rng.random() < 0.3:
                continue
            for j in range(1):
                a = rng.normal(0, 0.4)
                ln = rng.uniform(2.2, 3.8)
                sh.stroke([P(x, y + 0.6), P(x + np.sin(a) * ln, y + 0.6 + np.cos(a) * ln)], pen='white',
                          mode='white', pressure=0.16, taper=(0.1, 0.6), seed=700 + k)
                k += 1


# Thatched farmhouses read off the reference (ref px): ridge (x0, x1, y), eaves (x0, x1, y), wall bottom,
# windows [(x0, x1)] lit, chimney (x0, x1, top) or None, crossed sticks on the ridge, sunlit slope ('l'/'r'/None),
# and how far back it sits (0 near .. 1 in the mist).
HOUSES = [
    dict(ridge=(78, 147, 585), eave=(28, 196, 653), wall=683, win=[(61, 73), (84, 97), (99, 114), (155, 170)],
         chim=(112, 123, 566), sticks=True, lit='r', mist=0.0),
    dict(ridge=(-10, 4, 612), eave=(-10, 26, 650), wall=684, win=[], chim=None, sticks=False, lit='r', mist=0.0),
    dict(ridge=(262, 318, 617), eave=(247, 334, 650), wall=666, win=[], chim=None, sticks=False, lit=None, mist=0.7),
    dict(ridge=(891, 913, 659), eave=(884, 920, 668), wall=674, win=[], chim=None, sticks=False, lit=None, mist=0.3),
    dict(ridge=(946, 990, 644), eave=(928, 1002, 662), wall=676, win=[(943, 949), (976, 983)], chim=None,
         sticks=False, lit=None, mist=0.2),
    dict(ridge=(1026, 1078, 637), eave=(1006, 1110, 660), wall=678, win=[(1030, 1040), (1072, 1083)], chim=None,
         sticks=True, lit=None, mist=0.1),
]


def house(sh, d, k):
    (rx0, rx1, ry), (ex0, ex1, ey) = d['ridge'], d['eave']
    wb = d['wall']
    inset = (ex1 - ex0) * 0.06
    roof = [(rx0, ry), (rx1, ry), (ex1, ey), (ex0, ey)]
    wall = [(ex0 + inset, ey - 1), (ex1 - inset, ey - 1), (ex1 - inset, wb), (ex0 + inset, wb)]
    pr = [P(*q) for q in roof]; pw = [P(*q) for q in wall]
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pr, fill=1); rm = np.asarray(im, bool)
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pw, fill=1); wm = np.asarray(im, bool)
    both = rm | wm
    if d['chim']:
        cx0, cx1, ct = d['chim']
        im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).rectangle([P(cx0, ct), P(cx1, ry + 2)], fill=1)
        cm = np.asarray(im, bool); both |= cm
    mist = d['mist']
    sh.fill(both, mode='white')                       # clear what lies under it, no halo around it
    # walls: a dark tone, not solid black, so the houses sit back in the evening haze
    tone(sh, wm, level=0.72 - 0.25 * mist, lpi=LPI)
    for (wx0, wx1) in d['win']:
        a, b = P(wx0, ey + 1.5), P(wx1, min(ey + 9, wb - 3))
        im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).rectangle([a, b], fill=1)
        wmask = np.asarray(im, bool)
        sh.fill(wmask, mode='white')
        tone(sh, wmask, level=0.15, lpi=LPI)                             # lamplight, softened
    # thatch: a mid tone, darker along the eaves, straws from ridge to eave
    tone(sh, rm, level=0.36 - 0.12 * mist, lpi=LPI)
    eb = [(ex0 + (rx0 - ex0) * 0.15, ey - (ey - ry) * 0.15), (ex1 - (ex1 - rx1) * 0.15, ey - (ey - ry) * 0.15),
          (ex1, ey), (ex0, ey)]
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon([P(*q) for q in eb], fill=1)
    tone(sh, np.asarray(im, bool), level=0.48 - 0.15 * mist, lpi=LPI)
    rng = np.random.default_rng(300 + k)
    n = int((ex1 - ex0) / (4.5 + 4 * mist))
    for t in np.linspace(0.02, 0.98, n):
        top = (rx0 + (rx1 - rx0) * t + rng.normal(0, 0.4), ry + rng.uniform(0, (ey - ry) * 0.25))
        bot = (ex0 + (ex1 - ex0) * t, ey - rng.uniform(0, (ey - ry) * 0.2))
        sh.stroke([P(*top), P(*bot)], pen='maru', pressure=0.18 - 0.08 * mist, taper=(0.5, 0.3), seed=4000 + k * 100)
    if d['lit']:                                                        # the slope catching the low sun
        a, b = ((rx1, ry), (ex1, ey)) if d['lit'] == 'r' else ((rx0, ry), (ex0, ey))
        sh.stroke([P(a[0] - 1.5, a[1] + 1), P(b[0] - 2, b[1] - 1)], pen='white', mode='white', pressure=0.2)
    if d['chim']:
        sh.fill(cm)
    if d['sticks']:                                                     # crossed sticks holding the ridge
        for t in np.linspace(0.15, 0.85, 4):
            x = rx0 + (rx1 - rx0) * t
            sh.stroke([P(x - 3, ry - 4), P(x + 2, ry + 2)], pen='maru', pressure=0.4)
            sh.stroke([P(x + 3, ry - 4), P(x - 2, ry + 2)], pen='maru', pressure=0.4)
    sh.stroke([P(*q) for q in roof + roof[:1]], pen='maru', pressure=0.45 - 0.2 * mist, taper=(0, 0), smooth=False)


# ------------------------------------------------------------------ paddies, banks, path
GREEN = (ref[..., 1] > ref[..., 0] + 0.03) & (ref[..., 1] > ref[..., 2] - 0.02)


def seedlings_mask():
    """Green tufts standing in the water: small green blobs below the far fields."""
    g = GREEN & (YY > 900) & ~Z['path'] & ~Z['figure']
    lab, n = nd.label(g)
    area = nd.sum(g, lab, np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = area < 700
    m = keep[lab]
    near_water = nd.binary_dilation(Z['water'] | Z['shadow'], iterations=4)
    lab2, n2 = nd.label(m)
    touch = nd.maximum(near_water, lab2, np.arange(1, n2 + 1))
    k2 = np.zeros(n2 + 1, bool); k2[1:] = touch > 0
    sdm = k2[lab2]
    # the far seedlings show only as dark specks in the bright water
    wat = Z['water'].astype(np.float32)
    Lw = nd.gaussian_filter(L * wat, 5) / np.maximum(nd.gaussian_filter(wat, 5), 1e-3)
    spk = Z['water'] & (L < Lw - 0.10) & (YY > 860) & (YY < 1130)
    lab3, n3 = nd.label(spk)
    a3 = nd.sum(spk, lab3, np.arange(1, n3 + 1))
    k3 = np.zeros(n3 + 1, bool); k3[1:] = (a3 >= 2) & (a3 < 40)
    return sdm | k3[lab3]


REFL_ZONE = (XX < 560) & (YY > 745) & (YY < 905)


def reflections(sh):
    """The pine and the farmhouse mirrored in the paddy in front of them: the same shapes upside down,
    foreshortened, in a darker line tone, cut through by thin horizontal ripples."""
    src = nd.binary_fill_holes(nd.binary_closing(Z['pine'], iterations=2))
    trunk = np.zeros_like(src)
    for x, y in TRUNK:
        trunk[max(0, int(y) - 12):int(y) + 12, int(x) - 3:int(x) + 4] = True
    hs = HOUSES[0]
    (rx0, rx1, ry), (ex0, ex1, ey) = hs['ridge'], hs['eave']
    im = Image.new('L', (RW, RH), 0)
    ImageDraw.Draw(im).polygon([(rx0, ry), (rx1, ry), (ex1, ey), (ex0, ey)], fill=1)
    ImageDraw.Draw(im).rectangle([ex0 + 10, ey, ex1 - 10, hs['wall']], fill=1)
    src = src | trunk | np.asarray(im, bool)
    out = water.mirror(src, src_line=686, dst_line=760, squash=0.48, rows=(745, 905))
    out &= Z['water'] & REFL_ZONE
    out = nd.gaussian_filter(out.astype(np.float32), (0.6, 1.2)) > 0.5
    lay(sh, out, 0.62, kind='line', angle=water.HORIZONTAL)
    # ripples: broken white lines across the reflection, closer together toward the viewer
    water.ripples(sh, out, rows=(762, 900), to_page=P, seed=9000)


def paddies(sh, sd):
    w = Z['water'] | Z['shadow'] | sd
    # the water under each tuft takes the water around it (normalized blur over water pixels only)
    Lw, sdd = water.fill_under(L, Z['water'] | Z['shadow'], sd)
    # calm water: smooth the reflected light hard (the seedling rows must not print into it)
    lv = water.calm_level(Lw, floor=0)
    # far paddies: smooth hard along the rows, so the strips read as calm bands, not patches
    lvf = np.clip(1.0 * (0.78 - nd.gaussian_filter(Lw, (1.5, 16.0))), 0, 0.5)
    wf = np.clip((830 - YY) / 60, 0, 1)
    lv = lvf * wf + lv * (1 - wf)
    # the left village's reflections are drawn separately (reflections()); here the water is calm under them
    rz = REFL_ZONE & w
    calm = np.clip(0.06 + 0.10 * (YY - 745) / 160, 0.06, 0.16)     # bright evening water, a little darker near
    lv = np.where(rz, calm, lv)
    lv[lv < 0.04] = 0
    core = nd.gaussian_filter(Z['shadow'].astype(np.float32), 3.0)
    lv = np.maximum(lv, np.clip(core * 1.5, 0, 1) * 0.74)
    lay(sh, w, lv, kind='line', angle=water.HORIZONTAL)
    glint = (nd.gaussian_filter(Lw, (2.0, 6.0)) > 0.88) & w & ~sdd
    glint = nd.binary_opening(glint, iterations=3) & ~REFL_ZONE
    sh.fill(upm(glint), mode='white')
    # bright water: a few hand ripple lines so it still reads as water
    rng = np.random.default_rng(12)
    bright = w & (lv == 0) & ~sd & (YY > 700)
    bright = nd.binary_erosion(bright, iterations=3)
    ys, xs = np.nonzero(bright)
    for i in rng.choice(len(xs), size=min(len(xs) // 420, 300), replace=False):
        x, y = xs[i], ys[i]
        ln = (6 + 14 * (y - 700) / 1000) * rng.uniform(0.6, 1.4)
        if bright[y, min(int(x + ln), RW - 1)]:
            sh.stroke([P(x, y), P(x + ln, y + rng.normal(0, 0.15))], pen='maru', pressure=0.3, taper=(0.4, 0.6),
                      seed=7000 + int(i))
    mud = (w | Z['levee']) & (nd.gaussian_filter(L, 1) < 0.15) & (YY > 700) & ~Z['shadow']
    mud = nd.gaussian_filter(nd.binary_opening(mud, iterations=2).astype(np.float32), 2.0) > 0.5
    labm, nm = nd.label(mud)
    am = nd.sum(mud, labm, np.arange(1, nm + 1))
    km = np.zeros(nm + 1, bool); km[1:] = am > 1500
    mud = km[labm]
    mud = nd.binary_fill_holes(nd.gaussian_filter(mud.astype(np.float32), 3.0) > 0.5)   # one band, not blots
    mp = upm(mud)
    # black only in the deep channel along the right bank; elsewhere the soil is a dark tone
    labc, nc = nd.label(mud)
    cx = nd.center_of_mass(mud, labc, np.arange(1, nc + 1))
    deep = np.zeros(nc + 1, bool)
    deep[1:] = [c[1] > 540 for c in cx]
    core = upm(nd.binary_erosion(deep[labc], iterations=4))
    tone(sh, upm(mud & ~deep[labc]), level=0.62, lpi=LPI)
    mp = upm(deep[labc])
    tone(sh, mp & ~core, level=0.7, lpi=LPI)
    sh.fill(core)
    # the wet edge where mud meets water catches the light: a thin white rim, broken
    rim = mp & ~nd.binary_erosion(mp, iterations=2) & nd.binary_dilation(upm(w & ~mud), iterations=3)
    rng2 = np.random.default_rng(14)
    lab_r, nr = nd.label(rim)
    sh.fill(rim & (rng2.random(nr + 1) > 0.25)[lab_r], mode='white')



def banks(sh, sd):
    b = Z['levee'] & ~sd
    Lb = nd.gaussian_filter(L, 4.0)
    lv = np.where(Lb > 0.40, 0.26, 0.5)
    lv = np.where(Lb < 0.2, 0.66, lv)
    lay(sh, b, lv)
    n = grass(sh, b)
    dew = b & (L > 0.72) & (YY > 1000)
    sh.fill(upm(dew), mode='white')
    return n


def grass(sh, b):
    """Grass on the banks, drawn: blades along the edges (where a bank meets water or path) and tufts inside,
    bigger and denser toward the front."""
    rng = np.random.default_rng(33)
    depth = lambda y: (y - 860) / 860
    k = nature.edge_grass(sh, b, depth, P, rng)
    k += nature.tufts(sh, nd.binary_erosion(b, iterations=3) & (YY > 1000), depth, P, rng, seed=20000 + k)
    return k


def seedlings(sh, sd):
    """Each tuft drawn as a few leaves fanning up from its base, with a short broken reflection below."""
    return nature.seedlings(sh, sd, lambda y: (y - 900) / 800, P, seed=41)


def path(sh):
    p = Z['path']
    Lp = nd.gaussian_filter(L, 4.0)
    lv = np.where(Lp < 0.36, 0.38, 0.14)
    lv = np.where(Lp > 0.6, 0.0, lv)
    lv = np.where(Z['shadow'], 0.62, lv)
    lay(sh, p, lv)


# ------------------------------------------------------------------ the walker (hand-drawn from the reference)
HAT = [(688, 963), (700, 966), (717, 977), (724, 990), (723, 997), (706, 1000), (690, 1001), (672, 999), (654, 996),
       (651, 990), (656, 980), (676, 967)]
HAT_RIM = [(651, 989), (724, 989), (723, 997), (706, 1000), (690, 1001), (672, 999), (654, 996)]
HAT_SHADE = [(690, 965), (700, 966), (717, 977), (724, 989), (700, 990), (692, 989)]
HAORI = [(677, 1001), (700, 1001), (718, 1011), (729, 1028), (740, 1058), (744, 1082), (736, 1094), (716, 1098),
         (712, 1084), (706, 1068), (670, 1068), (664, 1084), (660, 1095), (646, 1094), (633, 1086), (636, 1056),
         (646, 1026), (656, 1013)]
HAKAMA = [(669, 1066), (706, 1066), (712, 1084), (716, 1100), (721, 1150), (722, 1192), (716, 1201), (700, 1206),
          (676, 1204), (657, 1199), (656, 1180), (659, 1120), (662, 1095), (664, 1084)]
KOSHIITA = [(671, 1066), (704, 1066), (703, 1076), (672, 1076)]
HAKAMA_SHADE = [(704, 1086), (712, 1086), (716, 1100), (721, 1150), (722, 1192), (716, 1201), (705, 1204), (703, 1150)]
LEG_SHADE = [(694, 1140), (712, 1150), (705, 1162), (690, 1176), (687, 1200), (680, 1202), (684, 1172)]
HANDS = [[(645, 1094), (656, 1095), (656, 1108), (653, 1115), (647, 1114), (644, 1105)],
         [(719, 1098), (733, 1095), (737, 1104), (735, 1115), (728, 1119), (722, 1113)]]
TABI = [[(671, 1204), (684, 1204), (685, 1218), (680, 1224), (673, 1222), (670, 1213)],
        [(686, 1200), (697, 1202), (700, 1214), (697, 1226), (690, 1227), (686, 1214)]]
SOLES = [[(672, 1219), (684, 1218), (682, 1226), (674, 1227)], [(689, 1222), (699, 1219), (700, 1229), (691, 1231)]]
SWORDS = [[(669, 1081), (673, 1085), (637, 1143), (632, 1140)], [(705, 1081), (710, 1080), (733, 1147), (728, 1150)]]


def pm(poly):
    return FR.poly(poly, smooth=6)


def line(pts, pen='g_pen', pressure=0.45, **kw):
    FR.line(sh_, pts, pen=pen, pressure=pressure, **kw)


disc = FR.disc


def walker(sh):
    """The walker, from behind: about 22 x 54 mm on the page, so every detail is a line or a tone patch."""
    global sh_
    sh_ = sh
    hat, rim, haori, hakama = pm(HAT), pm(HAT_RIM), pm(HAORI), pm(HAKAMA)
    hands = pm(HANDS[0]) | pm(HANDS[1])
    tabi = pm(TABI[0]) | pm(TABI[1]); soles = pm(SOLES[0]) | pm(SOLES[1])
    swords = pm(SWORDS[0]) | pm(SWORDS[1])
    allm = hat | haori | hakama | hands | tabi | soles | swords
    sh.fill(nd.binary_dilation(allm, iterations=2), mode='white')

    # ---- hat: woven sedge. Sunlit on the left, a tone step on the right, the brim's underside darker
    tone(sh, hat & ~rim, level=0.10, lpi=LPI)
    tone(sh, hat & ~rim & pm(HAT_SHADE), level=0.32, lpi=LPI)
    tone(sh, rim, level=0.45, lpi=LPI)
    sh.fill(rim & ~nd.binary_erosion(rim, iterations=2) & (np.arange(W)[None, :] < P(668, 0)[0]), mode='white')
    for k, ex in enumerate(np.linspace(655, 722, 13)):                 # ribs from the crown
        ey = 989 - 4 * np.cos((ex - 688) / 36)
        line([(688 + (ex - 688) * 0.08, 965.5), (ex, ey)], pen='maru', pressure=0.28 + 0.12 * (ex > 692),
             seed=60 + k, taper=(0.2, 0.6))
    for k, f in enumerate((0.35, 0.62, 0.85)):                        # weave rings across the cone
        xs = np.linspace(-1, 1, 9)
        pts = [(688 + xx_ * 34 * f, 965 + 24 * f - 3 * f * (1 - xx_ ** 2)) for xx_ in xs]
        line(pts, pen='maru', pressure=0.22, seed=75 + k, taper=(0.3, 0.3))
    line([(652, 989.5), (670, 992), (688, 993), (706, 992), (724, 989.5)], pen='maru', pressure=0.55, seed=70)
    sh.fill(disc((688, 965), 1.6))                                     # the knot at the crown

    # ---- haori: black, read through white-out: backlit rims, seams, folds, the crest
    sh.fill(haori)
    W_ = dict(pen='white', mode='white')
    line([(656, 1014), (645, 1030), (638, 1054), (634.5, 1083)], pressure=0.38, seed=80, **W_)   # sunlit rim, left
    line([(719, 1012), (729, 1030), (737, 1052), (742, 1078)], pressure=0.26, seed=81, **W_)     # rim, right
    line([(688, 1006), (688.5, 1036), (688, 1066)], pressure=0.16, seed=82, **W_)                # back seam
    line([(662, 1012), (676, 1005)], pressure=0.16, seed=83, **W_)                               # shoulder seams
    line([(701, 1005), (714, 1012)], pressure=0.16, seed=84, **W_)
    line([(663, 1030), (659, 1050), (655, 1074), (653, 1090)], pressure=0.2, seed=85, **W_)      # sleeve seams
    line([(714, 1030), (719, 1052), (724, 1074), (727, 1093)], pressure=0.2, seed=86, **W_)
    for k, fl in enumerate([[(669, 1036), (667, 1052), (668, 1064)], [(700, 1038), (702, 1054), (701, 1064)],
                            [(646, 1050), (643, 1070)], [(731, 1050), (735, 1070)]]):      # soft folds
        line(fl, pressure=0.12, seed=87 + k, taper=(0.8, 1.2), **W_)
    line([(647, 1093), (653, 1095), (659, 1094)], pressure=0.18, seed=92, **W_)                 # sleeve hems
    line([(717, 1097), (726, 1097), (735, 1093)], pressure=0.18, seed=93, **W_)
    sh.fill(disc((686, 1022), 4.3), mode='white')                                               # the crest
    sh.fill(disc((686, 1022), 4.3) & ~disc((686, 1022), 3.5))
    sh.fill(disc((686, 1022), 1.2))

    # ---- hakama: grey, the koshiita board and ties, pleats with shaded valleys, the stepping leg
    tone(sh, hakama, level=0.36, lpi=LPI)
    tone(sh, hakama & pm(HAKAMA_SHADE), level=0.58, lpi=LPI)
    tone(sh, hakama & pm(LEG_SHADE), level=0.55, lpi=LPI)
    tone(sh, pm(KOSHIITA), level=0.2, lpi=LPI)
    sh.fill(pm([(667, 1076), (707, 1076), (708, 1080), (666, 1080)]))                         # the himo ties
    prep.ink_outline(sh, pm(KOSHIITA), pen='maru', pressure=(0.4, 0.6), seed=100)
    for k, (pl, pr) in enumerate([([(676, 1084), (675.5, 1140), (673, 1198)], 0.42),
                                  ([(686, 1084), (687, 1130), (686, 1150)], 0.34),
                                  ([(694, 1086), (695, 1120), (694, 1142)], 0.3),
                                  ([(711, 1086), (714, 1140), (715, 1196)], 0.42),
                                  ([(666, 1092), (664, 1150), (663, 1195)], 0.32)]):
        line(pl, pressure=pr, seed=101 + k, taper=(0.3, 1.5))
    line([(694, 1142), (705, 1146), (712, 1151), (705, 1159), (692, 1171), (687, 1186), (686, 1201)],
         pressure=0.38, seed=110, taper=(0.4, 1.0))                                            # the stepping leg
    line([(698, 1162), (706, 1170), (708, 1190)], pressure=0.25, seed=111, taper=(0.5, 1.0))
    line([(657, 1120), (656, 1160), (657, 1196)], pressure=0.28, seed=112, mode='white', pen='white')  # backlight
    line([(658, 1199), (676, 1204.5), (700, 1206.5), (716, 1201)], pressure=0.45, seed=113)       # hem

    # ---- swords: black scabbards with a highlight, guards at the hip
    sh.fill(swords)
    line([(667, 1089), (640, 1135)], pen='white', pressure=0.12, mode='white', seed=120)
    line([(708, 1089), (727, 1138)], pen='white', pressure=0.12, mode='white', seed=121)
    for (c, d) in [((667, 1087), (-0.53, 0.85)), ((708, 1087), (0.35, 0.94))]:      # guards across each blade
        nx, ny = -d[1], d[0]
        g = [(c[0] + nx * 4 + d[0] * 1, c[1] + ny * 4 + d[1] * 1), (c[0] - nx * 4 + d[0] * 1, c[1] - ny * 4 + d[1] * 1),
             (c[0] - nx * 4 - d[0] * 1, c[1] - ny * 4 - d[1] * 1), (c[0] + nx * 4 - d[0] * 1, c[1] + ny * 4 - d[1] * 1)]
        sh.fill(pm(g))
    # ---- hands: skin left white; knuckles and a shaded edge
    for k, hnd in enumerate(HANDS):
        prep.ink_outline(sh, pm(hnd), pen='g_pen', pressure=(0.25, 0.45), seed=130 + k)
    tone(sh, pm([(652, 1096), (656, 1095), (656, 1108), (653, 1115), (650, 1112)]), level=0.4, lpi=LPI)
    tone(sh, pm([(729, 1097), (733, 1095), (737, 1104), (735, 1115), (729, 1112)]), level=0.4, lpi=LPI)
    line([(647, 1107), (652, 1109)], pen='maru', pressure=0.3, seed=140)
    line([(724, 1110), (730, 1112), (734, 1109)], pen='maru', pressure=0.3, seed=141)
    # ---- feet: white tabi, black zori soles, the back foot lifted
    sh.fill(soles)
    for k, t in enumerate(TABI):
        prep.ink_outline(sh, pm(t), pen='g_pen', pressure=(0.3, 0.5), seed=150 + k)
    line([(674, 1212), (678, 1209), (683, 1212)], pen='maru', pressure=0.35, seed=160)          # straps
    line([(689, 1210), (693, 1207), (698, 1211)], pen='maru', pressure=0.35, seed=161)
    # ---- silhouette, heavier on the side away from the setting sun
    prep.ink_outline(sh, allm, pen='g_pen', pressure=(0.45, 0.85), light=(-1.0, -0.4), seed=170)
    prep.ink_outline(sh, hat, pen='g_pen', pressure=(0.4, 0.6), seed=171)
    prep.ink_outline(sh, hakama, pen='g_pen', pressure=(0.35, 0.55), seed=172)


if __name__ == '__main__':
    t0 = time.time()
    sh = Sheet(W, H, px_per_mm=PPM, journal=sys.argv[1] if len(sys.argv) > 1 else None, seed=7)
    panel = [(OX, OY), (OX + 1840, OY), (OX + 1840, OY + 2760), (OX, OY + 2760)]
    with sh.clip(panel):
        sky(sh); print('sky', round(time.time() - t0))
        mountains(sh); print('mtn', round(time.time() - t0))
        village(sh); print('village', round(time.time() - t0))
        sd = seedlings_mask()
        paddies(sh, sd); print('paddies', round(time.time() - t0))
        reflections(sh)
        path(sh)
        nb = banks(sh, sd); print('banks', nb, round(time.time() - t0))
        ns = seedlings(sh, sd); print('seedlings', ns, round(time.time() - t0))
        walker(sh)
    effects.border(sh, panel, 0.8)
    sh.png('panel.png')
    print('strokes', sh.stats['strokes'], round(time.time() - t0), 's')
