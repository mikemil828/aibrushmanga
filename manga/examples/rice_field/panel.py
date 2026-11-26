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
from manga import Sheet, tone, hatch, effects, trace, prep, geom

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


def up(a):
    """A ref-resolution float array, scaled into page coordinates (zeros outside the panel)."""
    im = Image.fromarray(np.asarray(a, np.float32), 'F').resize((round(RW * S), round(RH * S)), Image.BILINEAR)
    out = np.zeros((H, W), np.float32)
    out[OY:OY + im.height, OX:OX + im.width] = np.asarray(im)
    return out


def upm(m):
    return up(np.asarray(m, np.float32)) > 0.5


def P(x, y):
    return (OX + S * x, OY + S * y)


def lay(sh, mask_ref, level_ref, kind='dot', angle=45.0):
    m = upm(mask_ref)
    tone(sh, m, level=up(np.where(mask_ref, level_ref, 0)), lpi=LPI, kind=kind, angle=angle)


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
    f = nd.gaussian_filter(cl.astype(np.float32), 1.0)
    n = 0
    for Pl in geom.outlines(f, 0.5, 25):
        Q = geom.simplify(Pl, 0.5)
        if len(Q) < 4:
            continue
        nrm = geom.normals(Q)
        low = nrm[:, 1] > 0.35                     # outward normal pointing down: the cloud's underside
        runs = np.split(np.arange(len(Q)), np.nonzero(np.diff(low.astype(int)))[0] + 1)
        for run in runs:
            if low[run[0]] and len(run) >= 4:
                sh.stroke([P(x, y) for x, y in Q[run]], pen='maru', pressure=0.3, seed=200 + n)
                n += 1


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
    n = 0
    for Pl in geom.outlines(skyf, 0.5, 12):
        ok = np.array([near_m[int(min(y, RH - 1)), int(min(x, RW - 1))] and y > 270 for x, y in Pl])
        runs = np.split(np.arange(len(Pl)), np.nonzero(np.diff(ok.astype(int)))[0] + 1)
        for run in runs:
            if not ok[run[0]] or len(run) < 15:
                continue
            Q = geom.simplify(Pl[run], 0.6)
            rr = near_r[int(Q[len(Q) // 2][1]), int(Q[len(Q) // 2][0])]
            pts = [P(x, y) for x, y in Q]
            for k in range(0, len(pts) - 1, 40):          # hand-length pieces
                seg = pts[k:k + 42]
                if len(seg) > 1:
                    sh.stroke(seg, pen='saji' if rr else 'maru', pressure=0.35 if rr else 0.45, seed=300 + n)
                    n += 1


# ------------------------------------------------------------------ village: trees, pine, houses
def village(sh):
    t = Z['trees']
    blk = nd.binary_opening(t & (Ls < 0.22), iterations=1)
    lay(sh, t & ~blk, np.full(L.shape, 0.62))
    sh.fill(upm(blk))
    sh.fill(upm(Z['pine']))
    h = Z['houses']
    hp = upm(h)
    sh.fill(nd.binary_dilation(hp, iterations=3) & ~hp, mode='white')     # keep roofs off the black trees
    win = h & (ref[..., 0] > 0.55) & (ref[..., 1] > 0.42) & (ref[..., 2] < 0.5)
    wall = h & (Ls < 0.26) & ~win
    roof = h & ~wall & ~win
    lay(sh, roof, np.clip(0.55 - 0.5 * (Ls - 0.35), 0.25, 0.6))
    sh.fill(upm(nd.binary_opening(wall, iterations=1)))
    # thatch: short strokes down the roof
    rng = np.random.default_rng(5)
    rm = upm(roof)
    ys, xs = np.nonzero(rm)
    pick = rng.choice(len(xs), size=min(len(xs) // 90, 900), replace=False)
    for i in pick:
        x, y = xs[i], ys[i]
        ln = rng.uniform(5, 11)
        if rm[min(int(y + ln), H - 1), x]:
            sh.stroke([(x, y), (x + rng.uniform(-1.5, 1.5), y + ln)], pen='maru', pressure=0.5, seed=int(i))
    prep.ink_outline(sh, upm(roof | wall), pen='maru', pressure=(0.4, 0.7), seed=8)
    lit = roof & (ref[..., 0] - ref[..., 2] > 0.18) & (Ls > 0.45)
    sh.fill(upm(lit), mode='white')
    sh.fill(upm(win), mode='white')


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


def paddies(sh, sd):
    w = Z['water'] | Z['shadow'] | sd
    # the water under each tuft takes the water around it (normalized blur over water pixels only)
    ww = (Z['water'] | Z['shadow']).astype(np.float32)
    Lw = nd.gaussian_filter(L * ww, 4) / np.maximum(nd.gaussian_filter(ww, 4), 1e-3)
    Lw = np.where(sd, Lw, L)
    lv = np.clip(1.0 * (0.78 - nd.gaussian_filter(Lw, 2.5)), 0, 0.5)
    lv[lv < 0.04] = 0
    lv = np.where(Z['shadow'], 0.72, lv)
    lay(sh, w, lv, kind='line', angle=90.0)          # angle 90 = horizontal lines
    glint = w & (Ls > 0.9)
    sh.fill(upm(nd.binary_opening(glint, iterations=1)), mode='white')
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
    km = np.zeros(nm + 1, bool); km[1:] = am > 900
    mud = km[labm]
    sh.fill(upm(mud))


def banks(sh, sd):
    b = Z['levee'] & ~sd
    Lb = nd.gaussian_filter(L, 2.0)
    lv = np.where(Lb > 0.42, 0.28, 0.52)
    lv = np.where(Lb < 0.22, 0.7, lv)
    lay(sh, b, lv)
    # grass blades: the dark lines of the reference, inked where the banks are close
    rgbp = np.stack([up(ref[..., i]) for i in range(3)], -1)
    lines = prep.xdog_lines(rgbp, sigma=1.1, p=22, eps=-0.012, phi=70, tidy=12) > 0.5
    near = upm(b & (YY > 1080))
    n, _ = trace.ink_line_art(sh, lines & near, pens=('maru', 'g_pen'), weight=1.0, min_len_px=10, min_px=1.0,
                              seed=31, even=0.5)
    dew = b & (L > 0.72) & (YY > 1000)
    sh.fill(upm(dew), mode='white')
    return n


def seedlings(sh, sd):
    """Each tuft drawn as a few leaves fanning up from its base, with a short broken reflection below."""
    lab, n = nd.label(sd)
    rng = np.random.default_rng(41)
    k = 0
    for i, sl in enumerate(nd.find_objects(lab), 1):
        ys, xs = np.nonzero(lab[sl] == i)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        top, base_y = ys.min(), ys.max()
        h = base_y - top + 1
        if h < 3:
            continue
        bx = xs[ys >= base_y - max(1, h * 0.15)].mean()
        w = xs.max() - xs.min() + 1
        near = np.clip((base_y - 900) / 800, 0, 1)                 # 0 far .. 1 front
        nleaf = 2 + (h > 10) + (h > 22) + (h > 40)
        spread = np.clip(0.6 * w / h, 0.25, 0.9)
        for j in range(nleaf):
            t = (j + 0.5) / nleaf * 2 - 1
            a = t * spread + rng.normal(0, 0.08)
            ln = h * rng.uniform(0.75, 1.05) * (1 - 0.25 * abs(t))
            tip = (bx + np.sin(a) * ln, base_y - np.cos(a) * ln)
            mid = (bx + np.sin(a * 0.6) * ln * 0.5, base_y - np.cos(a * 0.6) * ln * 0.5)
            pts = [P(bx + rng.normal(0, 0.3), base_y), P(*mid), P(*tip)]
            sh.stroke(pts, pen='g_pen', pressure=0.25 + 0.4 * near, taper=(0.1, 1.2 + 1.5 * near), seed=900 + k)
            k += 1
        if h > 8:                                                   # the reflection: two or three broken lines
            for j in range(2 + (h > 25)):
                a = (j - 1) * spread * 0.7
                ln = h * rng.uniform(0.35, 0.55)
                pts = [P(bx + np.sin(a) * 2, base_y + 2), P(bx + np.sin(a) * ln, base_y + np.cos(a) * ln)]
                sh.stroke(pts, pen='maru', pressure=0.35, taper=(0.3, 1.0), seed=5000 + k)
                k += 1
    return k


def path(sh):
    p = Z['path']
    Lp = nd.gaussian_filter(L, 4.0)
    lv = np.where(Lp < 0.36, 0.38, 0.14)
    lv = np.where(Lp > 0.6, 0.0, lv)
    lv = np.where(Z['shadow'], 0.62, lv)
    lay(sh, p, lv)


# ------------------------------------------------------------------ the walker (hand-drawn from the reference)
HAT = [(688, 963), (700, 966), (717, 977), (724, 990), (722, 996), (706, 1001), (690, 1001), (672, 999), (655, 995),
       (651, 990), (656, 980), (676, 967)]
HAT_RIM = [(651, 989), (724, 989), (722, 996), (706, 1001), (690, 1001), (672, 999), (655, 995)]
HAORI = [(675, 1001), (705, 1001), (719, 1010), (731, 1030), (742, 1062), (744, 1082), (736, 1094), (716, 1098),
         (712, 1084), (706, 1068), (670, 1068), (664, 1084), (660, 1095), (646, 1094), (633, 1086), (636, 1056),
         (648, 1022), (662, 1008)]
HAKAMA = [(670, 1068), (706, 1068), (712, 1084), (716, 1100), (721, 1150), (722, 1192), (714, 1203), (700, 1206),
          (682, 1206), (661, 1204), (659, 1180), (661, 1120), (662, 1095), (664, 1084)]
HANDS = [[(646, 1095), (658, 1096), (657, 1114), (650, 1116), (645, 1108)],
         [(717, 1098), (735, 1094), (737, 1110), (730, 1118), (722, 1114)]]
FEET = [[(672, 1204), (684, 1204), (684, 1218), (676, 1222), (671, 1214)],
        [(687, 1203), (698, 1204), (699, 1226), (691, 1229), (686, 1214)]]
HAKAMA_SHADE = [(704, 1086), (712, 1086), (716, 1100), (721, 1150), (722, 1192), (714, 1203), (704, 1205), (703, 1150)]
SWORDS = [[(670, 1080), (674, 1084), (637, 1141), (632, 1138)], [(705, 1080), (710, 1079), (732, 1146), (727, 1149)]]


def pm(poly):
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon([P(*q) for q in geom.catmull(np.array(poly + poly[:1], float), 6)], fill=1)
    return np.asarray(im, bool)


def line(pts, pen='g_pen', pressure=0.45, mode='ink', seed=0, per=8):
    Q = geom.catmull(np.array(pts, float), per) if len(pts) > 2 else np.array(pts, float)
    sh_.stroke([P(*q) for q in Q], pen=pen, pressure=pressure, mode=mode, seed=seed)


def walker(sh):
    global sh_
    sh_ = sh
    hat, rim, haori, hakama = pm(HAT), pm(HAT_RIM), pm(HAORI), pm(HAKAMA)
    hands = pm(HANDS[0]) | pm(HANDS[1]); feet = pm(FEET[0]) | pm(FEET[1])
    swords = pm(SWORDS[0]) | pm(SWORDS[1])
    allm = hat | haori | hakama | hands | feet | swords
    sh.fill(nd.binary_dilation(allm, iterations=2), mode='white')
    # materials
    tone(sh, hat & ~rim, level=0.12, lpi=LPI)
    tone(sh, rim, level=0.42, lpi=LPI)
    tone(sh, hakama, level=0.40, lpi=LPI)
    tone(sh, hakama & pm(HAKAMA_SHADE), level=0.6, lpi=LPI)          # the hakama's shaded side
    sh.fill(haori); sh.fill(feet); sh.fill(swords)
    # hat: weave lines from the apex, brim edge
    for k, ex in enumerate(np.linspace(656, 721, 9)):
        ey = 990 - 6 * np.cos((ex - 688) / 40)
        line([(688 + (ex - 688) * 0.12, 966), (ex, ey)], pen='maru', pressure=0.35, seed=60 + k)
    line([(652, 989), (670, 992), (688, 993), (706, 992), (723, 989)], pen='maru', pressure=0.5, seed=70)
    # haori: back seam, sleeve folds and the backlit rims, scraped in white
    line([(688, 1004), (688.5, 1035), (688, 1066)], pen='white', pressure=0.18, mode='white', seed=80)
    line([(661, 1028), (654, 1058), (651, 1088)], pen='white', pressure=0.22, mode='white', seed=81)
    line([(716, 1028), (723, 1058), (728, 1090)], pen='white', pressure=0.22, mode='white', seed=82)
    line([(646, 1024), (638, 1052), (635, 1084)], pen='white', pressure=0.3, mode='white', seed=83)
    line([(727, 1026), (737, 1052), (742, 1078)], pen='white', pressure=0.24, mode='white', seed=84)
    line([(664, 1010), (675, 1004)], pen='white', pressure=0.2, mode='white', seed=85)
    yy, xx = np.mgrid[:H, :W]
    cx, cy = P(686, 1022)
    sh.fill((xx - cx) ** 2 + (yy - cy) ** 2 < (4.2 * S) ** 2, mode='white')      # the crest
    # hakama: waist band and pleats
    line([(669, 1077), (688, 1079), (707, 1077)], pressure=0.35, seed=90)
    for k, pl in enumerate([[(676, 1086), (675, 1140), (673, 1198)], [(688, 1082), (689, 1120), (688, 1156)],
                            [(699, 1088), (701, 1140), (702, 1196)], [(708, 1100), (712, 1150), (713, 1185)],
                            [(667, 1100), (665, 1150), (664, 1190)]]):
        line(pl, pressure=0.32 + 0.06 * (k % 2), seed=91 + k)
    # hands: outline only
    for k, hnd in enumerate(HANDS):
        prep.ink_outline(sh, pm(hnd), pen='g_pen', pressure=(0.25, 0.45), seed=110 + k)
    # outline of the whole figure, heavier on the side away from the setting sun
    prep.ink_outline(sh, allm, pen='g_pen', pressure=(0.45, 0.8), light=(-1.0, -0.4), seed=120)
    prep.ink_outline(sh, hat, pen='g_pen', pressure=(0.4, 0.6), seed=121)
    prep.ink_outline(sh, hakama, pen='g_pen', pressure=(0.35, 0.55), seed=122)


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
        path(sh)
        nb = banks(sh, sd); print('banks', nb, round(time.time() - t0))
        ns = seedlings(sh, sd); print('seedlings', ns, round(time.time() - t0))
        walker(sh)
    effects.border(sh, panel, 0.8)
    sh.png('panel.png')
    print('strokes', sh.stats['strokes'], round(time.time() - t0), 's')
