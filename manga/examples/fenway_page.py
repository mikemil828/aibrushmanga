"""Fenway Park as a manga page: the first real test of the ink engine.

Decided before drawing (see the Advice in README):
  * Page: B4 manga original (257 x 364 mm), 8 px/mm draft; print by replaying the journal at --scale 2-3.
  * Panel 1, full width: establishing shot of the park. Primary: the Green Monster as the one big black mass,
    the diamond as the light shape. Sky stays paper with a soft gradient tone; crowd is sand tone, not lines.
  * Panel 2, wide: the Monster and the Citgo sign up close. Monster = solid black with white-out lettering;
    light towers ruled by hand with a fineliner; skyline in light tone and thin maru outlines.
  * Panel 3, tall: the fan with the phone. The only panel with heavy G-pen contour work and focus lines;
    clothes in flat tone, hair and cap in black.
  * Panel 4, wide: the infield at ground level, players lined up. Grass in two-tone stripes, dirt in sand tone,
    chalk lines in white-out, speed lines for the moment before the first pitch.
Usage: python fenway_page.py <ref_dir> <out_prefix>   (ref_dir holds photo_full.png and zones.npy)
"""
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage as nd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from manga import Sheet, hatch, tone, effects, prep  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'fenway_manga'
ONLY = sys.argv[3].split(',') if len(sys.argv) > 3 else None
W, H, PPM = 2056, 2912, 8.0
LPI = 32                                     # coarse enough to read on screen at 8 px/mm

photo = Image.open(os.path.join(REF, 'photo_full.png')).convert('RGB')
ZONES = np.load(os.path.join(REF, 'zones.npy'))           # 2800 x 1252 reference grid
RW, RH = 2800, 1252
SX = photo.size[0] / RW
SKY, STANDS, BUILDING, GRASS, DIRT, MONSTER, SKYLINE, FORE, BOARD = 1, 2, 3, 4, 5, 6, 7, 8, 9

t0 = time.time()
sh = Sheet(W, H, px_per_mm=PPM, journal=OUT + '.jsonl', seed=7)
rows = effects.layout(W, H, [(0.80, [1]), (1.0, [1.55, 1]), (0.78, [1])], margin_mm=11, gutter_mm=(3.5, 7),
                      px_per_mm=PPM, slant={(1, 0): (40, -40)})
P1, (P2, P3), P4 = rows[0][0], rows[1], rows[2][0]


def bbox(p):
    xs, ys = [q[0] for q in p], [q[1] for q in p]
    return int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))


def panel_source(poly, crop):
    """Photo + zones for a panel. crop = (x0, y0, x1, y1) in 2800-reference px; cover-fit to the panel box."""
    x0, y0, x1, y1 = bbox(poly)
    pw, ph = x1 - x0, y1 - y0
    cx0, cy0, cx1, cy1 = crop
    cw, ch = cx1 - cx0, cy1 - cy0
    s = max(pw / cw, ph / ch)
    vw, vh = pw / s, ph / s                       # visible part of the crop, centred
    vx0, vy0 = cx0 + (cw - vw) / 2, cy0 + (ch - vh) / 2
    rgb = photo.resize((pw, ph), Image.LANCZOS, box=(vx0 * SX, vy0 * SX, (vx0 + vw) * SX, (vy0 + vh) * SX))
    zi = Image.fromarray(ZONES).resize((pw, ph), Image.NEAREST, box=(vx0, vy0, vx0 + vw, vy0 + vh))
    to_ref = lambda x, y: (vx0 + (x - x0) / s, vy0 + (y - y0) / s)      # noqa: E731
    to_page = lambda rx, ry: (x0 + (rx - vx0) * s, y0 + (ry - vy0) * s)  # noqa: E731
    return np.asarray(rgb, np.float32) / 255, np.asarray(zi), (x0, y0, x1, y1), to_page, s


def page(a, box, fill=0):
    """Place a panel-size array onto a page-size array."""
    x0, y0, x1, y1 = box
    out = np.full((H, W), fill, dtype=a.dtype)
    out[y0:y1, x0:x1] = a[:y1 - y0, :x1 - x0]
    return out


def smooth_bool(m, sigma=1.2):
    return nd.gaussian_filter(m.astype(np.float32), sigma) > 0.5


def levels(sh, d, zone_mask, box, cuts, lpi=LPI, angle=45, outline=None, black=None, min_area=40):
    """Quantise a darkness map inside a zone into flat tones (manga uses a few stock tones, not a gradient).

    cuts  -- [(threshold, tone_level), ...] ascending: pixels darker than threshold get that tone.
    black -- threshold above which the area is solid black (beta).
    outline -- (pen, threshold list) to ink the borders between levels.
    """
    zm = zone_mask
    prev = None
    for thr, lv in cuts[::-1]:
        m = smooth_bool((d > thr) & zm)
        if prev is not None:
            m &= ~prev
        if black is not None:
            m &= ~(d > black)
        m = nd.binary_opening(m, iterations=1)
        lab, n = nd.label(m)
        if n:
            sz = nd.sum(m, lab, range(1, n + 1))
            m = np.isin(lab, 1 + np.nonzero(sz >= min_area)[0])
        if m.any():
            tone(sh, page(m, box, False), level=lv, lpi=lpi, angle=angle)
        prev = m if prev is None else (prev | m)
    if black is not None:
        b = smooth_bool((d > black) & zm, 1.0)
        lab, n = nd.label(b)
        if n:
            sz = nd.sum(b, lab, range(1, n + 1))
            b = np.isin(lab, 1 + np.nonzero(sz >= min_area // 2)[0])
        if b.any():
            sh.fill(page(b, box, False))
    if outline:
        pen, thrs, pr = outline
        for thr in thrs:
            m = smooth_bool((d > thr) & zm, 1.5)
            prep.ink_outline(sh, page(m, box, False), pen=pen, pressure=pr, min_len=14, sigma=1.5,
                             seed=int(thr * 1000) + box[0])


def light_tower(sh, to_page, s, base, top, head_w):
    """Hand-ruled light tower: two legs, cross bracing, a bank of lamps (all in reference px)."""
    bx, by = to_page(*base)
    tx, ty = to_page(*top)
    hw = head_w * s
    leg = max(4.0, hw * 0.12)
    for dx in (-leg, leg):
        sh.stroke([(bx + dx * 1.6, by), (tx + dx, ty + hw * 0.35)], pen='fineliner', taper=(0, 0), smooth=False)
    n = max(3, int((by - ty) / (leg * 4)))
    for i in range(n):
        f0, f1 = i / n, (i + 1) / n
        ya, yb = by + (ty + hw * 0.35 - by) * f0, by + (ty + hw * 0.35 - by) * f1
        wa, wb = leg * (1.6 - 0.6 * f0), leg * (1.6 - 0.6 * f1)
        xa, xb = bx + (tx - bx) * f0, bx + (tx - bx) * f1
        sh.stroke([(xa - wa, ya), (xb + wb, yb)], pen='maru', pressure=0.6, taper=(0, 0), smooth=False)
        sh.stroke([(xa + wa, ya), (xb - wb, yb)], pen='maru', pressure=0.6, taper=(0, 0), smooth=False)
    # lamp bank: black frame, white lamps
    hx0, hx1, hy0, hy1 = tx - hw / 2, tx + hw / 2, ty - hw * 0.28, ty + hw * 0.35
    sh.fill([(hx0, hy0), (hx1, hy0), (hx1, hy1), (hx0, hy1)])
    nx, ny = max(3, int(hw / 9)), 3
    for i in range(nx):
        for j in range(ny):
            cx = hx0 + (i + 0.5) * (hx1 - hx0) / nx
            cy = hy0 + (j + 0.5) * (hy1 - hy0) / ny
            sh.stroke([(cx, cy)], pen='white', pressure=0.9 if hw > 60 else 0.5, mode='white')


def citgo(sh, to_page, s, x0, y0, x1, y1):
    a, b = to_page(x0, y0), to_page(x1, y1)
    sh.fill([(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])], mode='white')
    w, h = b[0] - a[0], b[1] - a[1]
    tri = [(a[0] + w * 0.5, a[1] + h * 0.1), (a[0] + w * 0.86, a[1] + h * 0.68), (a[0] + w * 0.14, a[1] + h * 0.68)]
    sh.fill(tri)
    tone(sh, [(a[0], a[1] + h * 0.74), (b[0], a[1] + h * 0.74), (b[0], b[1]), (a[0], b[1])], level=0.5, lpi=LPI)
    sh.stroke([a, (b[0], a[1]), b, (a[0], b[1]), a], pen='fineliner', taper=(0, 0), smooth=False)


def establishing():
    rgb, Zp, box, to_page, s = panel_source(P1, (0, 0, RW, RH))
    d = prep.darkness(rgb, blur=1.0)
    dv = prep.darkness(rgb, blur=2.5, mode='value', mix=0.6)
    x0, y0, x1, y1 = box
    with sh.clip(P1):
        sh.note('panel 1: establishing shot')
        # sky: paper, light gradient dots from the top
        sky = Zp == SKY
        tone(sh, page(smooth_bool(sky), box, False), level=(0.16, 0.0, (x0, y0), (x0, y0 + (y1 - y0) * 0.22)),
             lpi=LPI)
        # skyline and buildings beyond: light, thin
        levels(sh, d, Zp == SKYLINE, box, [(0.30, 0.15), (0.45, 0.35)], black=0.70,
               outline=('maru', [0.45], (0.2, 0.5)))
        # stadium structure left/right
        levels(sh, d, Zp == BUILDING, box, [(0.25, 0.2), (0.45, 0.45)], black=0.66,
               outline=('saji', [0.45], (0.3, 0.7)))
        # stands: crowd as sand tone, shadows black
        st = Zp == STANDS
        tone(sh, page(smooth_bool(st, 2.0), box, False), level=0.3, lpi=LPI * 0.8, angle=-3, kind='line')
        b = smooth_bool(st & (dv > 0.66), 3.5)
        lab, n = nd.label(b)
        if n:
            sz = nd.sum(b, lab, range(1, n + 1))
            b = np.isin(lab, 1 + np.nonzero(sz >= 400)[0])
        sh.fill(page(b, box, False))
        hatch(sh, page(np.where(st, dv, 0).astype(np.float32), box, 0.0), page(st & ~b, box, False), pen='maru',
              angle=-4, spacing_mm=1.3, layers=((0.45, 0.0),), stroke_mm=(6, 14), pressure=(0.2, 0.6), seed=13)
        # the Monster: one black mass, lettering scraped white
        mon = (Zp == MONSTER) | (Zp == BOARD)
        sh.fill(page(smooth_bool(mon, 1.0), box, False))
        lett = smooth_bool(mon & (d < 0.42), 0.6)
        sh.fill(page(lett, box, False), mode='white')
        # grass: mowing stripes as two stock tones; dirt as sand
        gr = Zp == GRASS
        med = np.median(d[gr]) if gr.any() else 0.5
        tone(sh, page(smooth_bool(gr & (d <= med), 1.5), box, False), level=0.18, lpi=LPI, angle=45)
        tone(sh, page(smooth_bool(gr & (d > med), 1.5), box, False), level=0.32, lpi=LPI, angle=45)
        dirt = smooth_bool(Zp == DIRT, 1.2)
        tone(sh, page(dirt, box, False), level=0.22, kind='noise', grain_mm=0.15, seed=12)
        prep.ink_outline(sh, page(dirt, box, False), pen='maru', pressure=(0.3, 0.6), min_len=20, seed=3)
        # foreground: people, seats, rail - ink drawing with flat tones
        # foreground: a black silhouette band framing the bright park; only white shirts and lit edges survive
        # only what is really near is silhouetted: below the rail line, and the fans on the right
        ox, oy = to_page(0, 0)
        ry_, rx_ = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        rx_ = (rx_ - (ox - x0)) / s
        ry_ = (ry_ - (oy - y0)) / s
        rail = np.interp(rx_, [0, 260, 600, 1000, 1400, 2000, 2300, 2800], [600, 660, 850, 900, 890, 800, 640, 560])
        near = (Zp == FORE) & ((ry_ > rail) | ((rx_ > 1930) & (ry_ > 560)) | ((rx_ > 2480) & (ry_ > 430)))
        far = (Zp == FORE) & ~near
        tone(sh, page(smooth_bool(far, 2.0), box, False), level=0.3, lpi=LPI * 0.8, angle=-3, kind='line')
        fb = smooth_bool(far & (dv > 0.6), 3.5)
        sh.fill(page(fb, box, False))
        hatch(sh, page(np.where(far, dv, 0).astype(np.float32), box, 0.0), page(far & ~fb, box, False), pen='maru',
              angle=-4, spacing_mm=1.3, layers=((0.45, 0.0),), stroke_mm=(6, 14), pressure=(0.2, 0.6), seed=14)
        fg = smooth_bool(near, 1.5)
        sh.fill(page(fg, box, False))
        # shiro-nuki: seat backs, arms and shirts drawn as white lines inside the black
        for thr, pr in ((0.40, (0.25, 0.6)), (0.28, (0.35, 0.8))):
            m = smooth_bool(fg & (dv < thr), 2.5)
            prep.ink_outline(sh, page(m, box, False), pen='white', pressure=pr, min_len=40, sigma=2.0,
                             seed=int(thr * 100), mode='white')
        # rail: a lit top edge in white-out
        xs = np.linspace(260, 2300, 40)
        ys = np.interp(xs, [260, 600, 1000, 1400, 2000, 2300], [660, 850, 900, 890, 800, 640]) - 4
        sh.stroke([(*to_page(a, b), 0.55) for a, b in zip(xs, ys)], pen='white', mode='white', seed=15)
        lit = smooth_bool(fg & (dv < 0.24), 2.5)
        lab, n = nd.label(lit)
        if n:
            sz = nd.sum(lit, lab, range(1, n + 1))
            lit = np.isin(lab, 1 + np.nonzero(sz >= 150)[0])
        sh.fill(page(lit, box, False), mode='white')
        tone(sh, page(lit & (dv > 0.12), box, False), level=0.2, lpi=LPI)
        prep.ink_outline(sh, page(lit, box, False), pen='saji', pressure=(0.3, 0.7), min_len=30, seed=5)
        # hand-placed: light towers and the Citgo sign
        for base, top, hw in [((532, 260), (532, 70), 50), ((990, 300), (994, 110), 52),
                              ((1313, 330), (1313, 140), 38), ((1640, 330), (1638, 150), 38),
                              ((2384, 330), (2384, 185), 40)]:
            light_tower(sh, to_page, s, base, top, hw)
        citgo(sh, to_page, s, 1539, 202, 1580, 262)
    effects.border(sh, P1)
    effects.caption(sh, (x0 + 14, y0 + 14), 'FENWAY PARK, BOSTON\nJULY 30, 2019  6:56 PM', size_mm=3.2)


def monster_close():
    rgb, Zp, box, to_page, s = panel_source(P2, (1120, 60, 2060, 560))
    d = prep.darkness(rgb, blur=1.2)
    x0, y0, x1, y1 = box
    with sh.clip(P2):
        sh.note('panel 2: the Monster close up')
        sky = Zp == SKY
        tone(sh, page(smooth_bool(sky), box, False), level=(0.15, 0.0, (x0, y0), (x0, y0 + (y1 - y0) * 0.3)),
             lpi=LPI, angle=45)
        levels(sh, d, Zp == SKYLINE, box, [(0.28, 0.12), (0.45, 0.3)], black=0.72,
               outline=('maru', [0.36], (0.2, 0.55)))
        st = Zp == STANDS
        tone(sh, page(smooth_bool(st, 2.0), box, False), level=0.3, lpi=LPI * 0.8, angle=-3, kind='line')
        sh.fill(page(smooth_bool(st & (d > 0.66), 1.0), box, False))
        hatch(sh, page(np.where(st, d, 0).astype(np.float32), box, 0.0), page(st, box, False), pen='maru',
              angle=0, spacing_mm=1.1, layers=((0.55, 0.0),), stroke_mm=(8, 18), seed=22)
        mon = (Zp == MONSTER) | (Zp == BOARD)
        sh.fill(page(smooth_bool(mon, 1.0), box, False))
        lett = smooth_bool(mon & (d < 0.22), 0.5)
        sh.fill(page(lett, box, False), mode='white')
        gr = Zp == GRASS
        med = np.median(d[gr]) if gr.any() else 0.5
        tone(sh, page(smooth_bool(gr & (d <= med), 1.5), box, False), level=0.2, lpi=LPI)
        tone(sh, page(smooth_bool(gr & (d > med), 1.5), box, False), level=0.36, lpi=LPI)
        dirt = smooth_bool(Zp == DIRT, 1.2)
        tone(sh, page(dirt, box, False), level=0.25, kind='noise', grain_mm=0.15, seed=23)
        levels(sh, d, Zp == BUILDING, box, [(0.25, 0.2), (0.45, 0.45)], black=0.66,
               outline=('saji', [0.45], (0.3, 0.7)))
        for base, top, hw in [((1313, 380), (1313, 140), 38), ((1640, 380), (1638, 150), 38)]:
            light_tower(sh, to_page, s, base, top, hw)
        citgo(sh, to_page, s, 1539, 202, 1580, 262)
    effects.border(sh, P2)


def colour_classes(rgb, blur=2.0):
    """What an inker sees: black (hair, caps, deep shadow), red, blue, green, skin/light. Returns label map."""
    c = nd.gaussian_filter(rgb, (blur, blur, 0))
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = c.max(2)
    lab = np.zeros(r.shape, np.uint8)                       # 0 = paper (skin, white, light)
    lab[(r > g + 0.18) & (r > b + 0.15)] = 1                # red
    lab[(b > r + 0.10) & (b > g - 0.02)] = 2                # blue
    lab[(g > r + 0.04) & (g > b - 0.02) & (mx < 0.75)] = 3  # green
    lab[(mx < 0.30)] = 4                                    # black
    for k in range(5):                                      # tidy: no specks
        m = lab == k
        m2 = nd.binary_opening(m, iterations=2)
        lab[m & ~m2] = 255
    # fill tidied pixels from the nearest label
    bad = lab == 255
    if bad.any():
        idx = nd.distance_transform_edt(bad, return_distances=False, return_indices=True)
        lab = lab[tuple(idx)]
    return lab


def fan():
    rgb, Zp, box, to_page, s = panel_source(P3, (2230, 330, 2800, 1252))
    x0, y0, x1, y1 = box
    lab = colour_classes(rgb, 1.5)
    # the people and the near seats, outlined by hand on the photo (reference px)
    near = [(2465, 395), (2500, 345), (2600, 340), (2650, 360), (2680, 340), (2760, 340), (2800, 360),
            (2800, 1252), (2200, 1252), (2200, 770), (2310, 740), (2410, 780), (2440, 680), (2500, 600),
            (2500, 530), (2490, 470)]
    from PIL import ImageDraw
    m = Image.new('L', (x1 - x0, y1 - y0), 0)
    ox, oy = to_page(0, 0)
    ImageDraw.Draw(m).polygon([((rx * s + ox - x0), (ry * s + oy - y0)) for rx, ry in near], fill=255)
    bg = np.asarray(m) < 128
    people = ~bg
    with sh.clip(P3):
        sh.note('panel 3: the fan with the phone')
        # background drops away: focus lines toward the phone, on paper, behind the people
        c = to_page(2445, 1030)
        with sh.clip(page(bg, box, False)):
            effects.focus_lines(sh, c, (0.3 * (x1 - x0), 0.25 * (y1 - y0)), box, count=170, pen='g_pen',
                                pressure=(0.35, 0.9), seed=33)
        # people: flat stock tones by colour, black for hair, cap and shadow
        tone(sh, page(people & (lab == 1), box, False), level=0.32, lpi=LPI, angle=45)
        tone(sh, page(people & (lab == 2), box, False), level=0.55, lpi=LPI, angle=45)
        tone(sh, page(people & (lab == 3), box, False), level=0.2, lpi=LPI * 0.8, angle=0, kind='line')
        sh.fill(page(people & (lab == 4), box, False))
        # contour guides from the photo, kept only on the people; the silhouette gets a heavy G-pen line
        lines = prep.xdog_lines(rgb, sigma=1.6, p=22, tidy=30) * people
        sh.apply(page(lines.astype(np.float32), box, 0.0), kind='lines')
        prep.ink_outline(sh, page(people, box, False), pen='g_pen', pressure=(0.6, 1.0), min_len=25, sigma=2.5,
                         seed=45)
        # phone screen glow: white-out over the screen and a small flash
        sh.stroke([to_page(2445, 1020), to_page(2470, 1040)], pen='white', pressure=0.9, mode='white')
    effects.border(sh, P3)


def infield():
    rgb, Zp, box, to_page, s = panel_source(P4, (980, 520, 1950, 880))
    d = prep.darkness(rgb, blur=1.2)
    dv = prep.darkness(rgb, blur=2.0, mode='value', mix=0.6)
    x0, y0, x1, y1 = box
    c = nd.gaussian_filter(rgb, (2, 2, 0))
    green = (c[..., 0] - c[..., 1]) < 0.13
    with sh.clip(P4):
        sh.note('panel 4: the infield')
        fg = smooth_bool(Zp == FORE, 2)
        field = ~fg
        gr = smooth_bool(field & green, 2.5)
        dirt = smooth_bool(field & ~green, 2.5)
        med = np.median(d[gr]) if gr.any() else 0.5
        tone(sh, page(gr & (nd.gaussian_filter(d, 4) <= med), box, False), level=0.15, lpi=LPI, angle=45)
        tone(sh, page(gr & (nd.gaussian_filter(d, 4) > med), box, False), level=0.3, lpi=LPI, angle=45)
        tone(sh, page(dirt, box, False), level=0.22, kind='noise', grain_mm=0.14, seed=41)
        prep.ink_outline(sh, page(dirt, box, False), pen='saji', pressure=(0.3, 0.75), min_len=40, seed=42)
        chalk = smooth_bool(field & (d < 0.12), 0.6)
        sh.fill(page(chalk, box, False), mode='white')
        figs = smooth_bool(field & (dv > 0.55), 0.6)
        sh.fill(page(figs, box, False))
        sh.fill(page(fg, box, False))
        m = smooth_bool(fg & (dv < 0.35), 2.5)
        prep.ink_outline(sh, page(m, box, False), pen='white', pressure=(0.25, 0.6), min_len=30, sigma=2.0,
                         seed=44, mode='white')
    effects.border(sh, P4)
    effects.caption(sh, (x1 - 560, y0 + 20), 'FIRST PITCH IN TEN MINUTES.', size_mm=3.0)


panels = {'1': establishing, '2': monster_close, '3': fan, '4': infield}
for k, f in panels.items():
    if ONLY and k not in ONLY:
        continue
    t = time.time()
    f()
    print(f'panel {k}: {time.time() - t:.0f}s, strokes so far {sh.stats["strokes"]}', flush=True)
sh.png(OUT + '.png')
sh.close()
print('page done', round(time.time() - t0), 's', sh.stats)
