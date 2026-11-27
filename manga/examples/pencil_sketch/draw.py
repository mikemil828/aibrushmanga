"""A girl's bust drawn without a reference: pencil first, then ink over the decided sketch.

Plan, decided before drawing (the pencil layer is where it gets checked):
  primary    eyes (lash band heavy, iris with a dark top, two catch-lights), then the face contour
  secondary  hair: a bob with pointed bangs parted on her right, light tone and a few flow strands
  tertiary   neck, shoulders and a sailor collar, broad and plain
  background paper with a soft halo, nothing else
Order: two pose thumbnails in pencil, pick one; construct the head and body in pencil; rough the features;
then ink: eyes, face, hair, body, tone. Pens: g_pen for the silhouette and hair, maru for inner lines,
saji for clothes. Every inked line follows a point the pencil pass constructed (the dict head() returns).

Outputs: thumbnails.png (two poses), sketch.png (pencil only), sketch-under-ink.png, page.png, page.jsonl.
Run from this folder: python3 draw.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from manga import Sheet, tone, face, effects            # noqa: E402
import importlib                                        # noqa: E402
sk = importlib.import_module('manga.sketch')

W, H, PPM = 1000, 1250, 8.0
HERE = os.path.dirname(os.path.abspath(__file__))


def arc(c, rx, ry, a0, a1, n=24, rot=0.0):
    """Points on an ellipse from angle a0 to a1 (degrees, 0 = right, 90 = down)."""
    t = np.deg2rad(np.linspace(a0, a1, n))
    cr, sr = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    x, y = rx * np.cos(t), ry * np.sin(t)
    return [(c[0] + cr * a - sr * b, c[1] + sr * a + cr * b) for a, b in zip(x, y)]


def poly_mask(pts):
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in pts], fill=255)
    return np.asarray(im) > 127


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# ---------------------------------------------------------------- 1. two poses as pencil thumbnails
th = Sheet(1000, 560, px_per_mm=PPM, seed=2)
for i, (turn, tilt) in enumerate([(0.3, -5), (-0.25, 6)]):
    cx = 250 + 500 * i
    h = sk.head(th, (cx, 190), 110, turn=turn, tilt=tilt, seed=10 + i)
    for e in 'lr':
        sk.ellipse(th, h['eye_' + e], *h['eye_size_' + e], passes=1, seed=20 + i)
    sk.figure(th, {'neck': (cx + 10 * turn, 360), 'chest': (cx + 20 * turn, 470),
                   'l_shoulder': (cx - 150, 400), 'r_shoulder': (cx + 150, 400)}, 80, seed=30 + i)
th.png(os.path.join(HERE, 'thumbnails.png'), sketch='only')
# chosen: the first. Turned toward the light side of the page, the bangs fall away from the viewer.

# ---------------------------------------------------------------- 2. construction on the page
sh = Sheet(W, H, px_per_mm=PPM, journal=sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'page.jsonl'),
           seed=5)
C, R, TURN, TILT = (500, 470), 205, 0.3, -5
h = sk.head(sh, C, R, turn=TURN, tilt=TILT, seed=1)
ex, ey = h['eye_size']
# neck and shoulders: the neck leaves the jaw under the ear and below the far cheek; shoulders a short way down
chin = h['chin']
NB = chin[1] + 0.62 * R                                # neck base (where the collar sits)
neck_l, neck_r = (chin[0] - 0.55 * R, chin[1] - 0.32 * R), (chin[0] + 0.28 * R, chin[1] - 0.3 * R)
neck_lb, neck_rb = (neck_l[0] - 0.06 * R, NB), (neck_r[0] + 0.06 * R, NB - 4)
sk.curve(sh, [neck_l, lerp(neck_l, neck_lb, 0.5), neck_lb], 0.35, 40)
sk.curve(sh, [neck_r, lerp(neck_r, neck_rb, 0.5), neck_rb], 0.35, 41)
SH_L = [(40, NB + 0.95 * R), (150, NB + 0.32 * R), (neck_lb[0] - 0.2 * R, NB + 0.06 * R), neck_lb]
SH_R = [neck_rb, (neck_rb[0] + 0.25 * R, NB + 0.04 * R), (W - 130, NB + 0.36 * R), (W - 30, NB + 0.95 * R)]
sk.curve(sh, SH_L, 0.35, 42)
sk.curve(sh, SH_R, 0.35, 43)
# hair mass: the cranium plus hair thickness, and the bob's side locks down to the jaw
HC, HR = (C[0] - 6, C[1] - 12), R * 1.13
sk.ellipse(sh, HC, HR, HR * 1.02, rot=TILT, pressure=0.3, passes=1, seed=44)
part = (C[0] - 0.25 * R, C[1] - 0.98 * R)             # parting, on her right (viewer's left of centre)
# rough eyes, iris placement and gaze (slightly toward the viewer: irises shifted toward the near side)
iris = {}
for e in 'lr':
    c = h['eye_' + e]
    rx, ry = h['eye_size_' + e]
    sk.ellipse(sh, c, rx, ry, rot=TILT, passes=1, seed=50 + (e == 'r'))
    iris[e] = ((c[0] - 0.1 * rx, c[1] + 0.05 * ry), 0.62 * rx, 0.95 * ry)
    sk.ellipse(sh, iris[e][0], iris[e][1], iris[e][2], rot=TILT, pressure=0.3, passes=1, seed=52 + (e == 'r'))
# bangs: tips along the brow line, alternating long and short, as gesture strokes from the parting
tips = []
for k, (u, drop) in enumerate([(-0.95, 0.75), (-0.62, 0.42), (-0.3, 0.62), (0.0, 0.3), (0.3, 0.55),
                               (0.6, 0.35), (0.9, 0.8)]):
    base = h['lines']['brow']
    p = base[int(np.clip((u + 0.8) / 1.6 * (len(base) - 1), 0, len(base) - 1))]
    tip = (p[0] + u * 0.12 * R, p[1] + drop * R * 0.45 - 0.1 * R)
    tips.append(tip)
    sk.curve(sh, [part, lerp(part, tip, 0.5), tip], 0.28, 60 + k)
sk_png = os.path.join(HERE, 'sketch.png')
sh.png(sk_png, sketch='only')

# ---------------------------------------------------------------- 3. ink: eyes first
for e in 'lr':
    c = h['eye_' + e]
    rx, ry = h['eye_size_' + e]
    near = (e == 'l')                                   # turn > 0: the viewer's left eye is the near one
    outer = 180 if e == 'l' else 0
    # eye opening: upper lid arc a little flatter than the construction ellipse, lower lid shallow
    up = arc(c, rx * 1.02, ry * 1.0, 190, 350, 20, TILT)
    lo = arc((c[0], c[1] - 0.15 * ry), rx * 0.92, ry * 0.9, 25, 155, 16, TILT)
    opening = poly_mask(up + lo)
    # iris clipped to the opening, black under the lid, tone fading down, pupil and two catch-lights
    ic, irx, iry = iris[e]
    iris_m = poly_mask(arc(ic, irx, iry, 0, 360, 40, TILT)) & opening
    pupil_m = poly_mask(arc((ic[0], ic[1] + 0.05 * iry), 0.3 * irx, 0.42 * iry, 0, 360, 24, TILT))
    lights = (poly_mask(arc((ic[0] - 0.32 * irx, ic[1] - 0.3 * iry), 0.26 * irx, 0.2 * iry, 0, 360, 20)) |
              poly_mask(arc((ic[0] + 0.35 * irx, ic[1] + 0.38 * iry), 0.1 * irx, 0.08 * iry, 0, 360, 12)))
    face.manga_iris(sh, iris_m, pupil=pupil_m, lights=lights, solid=0.3, fade=0.92, peak=0.72, pupil_r=0.16,
                    sparkles=0, rim=0.7, seed=3)
    sh.fill(poly_mask(arc((ic[0], ic[1] + 0.05 * iry), 0.3 * irx, 0.42 * iry, 0, 360, 24, TILT)) & iris_m & ~lights)
    # lash band: a crescent, heaviest at the outer corner, plus a flick
    band_o = arc((c[0], c[1] - 0.06 * ry), rx * 1.08, ry * 1.12, 185, 352, 24, TILT)
    w = np.linspace(1, 0, 24) if e == 'l' else np.linspace(0, 1, 24)
    band_i = [(x, y + (0.16 + 0.22 * wi) * ry) for (x, y), wi in zip(arc(c, rx * 1.0, ry * 1.0, 185, 352, 24, TILT), w)]
    sh.fill(face.ink_shape(sh, poly_mask(band_o + band_i[::-1]), 1.0) if hasattr(face, 'ink_shape') else
            poly_mask(band_o + band_i[::-1]))
    oc = band_o[0] if e == 'l' else band_o[-1]
    dx = -1 if e == 'l' else 1
    sh.stroke([oc, (oc[0] + dx * 0.25 * rx, oc[1] - 0.05 * ry), (oc[0] + dx * 0.42 * rx, oc[1] - 0.22 * ry)],
              pen='g_pen', pressure=0.85, taper=(0.3, 1.6))
    # crease above the lid, lower lid as a short light stroke, inner corner tick
    sh.stroke(arc((c[0], c[1] - 0.32 * ry), rx * 0.95, ry * 0.9, 205 if e == 'l' else 240,
                  300 if e == 'l' else 335, 10, TILT), pen='maru', pressure=0.45)
    sh.stroke(arc((c[0], c[1] - 0.15 * ry), rx * 0.92, ry * 0.9, 50, 120, 10, TILT)[::-1], pen='maru',
              pressure=0.5, taper=(1.0, 1.0))
    # brow: one thin stroke, thick at the head (toward the centre line)
    b = h['brow_' + e]
    bpts = arc((b[0], b[1] + 0.25 * R), rx * 1.25, 0.32 * R, 235 if e == 'l' else 245, 295 if e == 'l' else 305,
               10, TILT)
    if e == 'l':
        bpts = bpts[::-1]
    sh.stroke(bpts, pen='g_pen', pressure=0.55, taper=(0.2, 2.5))

# ---------------------------------------------------------------- 4. face: contour, nose, mouth, ear
jaw = h['jaw']
lo_jaw = [p for p in jaw if p[1] > h['eye_l'][1] + 0.3 * R]
sh.stroke(lo_jaw, pen='g_pen', pressure=0.7, taper=(1.5, 2.0))
n = h['nose']
sh.stroke([(n[0] + 0.05 * R, n[1] - 0.06 * R), (n[0] + 0.09 * R, n[1] + 0.02 * R), (n[0] + 0.03 * R, n[1] + 0.04 * R)],
          pen='maru', pressure=0.55, taper=(0.6, 0.4))
m = h['mouth']
sh.stroke([(m[0] - 0.2 * R, m[1] - 0.05 * R), (m[0] - 0.03 * R, m[1] + 0.035 * R), (m[0] + 0.14 * R, m[1] - 0.03 * R)],
          pen='maru', pressure=0.7, taper=(0.6, 0.6))
sh.stroke([(m[0] - 0.04 * R, m[1] + 0.11 * R), (m[0] + 0.03 * R, m[1] + 0.11 * R)], pen='maru', pressure=0.35)
if h['ear_l'] is not None:
    el = h['ear_l']
    sh.stroke(arc(el, 0.11 * R, 0.22 * R, 100, 280, 14, TILT)[::-1], pen='g_pen', pressure=0.55)
    sh.stroke(arc((el[0] + 0.02 * R, el[1]), 0.05 * R, 0.12 * R, 120, 260, 10, TILT)[::-1], pen='maru', pressure=0.4)
face.blush(sh, (h['eye_l'][0] - 0.05 * R, h['eye_l'][1] + 0.45 * R), 0.16 * R, 0.06 * R, n=5, seed=1)
face.blush(sh, (h['eye_r'][0] + 0.02 * R, h['eye_r'][1] + 0.42 * R), 0.1 * R, 0.05 * R, n=3, seed=2)

# ---------------------------------------------------------------- 5. hair
# The bob frames the face: its side locks follow the face contour a little outside it, down to the jaw.
def Lp(u, v):
    """A point in head units (u across, v down, in cranium radii from the centre), leaning with the head."""
    t = np.deg2rad(TILT)
    return (C[0] + R * (u * np.cos(t) - v * np.sin(t)), C[1] + R * (u * np.sin(t) + v * np.cos(t)))
crown = arc(HC, HR, HR * 1.02, 152, 384, 48, TILT)
# near lock (viewer's left) covers the side of the head down to the jaw; far lock is a thin edge past the cheek
out_l = [crown[0], Lp(-1.13, 0.55), Lp(-1.02, 1.05), Lp(-0.8, 1.36)]
in_l = [Lp(-0.66, 0.2), Lp(-0.64, 0.6), Lp(-0.62, 1.0), Lp(-0.8, 1.36)]
out_r = [crown[-1], Lp(1.13, 0.5), Lp(1.08, 0.95), Lp(0.93, 1.2)]
in_r = [Lp(0.86, 0.25), Lp(0.9, 0.62), Lp(0.93, 1.2)]
low_l, low_r = out_l[-1], out_r[-1]
# bangs edge: tips with notches between them pulled up toward the parting
edge = [tips[0]]
valleys = []
for a, b in zip(tips, tips[1:]):
    v = lerp(lerp(a, b, 0.5), part, 0.38)
    valleys.append(v)
    edge += [v, b]
hair_poly = out_l[::-1] + crown + out_r + in_r[::-1] + edge[::-1] + in_l
hair_m = poly_mask(hair_poly)
sh.fill(hair_m, mode='white')
tone(sh, hair_m, level=0.24, lpi=60)
# the ring of light across the crown, with a jagged lower edge (scraped back to paper)
ring = []
for k, a in enumerate(np.linspace(196, 336, 15)):
    ring.append(arc((HC[0], HC[1] + 8), HR * 0.76, HR * 0.74, a, a, 1)[0])
for k, a in enumerate(np.linspace(336, 196, 29)):
    rr = 0.7 if k % 2 == 0 else 0.64
    ring.append(arc((HC[0], HC[1] + 8), HR * rr, HR * (rr - 0.02), a, a, 1)[0])
sh.fill(poly_mask(ring) & hair_m, mode='white')
# silhouette and lock edges
sh.stroke(out_l[::-1] + crown[1:-1] + out_r, pen='g_pen', pressure=0.8, taper=(2.5, 2.5))
sh.stroke([tips[0]] + in_l, pen='g_pen', pressure=0.6, taper=(2.0, 1.5))
sh.stroke([tips[-1]] + in_r, pen='g_pen', pressure=0.6, taper=(2.0, 1.5))
for k, t in enumerate(tips):
    for v in ([valleys[k - 1]] if k > 0 else []) + ([valleys[k]] if k < len(valleys) else []):
        bow = lerp(lerp(v, t, 0.5), part, -0.04)
        sh.stroke([lerp(v, part, 0.3), v, bow, t], pen='g_pen', pressure=0.5, taper=(2.0, 3.0))
# a few flow strands from the parting, curving with the skull, stopping short of the edge
for k, a in enumerate([168, 186, 204, 334, 352, 370]):
    end = arc(HC, HR * 0.9, HR * 0.92, a, a, 1)[0]
    mid = arc(HC, HR * 0.55, HR * 0.6, (a + 270) / 2, (a + 270) / 2, 1)[0]
    sh.stroke([lerp(part, mid, 0.3), lerp(mid, end, 0.3), end], pen='maru', pressure=0.42, taper=(3.0, 3.0))
for p, q in ((in_l, out_l), (in_r, out_r)):
    for t in (0.35, 0.65):
        a, b = lerp(p[0], q[1], t), lerp(p[-1], q[-1], t)
        sh.stroke([a, lerp(a, b, 0.5), b], pen='maru', pressure=0.38, taper=(3.0, 3.0))

# ---------------------------------------------------------------- 6. neck, shoulders, sailor collar
sh.stroke([neck_l, lerp(neck_l, neck_lb, 0.5), neck_lb], pen='g_pen', pressure=0.6, taper=(2.0, 0.5))
sh.stroke([neck_r, lerp(neck_r, neck_rb, 0.5), neck_rb], pen='g_pen', pressure=0.55, taper=(2.0, 0.5))
# shadow the jaw casts on the neck: the band under the jaw line
under = [p for p in jaw if neck_l[0] - 10 <= p[0] <= neck_r[0] + 10]
low = max(p[1] for p in under) + 0.12 * R
shadow = poly_mask(under + [(p[0], max(p[1] + 0.1 * R, low - 0.06 * R * (1 - np.sin(np.pi * i / max(len(under) - 1, 1)))))
                            for i, p in enumerate(under)][::-1])
neck_m = poly_mask([neck_l, neck_lb, neck_rb, neck_r, chin])
tone(sh, shadow & neck_m, level=0.34, lpi=60)
# sailor collar: two lapels from the neck to a V on the chest, a stripe inside each, and a small knot
V = (chin[0] - 0.05 * R, NB + 0.85 * R)
lap_l = [neck_lb, (neck_lb[0] - 0.42 * R, NB + 0.12 * R), (V[0] - 0.42 * R, V[1] + 0.05 * R), V]
lap_r = [neck_rb, (neck_rb[0] + 0.42 * R, NB + 0.1 * R), (V[0] + 0.42 * R, V[1] + 0.05 * R), V]
col_m = poly_mask([neck_lb, (neck_lb[0] - 0.42 * R, NB + 0.12 * R), (V[0] - 0.42 * R, V[1] + 0.05 * R), V,
                   (V[0] + 0.42 * R, V[1] + 0.05 * R), (neck_rb[0] + 0.42 * R, NB + 0.1 * R), neck_rb, V])
tone(sh, col_m & ~poly_mask([neck_lb, V, neck_rb]), level=0.48, lpi=60)
sh.stroke(SH_L, pen='saji', pressure=0.7, taper=(0.0, 0.5))
sh.stroke(SH_R, pen='saji', pressure=0.7, taper=(0.5, 0.0))
for lap in (lap_l, lap_r):
    sh.stroke(lap[1:], pen='saji', pressure=0.65, taper=(0.5, 0.3))
    sh.stroke([lap[0], V], pen='saji', pressure=0.6, taper=(0.5, 0.3))
    a, b = lerp(lap[1], lap[0], 0.25), lerp(lap[2], V, 0.25)
    sh.stroke([a, b], pen='white', pressure=0.35, mode='white', taper=(0.5, 0.5))
# scarf: a dark knot at the V and two short tails in the collar's tone
tails = []
for d in (-1, 1):
    tails.append([(V[0] + d * 0.05 * R, V[1]), (V[0] + d * 0.2 * R, V[1] + 0.4 * R),
                  (V[0] + d * 0.08 * R, V[1] + 0.5 * R), (V[0] - d * 0.01 * R, V[1] + 0.08 * R)])
for t in tails:
    m_t = poly_mask(t)
    sh.fill(m_t, mode='white')
    tone(sh, m_t, level=0.48, lpi=60)
    sh.stroke(t + [t[0]], pen='saji', pressure=0.55, taper=(0, 0))
knot = poly_mask(arc((V[0], V[1] + 0.01 * R), 0.1 * R, 0.08 * R, 0, 360, 20))
sh.fill(knot)
effects.border(sh, [(30, 30), (W - 30, 30), (W - 30, H - 30), (30, H - 30)], width_mm=0.8)

sh.png(os.path.join(HERE, 'sketch-under-ink.png'), sketch=True)
sh.png(os.path.join(HERE, 'page.png'))
sh.close()
print('strokes', sh.stats['strokes'])
