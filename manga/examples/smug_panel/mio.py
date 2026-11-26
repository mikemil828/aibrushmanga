"""Mio, an original character: the hair, pigtails, neck and striped shirt round the face from face.py."""
import numpy as np
from manga import tone, geom


def C(pts, per=10, closed=False):
    P = np.array(pts, float)
    if closed:
        P = np.vstack([P, P[:2]])
        out = geom.catmull(P, per=per)
        return out[: -per]
    return geom.catmull(P, per=per)


OUTER = [(470, 712), (450, 570), (458, 440), (505, 318), (598, 228), (720, 184), (842, 186), (952, 228),
         (1040, 315), (1082, 440), (1086, 572), (1064, 712)]
# the bangs' lower edge, left to right: a point down at each strand tip
BANGS = [(1064, 712), (1030, 735), (1002, 640), (992, 530), (980, 478), (966, 488), (950, 452), (912, 456),
         (892, 482), (870, 450), (826, 446), (806, 474), (784, 444), (742, 446), (722, 476), (700, 448),
         (660, 452), (638, 480), (618, 456), (582, 462), (566, 486), (546, 470), (522, 540), (512, 650),
         (500, 740)]
BOBBLES = [(452, 470, 27), (1086, 470, 27)]
TAILS = [[(440, 482), (392, 500), (350, 548), (330, 615), (345, 680), (372, 710), (380, 668), (398, 690),
          (402, 640), (420, 650), (418, 590), (440, 540)],
         [(1098, 482), (1146, 500), (1188, 548), (1208, 615), (1193, 680), (1166, 710), (1158, 668), (1140, 690),
          (1136, 640), (1118, 650), (1120, 590), (1098, 540)]]
CROWN = (775, 150)


def hair(sh):
    poly = [tuple(p) for p in C(OUTER, 8)] + [tuple(p) for p in C(BANGS, 6)]
    sh.fill(poly)
    # gloss: a ring of white streaks across the crown (an ellipse round the head's curve), each streak along
    # the hair's flow, longer in the middle where the light hits square on
    rng = np.random.default_rng(11)
    from PIL import Image, ImageDraw
    im = Image.new('L', (int(sh.W / sh.k), int(sh.H / sh.k)), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=1)
    hm = np.asarray(im, bool)
    with sh.clip(hm):
        for t in np.linspace(0.08, 0.92, 17):
            x = 520 + 510 * t
            yc = 318 - 70 * np.sin(np.pi * t) ** 0.5 + 40 * (1 - np.sin(np.pi * t))
            L = (22 + 30 * np.sin(np.pi * t)) * rng.uniform(0.7, 1.15)
            dx = (x - CROWN[0]) * 0.12
            sh.stroke([(x - dx * 0.5, yc - L * 0.5), (x + dx * 0.5, yc + L * 0.5)], pen='white',
                      pressure=rng.uniform(0.55, 0.9), mode='white', taper=(1.6, 1.6), smooth=False)
    # strand lines: thin white lines from the crown down into the bangs and the side locks
    for t in np.linspace(0.05, 0.95, 13):
        tx = 540 + 450 * t + rng.uniform(-10, 10)
        a = np.array([CROWN[0] + (tx - CROWN[0]) * 0.25, CROWN[1] + 60])
        b = np.array([tx, 430 + rng.uniform(-6, 12)])
        mid = (a + b) / 2 + [(tx - CROWN[0]) * 0.18, 0]
        pts = [tuple(a + (b - a) * 0.15), tuple(mid), tuple(b - (b - a) * 0.12)]
        sh.stroke(pts, pen='white', pressure=0.06, mode='white', taper=(1.5, 1.5))
    # the outline: G-pen round the silhouette (on black it shows only where it leaves the fill)
    sil = C(OUTER, 8)
    sh.stroke(np.column_stack([sil, np.full(len(sil), 0.7)]), pen='g_pen', taper=(0.3, 0.3))
    # pigtails: puffs below round bobbles
    for tail in TAILS:
        P = C(tail, 8, closed=True)
        sh.fill([tuple(p) for p in P])
        base = np.array(tail[0])
        for k in range(5):
            tip = np.array(tail[2 + 2 * k if 2 + 2 * k < len(tail) else -1])
            m = (base + tip) / 2 + rng.uniform(-6, 6, 2)
            sh.stroke([tuple(base + (tip - base) * 0.2), tuple(m), tuple(base + (tip - base) * 0.8)], pen='white',
                      pressure=0.08, mode='white', taper=(1.2, 1.2))
    for bx, by, r in BOBBLES:
        t = np.linspace(0, 2 * np.pi, 40)
        sh.fill([(bx + r * np.cos(a), by + r * np.sin(a)) for a in t], mode='white')
        sh.stroke(np.column_stack([bx + r * np.cos(t), by + r * np.sin(t), np.full(40, 0.6)]), pen='g_pen',
                  taper=(0, 0))
        # the bobble's own shading: dark tone with a white shine top left
        yy, xx = np.mgrid[0:sh.H // sh.k if hasattr(sh, 'k') else sh.H, 0:sh.W // sh.k]
        disk = (xx - bx) ** 2 + (yy - by) ** 2 < (r - 2) ** 2
        tone(sh, disk, level=0.5, lpi=60)
        sh.fill(((xx - bx + 9) ** 2 + (yy - by + 9) ** 2 < 7 ** 2), mode='white')


NECK = [(705, 870), (708, 940), (828, 935), (826, 860)]
SHIRT = [(340, 1200), (360, 1090), (430, 1020), (560, 975), (680, 938), (768, 956), (852, 935), (980, 972),
         (1110, 1020), (1175, 1090), (1195, 1200)]


def body(sh, W, H):
    from PIL import Image, ImageDraw
    def m(pts):
        im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=1)
        return np.asarray(im, bool)
    sh.fill(NECK, mode='white')
    # neck shadow under the chin
    tone(sh, m(list(map(tuple, C([(705, 868), (768, 888), (826, 858), (826, 905), (768, 922), (707, 908)], 6, closed=True)))), level=0.28, lpi=60)
    sh.stroke([(705, 868, 0.4), (707, 905, 0.5), (709, 940, 0.3)], pen='maru', taper=(0.3, 0.3))
    sh.stroke([(826, 858, 0.4), (827, 900, 0.5), (828, 935, 0.3)], pen='maru', taper=(0.3, 0.3))
    shirt = C(SHIRT, 8)
    sh.fill([tuple(p) for p in shirt], mode='white')
    sm = m(shirt)
    # stripes: black bands that sag with the shoulders' curve
    for y0 in (1010, 1072, 1134):
        xs = np.linspace(300, 1240, 60)
        top = y0 + 0.00012 * (xs - 768) ** 2 * -1 + 0.00004 * (xs - 768) ** 2
        band = [(x, y) for x, y in zip(xs, top)] + [(x, y + 26) for x, y in zip(xs[::-1], top[::-1])]
        sh.fill(m(band) & sm)
    # the shirt's turning sides in shadow
    from scipy import ndimage as nd
    dist = nd.distance_transform_edt(sm)
    xx = np.arange(W)[None, :] + np.zeros((H, 1))
    side = sm & (dist < 70) & (np.abs(xx - 768) > 220)
    tone(sh, side, level=0.3, lpi=60)
    sh.stroke(np.column_stack([shirt, np.full(len(shirt), 0.6)]), pen='g_pen', taper=(0, 0))
    # round collar: a ribbed band round the neck
    t = np.linspace(np.pi * 0.05, np.pi * 0.95, 40)
    col_o = np.column_stack([768 - 92 * np.cos(t), 928 + 40 * np.sin(t)])
    col_i = np.column_stack([768 - 70 * np.cos(t), 928 + 22 * np.sin(t)])
    sh.fill([tuple(p) for p in np.vstack([col_o, col_i[::-1]])], mode='white')
    sh.stroke(np.column_stack([col_o, np.full(40, 0.55)]), pen='g_pen', taper=(0.2, 0.2))
    sh.stroke(np.column_stack([col_i, np.full(40, 0.45)]), pen='maru', taper=(0.2, 0.2))
