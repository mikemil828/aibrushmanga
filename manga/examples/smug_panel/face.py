"""Mio's face: eyes, brows, mouth, nose, blush, freckles, outline. Points are panel px (1600 x 1200)."""
import numpy as np
from manga import tone, geom

FACE = [(505, 560), (510, 680), (540, 775), (612, 850), (735, 885), (852, 862), (935, 795), (982, 700), (995, 560)]
# eyes: (centre x, centre y, half width, half height); lid_drop = how far the heavy upper lid comes down
EYES = [dict(c=(648, 632), w=86, h=84, side=-1), dict(c=(878, 626), w=86, h=84, side=1)]
LOOK = (22, 6)                  # irises turned toward the reader's right and a little down
LID_DROP = 0.42                 # the lid covers this much of the eye's height: the smug half-lid


def _P(pts):
    return np.array(pts, float)


def eye(sh, e):
    cx, cy = e['c']; w, h = e['w'], e['h']; sd = e['side']
    # eye opening: almond, outer corner a little higher (sd: -1 left eye, its outer corner is on the left)
    t = np.linspace(0, np.pi, 40)
    xs = cx - w * np.cos(t) * sd * -1
    top_y = cy - h * 0.15 + (h * LID_DROP) * 0.2 - 0.0 * t      # flat, lowered upper lid
    lid = np.column_stack([cx + w * np.cos(t) * (-1), cy - h * (1 - LID_DROP) * np.sin(t) ** 0.75 * 0.55])
    low = np.column_stack([cx + w * 0.92 * np.cos(t) * (-1), cy + h * 0.62 * np.sin(t) ** 0.8])
    # tilt the outer corner up
    def tilt(P):
        u = (P[:, 0] - cx) / w * sd
        P = P.copy(); P[:, 1] -= 8 * np.clip(u, 0, 1); return P
    lid, low = tilt(lid), tilt(low)
    opening = np.vstack([lid, low[::-1]])
    sh.fill([tuple(p) for p in opening], mode='white')
    # iris + pupil, clipped to the opening
    from PIL import Image, ImageDraw
    W0, H0 = int(round(sh.W / sh.k)), int(round(sh.H / sh.k))
    im = Image.new('L', (W0, H0), 0); ImageDraw.Draw(im).polygon([tuple(p) for p in opening], fill=1)
    om = np.asarray(im, bool)
    ix, iy = cx + LOOK[0], cy + LOOK[1]
    rx, ry = 0.56 * w, 0.92 * h
    yy, xx = np.mgrid[0:H0, 0:W0]
    d = ((xx - ix) / rx) ** 2 + ((yy - iy) / ry) ** 2
    iris = (d < 1) & om
    tone(sh, iris, level=0.62, lpi=60)
    # darker top half of the iris (the lid's shadow), a lighter crescent at the bottom
    tone(sh, iris & (yy < iy - 0.1 * ry), level=0.35, lpi=60)
    pd = ((xx - ix) / (0.42 * rx)) ** 2 + ((yy - iy) / (0.5 * ry)) ** 2
    sh.fill(om & (pd < 1))
    sh.fill(iris & (d > 0.82))                                     # iris rim
    lite = om & ((yy - (iy + 0.55 * ry)) ** 2 / (0.22 * ry) ** 2 + (xx - ix) ** 2 / (0.6 * rx) ** 2 < 1) & iris
    sh.fill(lite & ~(pd < 1), mode='white')
    tone(sh, lite & ~(pd < 1), level=0.15, lpi=60)
    # catch-lights: one big oval upper left of the pupil, one small dot lower right
    for (dx, dy, a, b) in [(-0.36, -0.02, 0.30, 0.20), (0.40, 0.42, 0.11, 0.08)]:
        hx, hy = ix + dx * rx, iy + dy * ry
        sh.fill(om & (((xx - hx) / (a * rx)) ** 2 + ((yy - hy) / (b * ry)) ** 2 < 1), mode='white')
    # heavy upper lid: a black band above the lid line, thick in the middle, a wing past the outer corner
    band = []
    for i, (x, y) in enumerate(lid):
        u = i / (len(lid) - 1)
        uo = u if sd > 0 else 1 - u                      # 0 at the inner corner, 1 at the outer
        band.append((x, y - (3 + 22 * np.sin(np.pi * (0.15 + 0.7 * uo)) ** 0.6 * (0.4 + 0.6 * uo))))
    outer = lid[0] if sd < 0 else lid[-1]
    wing = (outer[0] + sd * 22, outer[1] - 10)
    poly = list(map(tuple, lid)) + ([wing] if sd > 0 else []) + band[::-1] + ([wing] if sd < 0 else [])
    sh.fill(poly)
    # lashes: three short strokes off the outer end
    for k in range(3):
        a = np.array(outer) + np.array([sd * (4 + 5 * k), -12 - 3 * k])
        b = a + np.array([sd * (16 - 3 * k), -9 + 5 * k])
        sh.stroke([tuple(a), tuple(b)], pen='g_pen', pressure=0.5, taper=(0, 0.6), smooth=False)
    # lid crease above: the half-lid fold, a thin line
    crease = band[::-1] if sd < 0 else band
    cr = np.array(crease)[4:-6] + [0, -12]
    sh.stroke(np.column_stack([cr, np.full(len(cr), 0.35)]), pen='maru', taper=(0.6, 0.6))
    # lower lid: a short thin line on the outer half, not closing the eye
    lw = low[10:34] if sd > 0 else low[6:30]
    sh.stroke(np.column_stack([lw, np.full(len(lw), 0.35)]), pen='maru', taper=(0.6, 0.6))


def brows(sh):
    # left brow raised and arched, right brow flatter and lower: the smug lift
    sh.stroke([(575, 515, 0.15), (612, 494, 0.45), (660, 490, 0.4), (700, 500, 0.15)], pen='brush', taper=(0.6, 1.0))
    sh.stroke([(822, 530, 0.15), (865, 522, 0.4), (912, 524, 0.4), (950, 537, 0.15)], pen='brush', taper=(0.6, 1.0))


def mouth(sh):
    # lopsided closed smirk: right corner up, a tick where the cheek bunches; a hint of a fang on the left
    m = [(688, 790, 0.25), (720, 800, 0.6), (760, 803, 0.8), (800, 795, 0.7), (832, 775, 0.5), (846, 760, 0.2)]
    sh.stroke(m, pen='g_pen', taper=(0.5, 0.8))
    sh.stroke([(842, 748, 0.2), (852, 760, 0.5), (850, 772, 0.2)], pen='maru', taper=(0.4, 0.4))
    sh.stroke([(686, 784, 0.2), (684, 793, 0.4)], pen='maru', taper=(0.3, 0.3))
    # small fang poking down from the smile on her left
    sh.fill([(708, 797), (722, 800), (714, 812)], mode='white')
    sh.stroke([(708, 797), (714, 812), (722, 800)], pen='maru', pressure=0.4, taper=(0, 0), smooth=False)


def nose(sh):
    sh.stroke([(770, 700, 0.2), (776, 712, 0.5), (770, 716, 0.2)], pen='maru', taper=(0.3, 0.3))


def blush(sh):
    rng = np.random.default_rng(5)
    for cx, cy in [(600, 735), (915, 725)]:
        for i in range(7):
            x = cx - 48 + i * 15 + rng.uniform(-2, 2)
            sh.stroke([(x, cy - 12, 0.2), (x - 9, cy + 12, 0.5)], pen='maru', taper=(0.5, 0.5), smooth=False)
        for dx, dy in [(-30, -18), (-8, -26), (14, -20)]:
            sh.fill([(cx + dx + 3 * np.cos(a), cy + dy - 12 + 3 * np.sin(a)) for a in np.linspace(0, 6.3, 12)])


def outline(sh):
    P = geom.catmull(_P(FACE), per=10)
    n = len(P)
    pr = 0.45 + 0.4 * np.sin(np.linspace(0, np.pi, n)) ** 2       # heavier round the jaw
    sh.stroke(np.column_stack([P, pr]), pen='g_pen', taper=(0.4, 0.4))


def fill_face(sh):
    sh.fill([tuple(p) for p in geom.catmull(_P(FACE), per=10)] + [(995, 400), (505, 400)], mode='white')
    outline(sh)


def features(sh):
    for e in EYES:
        eye(sh, e)
    brows(sh); nose(sh); mouth(sh); blush(sh)
