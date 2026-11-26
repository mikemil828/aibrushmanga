"""The big umbrella behind her in the tall panel, inked by hand from points read off the reference (ref px).

Traced at this size its ribs broke into dashes; a hand inks each rib as one ruled stroke."""
import numpy as np
from manga import geom

HUB = (1047, 145)
# every rib and stretcher as (from, to, half width at each end) in ref px, read off the reference: the long ribs come
# from the canopy's top (off the page, top right), the stretchers from the runner (HUB) on the pole to the ribs
RIBS = [((1025, 0), (695, 263), 3.4, 2.6),      # long rib down to the tip by her hair
        ((930, 0), (540, 98), 3.0, 2.2),        # long rib along the top edge
        ((1040, 140), (827, 25), 3.0, 2.4),     # stretcher up to the top rib
        ((1030, 145), (885, 120), 2.8, 2.4),    # stretcher to the long rib
        ((1040, 135), (910, 0), 2.8, 2.2),
        ((1040, 130), (1005, 0), 2.8, 2.2),
        ((1028, 165), (940, 400), 3.0, 2.4),    # stretcher down to the rib tip at the right
        ((940, 400), (930, 440), 2.6, 2.6)]
RIM = []
POLE = [(1055, 165), (975, 620)]
# the canopy's far edge, a thin line dropping behind her ahoge to her hair
EDGE = [(470.3, -4), (476, 14), (483.6, 32), (486.3, 64), (489, 101)]


def draw_umbrella(sh, to_page, s):
    P = lambda pts: np.array([to_page(x, y) for x, y in pts], float)     # noqa: E731
    hx, hy = to_page(*HUB)
    # ribs: two ruled lines each (a rib has thickness), filled black between
    for p0, p1, w0, w1 in RIBS:
        a, b = np.array(to_page(*p0)), np.array(to_page(*p1))
        d = (b - a) / np.hypot(*(b - a)); n = np.array([-d[1], d[0]])
        w0, w1 = w0 * s, w1 * s
        quad = [tuple(a + n * w0), tuple(b + n * w1), tuple(b - n * w1), tuple(a - n * w0)]
        sh.fill(quad)
        # a thin shine down the rib, off centre toward the light
        q0, q1 = a + d * 30 * s + n * 0.7 * s, b - d * 10 * s + n * 0.5 * s
        h = 0.45 * s
        sh.fill([tuple(q0 + n * h), tuple(q1 + n * h), tuple(q1 - n * h), tuple(q0 - n * h)], mode='white')
    # canopy rim: one long G-pen line with a slight sag between rib tips
    if len(RIM) > 1:
        rim = geom.catmull(P(RIM), per=10)
        sh.stroke(np.column_stack([rim, np.full(len(rim), 0.65)]), pen='g_pen', taper=(0.5, 0))
    e = geom.catmull(P(EDGE), per=8)
    sh.stroke(np.column_stack([e, np.full(len(e), 0.45)]), pen='maru', taper=(0, 0.4))
    # pole: a dark tube with a light edge on the left
    a, b = np.array(to_page(*POLE[0])), np.array(to_page(*POLE[1]))
    d = (b - a) / np.hypot(*(b - a)); n = np.array([-d[1], d[0]])
    w = 6.5 * s
    sh.fill([tuple(a + n * w), tuple(b + n * w), tuple(b - n * w), tuple(a - n * w)])
    sh.stroke([tuple(a - n * w * 0.45), tuple(b - n * w * 0.45)], pen='white', pressure=0.05, mode='white',
              taper=(0, 2.0), smooth=False)
    # hub: a black collar with two rings
    for k, r in enumerate((11 * s, 8 * s)):
        t = np.linspace(0, 2 * np.pi, 40)
        ring = np.column_stack([hx + r * np.cos(t), hy + 0.7 * r * np.sin(t)])
        if k == 0:
            sh.fill([tuple(p) for p in ring])
        else:
            sh.stroke(ring, pen='white', pressure=0.05, mode='white', taper=(0, 0))


def drawn_zone(to_page, s, box):
    """Where the hand-inked ribs, rim and pole lie (box-local mask): traced lines there are left out."""
    from PIL import Image, ImageDraw
    X0, Y0, X1, Y1 = box
    im = Image.new('L', (X1 - X0, Y1 - Y0), 0)
    d = ImageDraw.Draw(im)
    L = lambda p: tuple(np.array(to_page(*p)) - (X0, Y0))     # noqa: E731
    w = int(12 * s)
    for p0, p1, _, _ in RIBS:
        d.line([L(p0), L(p1)], fill=1, width=w)
    if len(RIM) > 1:
        d.line([L(p) for p in RIM], fill=1, width=w)
    d.line([L(POLE[0]), L(POLE[1])], fill=1, width=int(20 * s))
    return np.asarray(im, bool)


# the pale fin tip above her right cheek reads as sky on the page: no traced lines there either
FIN_SKY = [(684, 333), (702, 333), (737, 368), (733, 378), (688, 378)]


def drawn_zone_and_sky(to_page, s, box):
    from PIL import Image, ImageDraw
    X0, Y0, X1, Y1 = box
    im = Image.new('L', (X1 - X0, Y1 - Y0), 0)
    ImageDraw.Draw(im).polygon([tuple(np.array(to_page(*p)) - (X0, Y0)) for p in FIN_SKY], fill=1)
    import bushes3
    return drawn_zone(to_page, s, box) | np.asarray(im, bool) | bushes3.area_zone(to_page, s, box)
