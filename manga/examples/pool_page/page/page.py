"""A manga page drawn from the pool reference: an introduction page for the character.

Decided before drawing:
  * B4 original, 257 x 364 mm at 8 px/mm. Left column: two small panels; right: one tall panel, the page's anchor.
  * Panel 1 (top left): where we are. Palm, umbrella and loungers by the water, all in light maru line,
    sky in a soft gradient tone, water in one light tone with ripples scraped out. Caption box.
  * Panel 2 (bottom left): her face up close, the emotional beat. Background dropped for a flash (uni-flash),
    the face drawn in full: hand-drawn eyes, black hair with gloss, blush hatching. Speech balloon.
  * Panel 3 (right, tall): the full figure, waving. G-pen contour, maru inner lines, two-tone cel shading per
    material (skin, swimsuit, scales), black hair. The pool scene behind her stays light and sparse.
"""
import sys, time, numpy as np
_here = __import__('os').path.dirname(__file__); sys.path.insert(0, __import__('os').path.join(_here, '..', '..', '..', '..')); sys.path.insert(0, _here)
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
from manga import Sheet, tone, trace, prep, effects, geom
import figure

W, H, PPM = 2056, 2912, 8.0
t0 = time.time()
sh = Sheet(W, H, px_per_mm=PPM, journal=sys.argv[1] if len(sys.argv) > 1 else 'page.jsonl', seed=3)
rows = effects.layout(W, H, [(1.0, [0.78, 1.22])], margin_mm=11, gutter_mm=(4, 6), px_per_mm=PPM)
L, R = rows[0]
lx0, ly0 = L[0]; lx1, ly1 = L[2]
rx0, ry0 = R[0]; rx1, ry1 = R[2]
gy = 6 * PPM
hh = (ly1 - ly0 - 2 * gy)
y1a = ly0 + hh * 0.36; y2a = y1a + gy; y2b = y2a + hh * 0.25; y3a = y2b + gy
P1 = [(lx0, ly0), (lx1, ly0), (lx1, y1a), (lx0, y1a)]
P4 = [(lx0, y2a), (lx1, y2a - 40), (lx1, y2b - 40), (lx0, y2b)]        # the wave: a slanted strip
P2 = [(lx0, y3a), (lx1, y3a - 40), (lx1, ry1), (lx0, ry1)]
P3 = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]


def ibox(p):
    xs, ys = [q[0] for q in p], [q[1] for q in p]
    return int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))


def pmask(p):
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon([tuple(q) for q in p], fill=1)
    return np.asarray(im, bool)


def place(a, box):
    X0, Y0, X1, Y1 = box
    out = np.zeros((H, W), a.dtype); out[Y0:Y1, X0:X1] = a; return out


def background(box, crop, light=0.55, min_len=18, sky_to=None, water=None):
    """The scene, drawn lightly: pale area tones, thin maru lines with the scraps left out."""
    rgb, mat, to_page, s = figure.source(box, crop)
    d = 1 - rgb.mean(2)
    fig = nd.binary_dilation(mat > 0, iterations=int(2 * s))
    lines = prep.xdog_lines(rgb, sigma=0.9 * s, p=24, eps=-0.015, phi=80, tidy=int(12 * s * s)) > 0.5
    lines = nd.binary_closing(lines, iterations=max(1, int(s / 1.2)))
    lines &= ~fig
    lv = np.minimum(trace.area_tones(d, nd.binary_dilation(lines, iterations=1)), 0.75) * light
    lv[fig] = 0
    if sky_to is not None:                       # sky: a soft gradient tone, darker at the top
        yy = np.arange(lv.shape[0])[:, None] + np.zeros((1, lv.shape[1]))
        ys = np.array([to_page(0, 0)[1], to_page(0, sky_to)[1]]) - box[1]
        sky = (yy < ys[1]) & (nd.gaussian_filter(d, 2) < 0.35) & ~fig
        lv = np.where(sky, np.clip(0.30 - 0.26 * (yy - ys[0]) / max(ys[1] - ys[0], 1), 0.04, 0.30), lv)
    if water is not None:                        # the pool: one light tone, ripples scraped out below
        wy0, wy1 = to_page(0, water[0])[1] - box[1], to_page(0, water[1])[1] - box[1]
        yy = np.arange(lv.shape[0])[:, None] + np.zeros((1, lv.shape[1]))
        wat = (yy > wy0) & (yy < wy1) & ~fig
        lv = np.where(wat, 0.30, lv)
    tone(sh, place(lv > 0, box), level=place(lv.astype(np.float32), box), lpi=figure.LPI)
    if water is not None:
        gw = nd.gaussian_filter(rgb.mean(2), 1.2 * s)
        rip = wat & (gw - nd.gaussian_filter(gw, 8 * s) > 0.06)
        sh.fill(place(nd.binary_opening(rip, iterations=1), box), mode='white')
    n, _ = trace.ink_line_art(sh, place(lines, box), pens=('maru', 'maru'), weight=0.9, min_len_px=min_len * s,
                              min_px=1.0, seed=box[0] + box[1])
    return n, fig


def balloon(center, rx, ry, text, tail, size_mm=5.0):
    cx, cy = center
    t = np.linspace(0, 2 * np.pi, 120)
    # tail: a narrow wedge from the balloon's edge toward the speaker
    ang = np.arctan2(tail[1] - cy, tail[0] - cx)
    a0, a1 = ang - 0.16, ang + 0.16
    e0 = (cx + rx * np.cos(a0), cy + ry * np.sin(a0)); e1 = (cx + rx * np.cos(a1), cy + ry * np.sin(a1))
    shape = [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t if not (a0 < a < a1 or a0 < a - 2 * np.pi < a1)]
    sh.fill([(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t], mode='white')
    sh.fill([e0, tuple(tail), e1], mode='white')
    ell = np.array([(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in np.linspace(a1, a0 + 2 * np.pi, 140)])
    sh.stroke(np.vstack([[tail], ell, [tail]]), pen='g_pen', pressure=0.35, taper=(0, 0), smooth=False)
    f = effects._font(size_mm * PPM)
    lines = text.split('\n')
    tmp = ImageDraw.Draw(Image.new('L', (1, 1)))
    bbs = [tmp.textbbox((0, 0), ln, font=f) for ln in lines]
    lh = size_mm * PPM * 1.2
    y = cy - lh * len(lines) / 2
    for ln, bb in zip(lines, bbs):
        effects.caption(sh, (cx - (bb[2] - bb[0]) / 2 - 0.5, y - bb[1] * 0.5), ln, size_mm=size_mm, pad_mm=0, box=False)
        y += lh


strokes = 0
# ---- Panel 1: the place
b1 = ibox(P1)
with sh.clip(pmask(P1)):
    import panel1
    s1 = (b1[3] - b1[1]) / (1085 - 650)                    # hand-drawn: frame ref y 650-1085, x centred on the pole
    tp1 = lambda rx, ry: (b1[0] + (b1[2] - b1[0]) / 2 + (rx - 112) * s1, b1[1] + (ry - 650) * s1)
    panel1.draw_panel1(sh, b1, tp1, s1, W, H)
effects.border(sh, P1, 0.8)
effects.caption(sh, (b1[0] + 3 * PPM, b1[1] + 3 * PPM), 'Resort pool.\n2 p.m.', size_mm=3.8)

# ---- Panel 3: the full figure
b3 = ibox(P3)
with sh.clip(pmask(P3)):
    n, fig3 = background(b3, (130, 0, 1035, 1920), light=0.5, min_len=34, sky_to=880, water=(1090, 1530))
    strokes += n
    sh.fill(place(figure.source(b3, (130, 0, 1035, 1920))[1] > 0, b3), mode='white')
    r3 = figure.draw_figure(sh, b3, (130, 0, 1035, 1920))
    strokes += r3['strokes']
effects.border(sh, P3, 0.8)

# ---- Panel 2: the face, with a flash behind it
b2 = ibox(P2)
CROP2 = (300, 150, 720, 760)
with sh.clip(pmask(P2)):
    rgb2, mat2, tp2, s2 = figure.source(b2, CROP2)
    c = tp2(510, 400)
    effects.flash(sh, c, r_in=0.42 * (b2[2] - b2[0]), r_out=1.2 * (b2[3] - b2[1]), count=150, seed=4)
    sh.fill(place(nd.binary_dilation(mat2 > 0, iterations=3), b2), mode='white')
    r2 = figure.draw_figure(sh, b2, CROP2, dict(contour_w=2.6))
    strokes += r2['strokes']
    balloon((b2[0] + 0.30 * (b2[2] - b2[0]), b2[1] + 0.12 * (b2[3] - b2[1])), 0.22 * (b2[2] - b2[0]),
            0.075 * (b2[3] - b2[1]), 'Over\nhere!!', tail=tp2(470, 330), size_mm=4.6)
effects.border(sh, P2, 0.8)

# ---- Panel 4 (middle left): the wave, hand up close, with motion marks
b4 = ibox(P4)
CROP4 = (680, 250, 1000, 560)
with sh.clip(pmask(P4)):
    rgb4, mat4, tp4, s4 = figure.source(b4, CROP4)
    effects.speed_lines(sh, b4, angle=-12, count=70, length=(0.3, 0.8), seed=6)
    sh.fill(place(nd.binary_dilation(mat4 > 0, iterations=3), b4), mode='white')
    r4 = figure.draw_figure(sh, b4, CROP4, dict(contour_w=2.6, blush=False))
    strokes += r4['strokes']
    # wave marks: short arcs either side of the fingers, the shorthand for a hand in motion
    for (cx, cy, sgn) in [(735, 330, -1), (925, 345, 1)]:
        for k, rr in enumerate((26, 40, 54)):
            a = np.linspace(-0.55, 0.55, 12)
            pts = [tp4(cx + sgn * rr * np.cos(t) - sgn * 0, cy + rr * np.sin(t)) for t in a]
            sh.stroke(pts, pen='g_pen', pressure=0.5, taper=(0.6, 0.9), seed=40 + k)
effects.border(sh, P4, 0.8)

sh.png('page.png'); sh.close()
print('strokes', strokes + sh.stats.get('strokes', 0) * 0, 'sheet strokes', sh.stats['strokes'], round(time.time() - t0), 's')
